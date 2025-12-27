import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from rouge_score import rouge_scorer
from bert_score import score as bert_score
from src.graph_rag.utils.config_loader import load_config
from src.graph_rag.utils.logger import get_logger
from src.graph_rag.linking.entity_linker import EntityLinker
from src.graph_rag.retrieval.subgraph_retriever import SubgraphRetriever
from src.graph_rag.generator.generate import GraphAnswerGenerator

# --------------------------------------------------
# Setup
# --------------------------------------------------
config = load_config()
logger = get_logger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def encode(texts: List[str]):
    return model.encode(texts, normalize_embeddings=True)

def semantic_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    emb = encode([a, b])
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

def soft_f1(pred: List[str], gold: List[str], threshold=0.5) -> float:
    if not pred or not gold:
        return 0.0

    pred_emb = encode(pred)
    gold_emb = encode([g.replace("_", " ") for g in gold])

    sim = cosine_similarity(pred_emb, gold_emb)

    matched_p, matched_g = set(), set()

    for i in range(len(pred)):
        j = np.argmax(sim[i])
        if sim[i][j] >= threshold and j not in matched_g:
            matched_p.add(i)
            matched_g.add(j)

    p = len(matched_p) / len(pred)
    r = len(matched_g) / len(gold)

    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(predictions, references) if p.strip() and r.strip()]
    return float(np.mean(scores)) if scores else 0.0

def compute_bertscore(predictions: List[str], references: List[str]) -> float:
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if len(p.strip()) > 20 and len(r.strip()) > 20]
    if not valid_pairs:
        return 0.0
    preds, refs = zip(*valid_pairs)
    P, R, F1 = bert_score(list(preds), list(refs), lang="en", rescale_with_baseline=False, device=device)
    return float(F1.mean())

def parse_answer(answer: str) -> Dict:
    problem, causes, actions = "", [], []
    section = None

    for line in answer.splitlines():
        line = line.strip().lower()
        if not line:
            continue
        if "problem" in line:
            section = "problem"
        elif "cause" in line:
            section = "causes"
        elif "action" in line:
            section = "actions"
        elif line.startswith("-"):
            content = line[1:].strip()
            if section == "problem" and not problem:
                problem = content
            elif section == "causes":
                causes.append(content)
            elif section == "actions":
                actions.append(content)

    return {
        "problem": problem,
        "causes": causes,
        "actions": actions
    }

def load_test_data(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8")]
# --------------------------------------------------
# Main evaluation
# --------------------------------------------------
def run_graph_evaluation(test_size: int) -> Dict[str, float]:

    data_path = Path(config["test_data_path"])
    test_data = load_test_data(data_path)[:10]

    linker = EntityLinker()
    retriever = SubgraphRetriever()
    generator = GraphAnswerGenerator()

    metrics = {
        "semantic_similarity": [],
        "cause_f1": [],
        "action_f1": [],
        "faithfulness": [],
        "hallucination": [],
        "latency": [],
        "rouge":[],
        "bert":[],
    }

    for sample in test_data:
        query, gt = sample["question"], sample["ground_truth"]
        question = sample["question"].replace("_", " ")

        linked_problem, _ = linker.link(question)
        subgraph = retriever.retrieve(linked_problem)

        start = time.time()
        answer = generator.generate(question, subgraph)
        latency = time.time() - start

        parsed = parse_answer(answer)

        gold_problem = sample["question"].replace("_", " ")
        gold_causes = gt["possible_causes"]
        gold_actions = gt["recommended_actions"]

        pred_full = (
            f"{parsed['problem']}. "
            f"{' '.join(parsed['causes'])}. "
            f"{' '.join(parsed['actions'])}"
        )

        gold_full = (
            f"{gold_problem}. "
            f"{' '.join(gold_causes)}. "
            f"{' '.join(gold_actions)}"
        )

        sim = semantic_similarity(pred_full, gold_full)

        cause_f1 = soft_f1(parsed["causes"], gold_causes)
        action_f1 = soft_f1(parsed["actions"], gold_actions)

        # Graph RAG faithfulness = did it stay inside graph?
        faith = 1.0 if linked_problem == sample["question"] else 0.5

        rouge = compute_rouge_l(pred_full, gold_full)
        bert = compute_bertscore(pred_full, gold_full)

        metrics["semantic_similarity"].append(sim)
        metrics["cause_f1"].append(cause_f1)
        metrics["action_f1"].append(action_f1)
        metrics["faithfulness"].append(faith)
        metrics["hallucination"].append(1 - faith)
        metrics["latency"].append(latency)
        metrics['rouge'].append(rouge)
        metrics['bert'].append(bert)        

    return {k: float(np.mean(v)) for k, v in metrics.items()}
