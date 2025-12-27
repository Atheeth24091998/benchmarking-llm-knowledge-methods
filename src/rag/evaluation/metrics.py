import json
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from src.rag.utils.logger import get_logger
from src.rag.utils.config_loader import load_config
from src.rag.vectorstore.faiss_store import FaissVectorStore
from src.rag.generator.generate import generate_answer

# --- Config ---
config = load_config()
test_path = Path(config["test_data_path"])
device = "cuda" if torch.cuda.is_available() else "cpu"
logger = get_logger(__name__)
print(f"Using device: {device}")

similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# --- Helpers ---
def encode_texts(texts: List[str]):
    return similarity_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

def semantic_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    emb1, emb2 = encode_texts([text1, text2])
    return float(cosine_similarity([emb1], [emb2])[0][0])

def soft_f1(pred_list: List[str], gold_list: List[str], threshold: float = 0.5) -> float:
    if not pred_list or not gold_list:
        return 0.0
    pred_embs = encode_texts(pred_list)
    gold_embs = encode_texts([g.replace("_", " ") for g in gold_list])
    sim_matrix = cosine_similarity(pred_embs, gold_embs)
    matched_pred, matched_gold = set(), set()
    for i in range(len(pred_list)):
        best_j, best_sim = max(((j, sim_matrix[i, j]) for j in range(len(gold_list)) if j not in matched_gold),
                               key=lambda x: x[1], default=(-1, 0))
        if best_sim >= threshold:
            matched_pred.add(i)
            matched_gold.add(best_j)
    precision = len(matched_pred) / len(pred_list)
    recall = len(matched_gold) / len(gold_list)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

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

def faithfulness_score(pred_text: str, retrieved_chunks: List[Dict]) -> float:
    if not pred_text or not retrieved_chunks:
        return 0.0
    context = " ".join([c["text"] for c in retrieved_chunks])
    sentences = [s.strip() for s in pred_text.split(".") if len(s.strip()) > 15]
    if not sentences:
        return 0.0
    sent_embs = encode_texts(sentences)
    context_emb = encode_texts([context])
    sim = cosine_similarity(sent_embs, context_emb).flatten()
    return sum(s > 0.3 for s in sim) / len(sentences)

def measure_latency(fn, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    return result, time.time() - start

def parse_answer(answer: str) -> Dict:
    problem, causes, actions = "", [], []
    section = None
    for line in answer.splitlines():
        line = line.strip()
        if not line: continue
        if "problem" in line.lower(): section = "problem"
        elif "cause" in line.lower(): section = "causes"
        elif "action" in line.lower(): section = "actions"
        elif line.startswith("-"):
            content = line[1:].strip()
            if section == "problem" and not problem: problem = content
            elif section == "causes" and len(content) > 5: causes.append(content)
            elif section == "actions" and len(content) > 5: actions.append(content)
    return {"problem": problem.lower(), "causes": [c.lower() for c in causes], "actions": [a.lower() for a in actions]}

def load_test_data(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8")]

def run_evaluation(test_size: int) -> Dict[str, float]:
    test_data = load_test_data(test_path)[:test_size]

    store = FaissVectorStore()
    store.load()

    metrics = {
        "cause_f1": [],
        "action_f1": [],
        "semantic_similarity": [],
        "faithfulness": [],
        "hallucination": [],
        "latency": [],
        "rouge":[],
        "bert":[],
    }

    all_preds, all_refs = [], []

    for sample in test_data:
        query, gt = sample["question"], sample["ground_truth"]

        retrieved = store.retrieve(query, top_k=5)
        answer, latency = measure_latency(generate_answer, query, retrieved)
        parsed = parse_answer(answer)

        metrics["cause_f1"].append(
            soft_f1(parsed["causes"], gt["possible_causes"])
        )
        metrics["action_f1"].append(
            soft_f1(parsed["actions"], gt["recommended_actions"])
        )

        pred_full = f"{parsed['problem']}. {' '.join(parsed['causes'])}. {' '.join(parsed['actions'])}"
        gold_full = (
            f"{gt['problem'].replace('_',' ')}. "
            f"{' '.join([c.replace('_',' ') for c in gt['possible_causes']])}. "
            f"{' '.join([a.replace('_',' ') for a in gt['recommended_actions']])}"
        )

        sim = semantic_similarity(pred_full, gold_full)
        faith = faithfulness_score(pred_full, retrieved)

        metrics["semantic_similarity"].append(sim)
        metrics["faithfulness"].append(faith)
        metrics["hallucination"].append(1 - faith)
        metrics["latency"].append(latency)

        all_preds.append(pred_full)
        all_refs.append(gold_full)

    # Optional batch metrics (not shown in Streamlit yet)
    rouge = compute_rouge_l(all_preds, all_refs)
    bert = compute_bertscore(all_preds, all_refs)

    return {
        "semantic_similarity": float(np.mean(metrics["semantic_similarity"])),
        "faithfulness": float(np.mean(metrics["faithfulness"])),
        "hallucination": float(np.mean(metrics["hallucination"])),
        "cause_f1": float(np.mean(metrics["cause_f1"])),
        "action_f1": float(np.mean(metrics["action_f1"])),
        "latency": float(np.mean(metrics["latency"])),
        "rouge_l": rouge,
        "bert_score": bert,
    }


# --- Main ---
def main():
    test_data = load_test_data(test_path)[:10]
    store = FaissVectorStore()
    store.load()

    metrics = {"soft_cause_f1": [], "soft_action_f1": [], "rouge_l": [], "bert_score": [],
               "semantic_similarity": [], "faithfulness": [], "hallucination": [], "latency": []}

    all_preds, all_refs = [], []

    for sample in test_data:
        query, gt = sample["question"], sample["ground_truth"]
        retrieved = store.retrieve(query, top_k=5)
        answer, latency = measure_latency(generate_answer, query, retrieved)
        parsed = parse_answer(answer)

        metrics["soft_cause_f1"].append(soft_f1(parsed["causes"], gt["possible_causes"]))
        metrics["soft_action_f1"].append(soft_f1(parsed["actions"], gt["recommended_actions"]))

        pred_full = f"{parsed['problem']}. {' '.join(parsed['causes'])}. {' '.join(parsed['actions'])}"
        gold_full = f"{gt['problem'].replace('_',' ')}. {' '.join([c.replace('_',' ') for c in gt['possible_causes']])}. {' '.join([a.replace('_',' ') for a in gt['recommended_actions']])}"

        metrics["semantic_similarity"].append(semantic_similarity(pred_full, gold_full))
        metrics["faithfulness"].append(faithfulness_score(pred_full, retrieved))
        metrics["hallucination"].append(1 - metrics["faithfulness"][-1])
        metrics["latency"].append(latency)

        all_preds.append(pred_full)
        all_refs.append(gold_full)

    metrics["rouge_l"] = [compute_rouge_l(all_preds, all_refs)]
    metrics["bert_score"] = [compute_bertscore(all_preds, all_refs)]

    for k, v in metrics.items():
        logger.info(f"{k:20s}: {np.mean(v):.4f}")

if __name__ == "__main__":
    main()
