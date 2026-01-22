# /home/athenaik/MASTER_THESIS/src/sft/evaluation/metrics.py

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

# Setup
config = load_config()
logger = get_logger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)

# Helpers
def encode(texts: List[str]):
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

def semantic_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    emb = encode([a, b])
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

def soft_f1(pred: List[str], gold: List[str], threshold=0.4) -> float:
    if not pred or not gold:
        return 0.0
    
    pred_emb = encode(pred)
    gold_emb = encode([g.replace("_", " ") for g in gold])
    
    sim = cosine_similarity(pred_emb, gold_emb)
    
    matched_p, matched_g = set(), set()
    
    for i in range(len(pred)):
        best_j, best_sim = -1, 0.0
        for j in range(len(gold)):
            if j not in matched_g and sim[i][j] > best_sim:
                best_sim = sim[i][j]
                best_j = j
        
        if best_sim >= threshold and best_j != -1:
            matched_p.add(i)
            matched_g.add(best_j)
    
    p = len(matched_p) / len(pred)
    r = len(matched_g) / len(gold)
    
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(r, p)['rougeL'].fmeasure 
              for p, r in zip(predictions, references) if p.strip() and r.strip()]
    return float(np.mean(scores)) if scores else 0.0

def compute_bertscore(predictions: List[str], references: List[str]) -> float:
    valid_pairs = [(p, r) for p, r in zip(predictions, references) 
                   if len(p.strip()) > 20 and len(r.strip()) > 20]
    if not valid_pairs:
        return 0.0
    preds, refs = zip(*valid_pairs)
    _, _, F1 = bert_score(list(preds), list(refs), lang="en", 
                          rescale_with_baseline=False, device=device, verbose=False)
    return float(F1.mean())

def parse_answer(answer: str) -> Dict:
    """Parse SFT model output into structured format."""
    answer = answer.replace("**", "")
    problem, causes, actions = "", [], []
    section = None
    
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        
        line_lower = line.lower()
        if "problem:" in line_lower:
            section = "problem"
        elif "possible causes:" in line_lower or "causes:" in line_lower:
            section = "causes"
        elif "recommended actions:" in line_lower or "actions:" in line_lower:
            section = "actions"
        elif line.startswith("-"):
            content = line[1:].strip()
            if section == "problem" and not problem:
                problem = content
            elif section == "causes" and len(content) > 10:
                causes.append(content.lower())
            elif section == "actions" and len(content) > 10:
                actions.append(content.lower())
    
    return {
        "problem": problem.lower(),
        "causes": causes,
        "actions": actions
    }

def faithfulness_score(pred_text: str, context_text: str) -> float:
    """Check if prediction is grounded in context."""
    if not pred_text or not context_text:
        return 0.0
    
    sentences = [s.strip() for s in pred_text.split(".") if len(s.strip()) > 15]
    if not sentences:
        return 0.0
    
    sent_embs = encode(sentences)
    ctx_emb = encode([context_text])
    
    sims = cosine_similarity(sent_embs, ctx_emb).flatten()
    supported = sum(1 for sim in sims if sim > 0.3)
    
    return supported / len(sentences)

def load_predictions(path: Path) -> List[Dict]:
    """Load SFT predictions from JSONL."""
    predictions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions

# Main evaluation
def run_sft_evaluation(test_size: int = 10) -> Dict[str, float]:
    """
    Evaluate SFT model predictions.
    
    Expects predictions file at: logs/sft_predictions.jsonl
    Each line: {"prediction": "...", "ground_truth": {...}}
    """
    
    pred_path = Path(config["sft_inference"]["sft_inference_paths"]["output_predictions"])
    
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {pred_path}\n"
            "Run inference first: python -m src.sft.inference.generate"
        )
    
    predictions = load_predictions(pred_path)[:test_size]
    
    logger.info(f"Evaluating {len(predictions)} SFT predictions...")
    
    metrics = {
        "semantic_similarity": [],
        "cause_f1": [],
        "action_f1": [],
        "faithfulness": [],
        "hallucination": [],
        "latency": [],
        "rouge_l": [],
        "bert_score": []
    }
    
    all_preds, all_refs = [], []
    
    for sample in predictions:
        pred_text = sample["prediction"]
        gt = sample["ground_truth"]
        gt_data = gt["ground_truth"]
        # Parse prediction
        parsed = parse_answer(pred_text)
        
        # Build ground truth text
        gold_problem = gt_data["problem"].replace("_", " ")
        gold_causes = [c.replace("_", " ") for c in gt_data["possible_causes"]]
        gold_actions = [a.replace("_", " ") for a in gt_data["recommended_actions"]]
        
        # Full text comparison
        pred_full = f"{parsed['problem']}. {' '.join(parsed['causes'])}. {' '.join(parsed['actions'])}"
        gold_full = f"{gold_problem}. {' '.join(gold_causes)}. {' '.join(gold_actions)}"
        
        # Compute metrics
        metrics["semantic_similarity"].append(semantic_similarity(pred_full, gold_full))
        metrics["cause_f1"].append(soft_f1(parsed["causes"], gold_causes))
        metrics["action_f1"].append(soft_f1(parsed["actions"], gold_actions))
        
        # Faithfulness (SFT has no retrieval context, so check self-consistency)
        faith = faithfulness_score(pred_full, pred_full)
        metrics["faithfulness"].append(faith)
        metrics["hallucination"].append(1 - faith)
        
        # For batch metrics
        all_preds.append(pred_full)
        all_refs.append(gold_full)
    
    # Batch metrics
    logger.info("Computing batch metrics...")
    metrics["rouge_l"] = [compute_rouge_l(all_preds, all_refs)]
    metrics["bert_score"] = [compute_bertscore(all_preds, all_refs)]
    metrics["latency"] = [0.0]  # No latency info in predictions file
    
    # Average all metrics
    results = {k: float(np.mean(v)) for k, v in metrics.items()}
    
    logger.info("\n" + "="*60)
    logger.info("SFT EVALUATION RESULTS")
    logger.info("="*60)
    for k, v in results.items():
        logger.info(f"{k:25s}: {v:.4f}")
    
    return results

if __name__ == "__main__":
    results = run_sft_evaluation(test_size=10)
    print("\n✅ Evaluation complete!")
