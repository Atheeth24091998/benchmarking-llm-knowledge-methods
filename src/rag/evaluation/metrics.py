# import json
# import time
# from pathlib import Path
# from typing import List, Dict, Set

# import numpy as np
# from bert_score import score as bert_score

# from src.rag.utils.logger import get_logger
# from src.rag.utils.config_loader import load_config

# from src.rag.vectorstore.faiss_store import FaissVectorStore
# from src.rag.generator.generate import generate_answer

# config = load_config()
# #print("CONFIG KEYS:", config.keys())

# test_path = Path(config["test_data_path"])
# #print(test_path)


# def f1_score(pred: Set[str], gold: Set[str]) -> float:
#     if not pred and not gold:
#         return 1.0
#     if not pred or not gold:
#         return 0.0

#     tp = len(pred & gold)
#     precision = tp / len(pred)
#     recall = tp / len(gold)

#     if precision + recall == 0:
#         return 0.0

#     return 2 * precision * recall / (precision + recall)

# def exact_match_problem(pred_problem: str, gold_problem: str) -> int:
#     print("exact_match_problem====================================================")
#     print("pred_problem",pred_problem)
#     print("gold_problem",gold_problem)
#     print("exact_match_problem====================================================")
#     return int(pred_problem.strip().lower() == gold_problem.strip().lower())

# def cause_f1(pred_causes: List[str], gold_causes: List[str]) -> float:
#     print("cause_f1====================================================")
#     print("pred_causes",pred_causes)
#     print("gold_causes",gold_causes)
#     print("cause_f1====================================================")
#     return f1_score(set(pred_causes), set(gold_causes))

# def action_f1(pred_actions: List[str], gold_actions: List[str]) -> float:
#     print("action_f1====================================================")
#     print("pred_actions",pred_actions)
#     print("gold_actions",gold_actions)
#     print("action_f1====================================================")
#     return f1_score(set(pred_actions), set(gold_actions))

# def compute_bertscore(predictions: List[str], references: List[str]) -> float:
#     print("compute_bertscore====================================================")
#     print("predictions",predictions)
#     print("references",references)
#     print("compute_bertscore====================================================")
#     P, R, F1 = bert_score(
#         predictions,
#         references,
#         lang="en",
#         rescale_with_baseline=True
#     )
#     return float(F1.mean())

# def faithfulness_score(pred_actions: List[str], retrieved_chunks: List[Dict]) -> float:
#     context = " ".join([c["text"].lower() for c in retrieved_chunks])

#     if not pred_actions:
#         return 1.0

#     supported = 0
#     for action in pred_actions:
#         if action.lower() in context:
#             supported += 1

#     return supported / len(pred_actions)

# def hallucination_rate(pred_actions: List[str], retrieved_chunks: List[Dict]) -> float:
#     context = " ".join([c["text"].lower() for c in retrieved_chunks])

#     if not pred_actions:
#         return 0.0

#     hallucinated = 0
#     for action in pred_actions:
#         if action.lower() not in context:
#             hallucinated += 1

#     return hallucinated / len(pred_actions)

# def measure_latency(fn, *args, **kwargs):
#     start = time.time()
#     result = fn(*args, **kwargs)
#     latency = time.time() - start
#     return result, latency

# def load_test_data(path: Path) -> List[Dict]:
#     data = []
#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             data.append(json.loads(line))
#     return data

# import re

# def parse_answer(answer: str) -> Dict:
#     """Parse structured answer with improved regex."""
    
#     # Initialize
#     problem = ""
#     causes = []
#     actions = []
    
#     # Split into sections
#     sections = answer.split("\n\n")
#     current_section = None
    
#     for line in answer.split("\n"):
#         line = line.strip()
#         if not line:
#             continue
        
#         # Detect section headers
#         if re.match(r"^problem:?$", line, re.IGNORECASE):
#             current_section = "problem"
#             continue
#         elif re.match(r"^possible causes:?$", line, re.IGNORECASE):
#             current_section = "causes"
#             continue
#         elif re.match(r"^recommended actions:?$", line, re.IGNORECASE):
#             current_section = "actions"
#             continue
        
#         # Extract content (lines starting with -)
#         if line.startswith("-"):
#             content = line[1:].strip()
            
#             # Skip placeholder text
#             if "[" in content and "]" in content:
#                 continue
#             if "cause 1" in content.lower() or "action 1" in content.lower():
#                 continue
            
#             # Assign to correct section
#             if current_section == "problem" and not problem:
#                 problem = content
#             elif current_section == "causes":
#                 causes.append(content)
#             elif current_section == "actions":
#                 actions.append(content)
    
#     # Fallback: extract from unstructured text
#     if not problem and not causes and not actions:
#         # Try to extract any useful info
#         for line in answer.split("\n"):
#             if line.strip() and not any(x in line.lower() for x in ["problem:", "causes:", "actions:"]):
#                 if not problem:
#                     problem = line.strip()
#                 elif len(causes) < 3:
#                     causes.append(line.strip())
#                 elif len(actions) < 3:
#                     actions.append(line.strip())
    
#     return {
#         "problem": problem.lower().strip(),
#         "causes": [c.lower().strip() for c in causes if c],
#         "actions": [a.lower().strip() for a in actions if a]
#     }



# def main():
#     test_data = load_test_data(test_path)
#     test_data = test_data[:10]
#     logger = get_logger(__name__)

#     store = FaissVectorStore()
#     store.load()

#     metrics = {
#         "exact_match": [],
#         "cause_f1": [],
#         "action_f1": [],
#         "faithfulness": [],
#         "hallucination": [],
#         "latency": []
#     }

#     for sample in test_data:
#         query = sample["question"]
#         gt = sample["ground_truth"]

#         retrieved_chunks = store.retrieve(query, top_k=3)

#         answer, latency = measure_latency(
#             generate_answer,
#             query,
#             retrieved_chunks
#         )
#         print(latency)

#         parsed = parse_answer(answer)

#         metrics["latency"].append(latency)

#         metrics["exact_match"].append(
#             exact_match_problem(parsed["problem"], gt["problem"])
#         )

#         metrics["cause_f1"].append(
#             cause_f1(parsed["causes"], gt["possible_causes"])
#         )

#         metrics["action_f1"].append(
#             action_f1(parsed["actions"], gt["recommended_actions"])
#         )

#         metrics["faithfulness"].append(
#             faithfulness_score(parsed["actions"], retrieved_chunks)
#         )

#         metrics["hallucination"].append(
#             hallucination_rate(parsed["actions"], retrieved_chunks)
#         )

#     # ---- Aggregate results ----
#     logger.info("===== FINAL RESULTS =====")
#     for k, v in metrics.items():
#         logger.info(f"{k}: {np.mean(v):.4f}")


# if __name__ == "__main__":
#     main()

# import json
# import time
# from pathlib import Path
# from typing import List, Dict, Set
# import re

# import numpy as np
# import torch
# from bert_score import score as bert_score
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from rouge_score import rouge_scorer

# from src.rag.utils.logger import get_logger
# from src.rag.utils.config_loader import load_config
# from src.rag.vectorstore.faiss_store import FaissVectorStore
# from src.rag.generator.generate import generate_answer

# config = load_config()
# test_path = Path(config["test_data_path"])

# # 🔥 GPU-accelerated embedding model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Metrics computation using device: {device}")
# similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


# def semantic_similarity(text1: str, text2: str) -> float:
#     """Compute cosine similarity between two texts (GPU-accelerated)."""
#     if not text1 or not text2:
#         return 0.0
    
#     # Encode on GPU, convert to numpy for sklearn
#     emb1 = similarity_model.encode([text1], normalize_embeddings=True, convert_to_numpy=True)
#     emb2 = similarity_model.encode([text2], normalize_embeddings=True, convert_to_numpy=True)
    
#     return float(cosine_similarity(emb1, emb2)[0][0])


# def soft_exact_match(pred_problem: str, gold_problem: str, threshold: float = 0.65) -> float:
#     """Soft matching using semantic similarity instead of exact match."""
#     if not pred_problem or not gold_problem:
#         return 0.0
    
#     # Convert underscore-separated to readable text
#     gold_readable = gold_problem.replace("_", " ")
    
#     similarity = semantic_similarity(pred_problem, gold_readable)
    
#     print(f"Soft Match: '{pred_problem[:60]}' <-> '{gold_readable}' = {similarity:.3f}")
    
#     # Return 1.0 if above threshold, otherwise the similarity score
#     return 1.0 if similarity >= threshold else similarity


# def soft_f1_score(pred_list: List[str], gold_list: List[str], threshold: float = 0.55) -> float:
#     """F1 with semantic matching instead of exact string matching (GPU-accelerated)."""
#     if not pred_list or not gold_list:
#         return 0.0
    
#     # Convert gold items to readable format
#     gold_readable = [item.replace("_", " ") for item in gold_list]
    
#     matched_pred = set()
#     matched_gold = set()
    
#     # 🔥 Batch encode predictions and gold on GPU for speed
#     if len(pred_list) > 1 and len(gold_readable) > 1:
#         pred_embs = similarity_model.encode(pred_list, normalize_embeddings=True, convert_to_numpy=True)
#         gold_embs = similarity_model.encode(gold_readable, normalize_embeddings=True, convert_to_numpy=True)
        
#         # Compute similarity matrix
#         sim_matrix = cosine_similarity(pred_embs, gold_embs)
        
#         # Greedy matching
#         for i in range(len(pred_list)):
#             best_j = -1
#             best_sim = 0.0
            
#             for j in range(len(gold_readable)):
#                 if j not in matched_gold and sim_matrix[i, j] > best_sim:
#                     best_sim = sim_matrix[i, j]
#                     best_j = j
            
#             if best_sim >= threshold and best_j != -1:
#                 matched_pred.add(i)
#                 matched_gold.add(best_j)
#                 print(f"  ✓ Matched: '{pred_list[i][:40]}' <-> '{gold_readable[best_j][:40]}' ({best_sim:.3f})")
#     else:
#         # Fallback for small lists
#         for i, pred in enumerate(pred_list):
#             best_sim = 0.0
#             best_j = -1
            
#             for j, gold in enumerate(gold_readable):
#                 if j in matched_gold:
#                     continue
                
#                 sim = semantic_similarity(pred, gold)
#                 if sim > best_sim:
#                     best_sim = sim
#                     best_j = j
            
#             if best_sim >= threshold and best_j != -1:
#                 matched_pred.add(i)
#                 matched_gold.add(best_j)
#                 print(f"  ✓ Matched: '{pred[:40]}' <-> '{gold_readable[best_j][:40]}' ({best_sim:.3f})")
    
#     # Calculate precision and recall
#     precision = len(matched_pred) / len(pred_list) if pred_list else 0.0
#     recall = len(matched_gold) / len(gold_list) if gold_list else 0.0
    
#     # F1 score
#     if precision + recall == 0:
#         return 0.0
    
#     f1 = 2 * precision * recall / (precision + recall)
#     print(f"Soft F1: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
#     return f1


# def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
#     """Compute ROUGE-L score."""
#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     scores = []
    
#     for pred, ref in zip(predictions, references):
#         score = scorer.score(ref, pred)
#         scores.append(score['rougeL'].fmeasure)
    
#     return float(np.mean(scores))


# # At the top, update BERTScore function:
# def compute_bertscore(predictions: List[str], references: List[str]) -> float:
#     """Compute BERTScore F1 (GPU-accelerated)."""
#     if not predictions or not references:
#         return 0.0
    
#     valid_pairs = [(p, r) for p, r in zip(predictions, references) 
#                    if p.strip() and r.strip() and len(p.strip()) > 10]
    
#     if not valid_pairs:
#         print("WARNING: No valid prediction-reference pairs for BERTScore")
#         return 0.0
    
#     preds, refs = zip(*valid_pairs)
    
#     try:
#         P, R, F1 = bert_score(
#             list(preds),
#             list(refs),
#             lang="en",
#             rescale_with_baseline=True,
#             verbose=False,
#             device=device,
#             batch_size=4
#         )
#         score = float(F1.mean())
#         print(f"BERTScore computed: {score:.4f} (from {len(preds)} pairs)")
#         return score
#     except Exception as e:
#         print(f"BERTScore error: {e}")
#         return 0.0

# # Update faithfulness threshold:
# def faithfulness_score(pred_text: str, retrieved_chunks: List[Dict]) -> float:
#     if not pred_text:
#         return 0.0
    
#     context = " ".join([c["text"] for c in retrieved_chunks])
#     pred_sentences = [s.strip() for s in pred_text.split(".") 
#                       if s.strip() and len(s.strip()) > 10]
    
#     if not pred_sentences:
#         return 0.0
    
#     if len(pred_sentences) > 1:
#         sent_embs = similarity_model.encode(pred_sentences, normalize_embeddings=True, convert_to_numpy=True)
#         context_emb = similarity_model.encode([context], normalize_embeddings=True, convert_to_numpy=True)
        
#         similarities = cosine_similarity(sent_embs, context_emb).flatten()
#         supported = sum(1 for sim in similarities if sim > 0.35)  # Lowered threshold
#     else:
#         supported = 1 if semantic_similarity(pred_sentences[0], context) > 0.35 else 0
    
#     return supported / len(pred_sentences)



# def faithfulness_score(pred_text: str, retrieved_chunks: List[Dict]) -> float:
#     """Check if predicted text is semantically grounded in retrieved context (GPU-accelerated)."""
#     if not pred_text:
#         return 0.0
    
#     context = " ".join([c["text"] for c in retrieved_chunks])
#     pred_sentences = [s.strip() for s in pred_text.split(".") if s.strip()]
    
#     if not pred_sentences:
#         return 0.0
    
#     # 🔥 Batch encode sentences on GPU
#     if len(pred_sentences) > 1:
#         sent_embs = similarity_model.encode(pred_sentences, normalize_embeddings=True, convert_to_numpy=True)
#         context_emb = similarity_model.encode([context], normalize_embeddings=True, convert_to_numpy=True)
        
#         # Compute similarities
#         similarities = cosine_similarity(sent_embs, context_emb).flatten()
#         supported = sum(1 for sim in similarities if sim > 0.5)
#     else:
#         # Single sentence
#         supported = 1 if semantic_similarity(pred_sentences[0], context) > 0.5 else 0
    
#     return supported / len(pred_sentences)


# def hallucination_rate(pred_text: str, retrieved_chunks: List[Dict]) -> float:
#     """Inverse of faithfulness - how much is NOT grounded."""
#     return 1.0 - faithfulness_score(pred_text, retrieved_chunks)


# def measure_latency(fn, *args, **kwargs):
#     start = time.time()
#     result = fn(*args, **kwargs)
#     latency = time.time() - start
#     return result, latency


# def load_test_data(path: Path) -> List[Dict]:
#     data = []
#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             data.append(json.loads(line))
#     return data


# def parse_answer(answer: str) -> Dict:
#     """Parse structured answer - handle markdown bold."""
#     problem = ""
#     causes = []
#     actions = []
    
#     current_section = None
    
#     for line in answer.split("\n"):
#         line = line.strip()
#         if not line:
#             continue
        
#         # Remove markdown bold
#         line = line.replace("**", "")
        
#         # Detect section headers (case-insensitive)
#         if re.match(r"^problem:?\s*$", line, re.IGNORECASE):
#             current_section = "problem"
#             continue
#         elif re.match(r"^possible causes:?\s*$", line, re.IGNORECASE):
#             current_section = "causes"
#             continue
#         elif re.match(r"^recommended actions:?\s*$", line, re.IGNORECASE):
#             current_section = "actions"
#             continue
        
#         # Extract content (lines starting with -)
#         if line.startswith("-"):
#             content = line[1:].strip()
            
#             # Skip placeholder text or meta-text
#             if any(skip in content.lower() for skip in [
#                 "[", "]", "cause 1", "cause 2", "action 1", "action 2",
#                 "not specified in manual", "as mentioned in", "section "
#             ]):
#                 continue
            
#             # Assign to correct section
#             if current_section == "problem" and not problem:
#                 problem = content
#             elif current_section == "causes" and content:
#                 causes.append(content)
#             elif current_section == "actions" and content:
#                 actions.append(content)
    
#     return {
#         "problem": problem.lower().strip(),
#         "causes": [c.lower().strip() for c in causes if c],
#         "actions": [a.lower().strip() for a in actions if a]
#     }



# def main():
#     test_data = load_test_data(test_path)
#     test_data = test_data[:10]  # Test on first 10
#     logger = get_logger(__name__)

#     store = FaissVectorStore()
#     store.load()

#     metrics = {
#         "soft_exact_match": [],
#         "soft_cause_f1": [],
#         "soft_action_f1": [],
#         "rouge_l": [],
#         "bert_score": [],
#         "semantic_similarity": [],
#         "faithfulness": [],
#         "hallucination": [],
#         "latency": []
#     }
    
#     all_predictions = []
#     all_references = []

#     for idx, sample in enumerate(test_data):
#         logger.info(f"\n{'='*80}\nSample {idx+1}/{len(test_data)}\n{'='*80}")
        
#         query = sample["question"]
#         gt = sample["ground_truth"]

#         retrieved_chunks = store.retrieve(query, top_k=5)

#         answer, latency = measure_latency(generate_answer, query, retrieved_chunks)
        
#         logger.info(f"Query: {query[:80]}...")
#         logger.info(f"Answer: {answer[:150]}...")
        
#         parsed = parse_answer(answer)
        
#         # Soft semantic matching
#         print(f"\n--- Problem Matching ---")
#         metrics["soft_exact_match"].append(
#             soft_exact_match(parsed["problem"], gt["problem"])
#         )
        
#         print(f"\n--- Cause F1 ---")
#         metrics["soft_cause_f1"].append(
#             soft_f1_score(parsed["causes"], gt["possible_causes"])
#         )
        
#         print(f"\n--- Action F1 ---")
#         metrics["soft_action_f1"].append(
#             soft_f1_score(parsed["actions"], gt["recommended_actions"])
#         )
        
#         # Semantic similarity between full answers
#         pred_full = f"{parsed['problem']}. {' '.join(parsed['causes'])}. {' '.join(parsed['actions'])}"
#         gold_full = f"{gt['problem'].replace('_', ' ')}. {' '.join([c.replace('_', ' ') for c in gt['possible_causes']])}. {' '.join([a.replace('_', ' ') for a in gt['recommended_actions']])}"
        
#         metrics["semantic_similarity"].append(
#             semantic_similarity(pred_full, gold_full)
#         )
        
#         # Faithfulness
#         metrics["faithfulness"].append(
#             faithfulness_score(pred_full, retrieved_chunks)
#         )
#         metrics["hallucination"].append(
#             hallucination_rate(pred_full, retrieved_chunks)
#         )
        
#         metrics["latency"].append(latency)
        
#         # Collect for ROUGE and BERTScore
#         all_predictions.append(pred_full)
#         all_references.append(gold_full)

#     # Compute batch metrics
#     logger.info(f"\n{'='*80}\nComputing ROUGE-L and BERTScore...\n{'='*80}")
#     metrics["rouge_l"] = [compute_rouge_l(all_predictions, all_references)]
#     metrics["bert_score"] = [compute_bertscore(all_predictions, all_references)]

#     # Final results
#     logger.info(f"\n{'='*80}")
#     logger.info("FINAL RESULTS (Semantic Evaluation)")
#     logger.info(f"{'='*80}")
    
#     for k, v in metrics.items():
#         logger.info(f"{k:25s}: {np.mean(v):.4f}")


# if __name__ == "__main__":
#     main()































# #WORKING

# import json
# import time
# from pathlib import Path
# from typing import List, Dict
# import re

# import numpy as np
# import torch
# from bert_score import score as bert_score
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from rouge_score import rouge_scorer

# from src.rag.utils.logger import get_logger
# from src.rag.utils.config_loader import load_config
# from src.rag.vectorstore.faiss_store import FaissVectorStore
# from src.rag.generator.generate import generate_answer

# config = load_config()
# test_path = Path(config["test_data_path"])

# # GPU-accelerated embedding model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Metrics computation using device: {device}")
# similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


# def semantic_similarity(text1: str, text2: str) -> float:
#     """Compute cosine similarity between two texts."""
#     if not text1 or not text2:
#         return 0.0
    
#     emb1 = similarity_model.encode([text1], normalize_embeddings=True, convert_to_numpy=True)
#     emb2 = similarity_model.encode([text2], normalize_embeddings=True, convert_to_numpy=True)
    
#     return float(cosine_similarity(emb1, emb2)[0][0])


# def soft_exact_match(pred_problem: str, gold_problem: str, threshold: float = 0.6) -> float:
#     """Soft matching using semantic similarity."""
#     if not pred_problem or not gold_problem:
#         return 0.0
    
#     gold_readable = gold_problem.replace("_", " ")
#     similarity = semantic_similarity(pred_problem, gold_readable)
    
#     print(f"Problem Match: {similarity:.3f} - '{pred_problem[:50]}' vs '{gold_readable}'")
    
#     return 1.0 if similarity >= threshold else similarity


# def soft_f1_score(pred_list: List[str], gold_list: List[str], threshold: float = 0.5) -> float:
#     """F1 with semantic matching."""
#     if not pred_list or not gold_list:
#         print(f"  Empty lists: pred={len(pred_list)}, gold={len(gold_list)}")
#         return 0.0
    
#     gold_readable = [item.replace("_", " ") for item in gold_list]
    
#     matched_pred = set()
#     matched_gold = set()
    
#     # Batch encode
#     if len(pred_list) > 0 and len(gold_readable) > 0:
#         pred_embs = similarity_model.encode(pred_list, normalize_embeddings=True, convert_to_numpy=True)
#         gold_embs = similarity_model.encode(gold_readable, normalize_embeddings=True, convert_to_numpy=True)
        
#         sim_matrix = cosine_similarity(pred_embs, gold_embs)
        
#         for i in range(len(pred_list)):
#             best_j = -1
#             best_sim = 0.0
            
#             for j in range(len(gold_readable)):
#                 if j not in matched_gold and sim_matrix[i, j] > best_sim:
#                     best_sim = sim_matrix[i, j]
#                     best_j = j
            
#             if best_sim >= threshold and best_j != -1:
#                 matched_pred.add(i)
#                 matched_gold.add(best_j)
#                 print(f"  ✓ {best_sim:.3f}: '{pred_list[i][:35]}' <-> '{gold_readable[best_j][:35]}'")
    
#     precision = len(matched_pred) / len(pred_list) if pred_list else 0.0
#     recall = len(matched_gold) / len(gold_list) if gold_list else 0.0
    
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
#     print(f"  F1={f1:.3f} (P={precision:.3f}, R={recall:.3f})")
    
#     return f1


# def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
#     """Compute ROUGE-L score."""
#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     scores = []
    
#     for pred, ref in zip(predictions, references):
#         if pred.strip() and ref.strip():
#             score = scorer.score(ref, pred)
#             scores.append(score['rougeL'].fmeasure)
    
#     return float(np.mean(scores)) if scores else 0.0


# def compute_bertscore(predictions: List[str], references: List[str]) -> float:
#     """Compute BERTScore F1."""
#     if not predictions or not references:
#         return 0.0
    
#     # Filter valid pairs (minimum 20 characters)
#     valid_pairs = [(p, r) for p, r in zip(predictions, references) 
#                    if p.strip() and r.strip() and len(p.strip()) > 20 and len(r.strip()) > 20]
    
#     if not valid_pairs:
#         print("WARNING: No valid pairs for BERTScore (texts too short)")
#         return 0.0
    
#     preds, refs = zip(*valid_pairs)
    
#     try:
#         # Disable rescaling if causing issues
#         P, R, F1 = bert_score(
#             list(preds),
#             list(refs),
#             lang="en",
#             rescale_with_baseline=False,  # Changed to False
#             verbose=False,
#             device=device,
#             batch_size=4
#         )
#         score = float(F1.mean())
#         print(f"BERTScore: {score:.4f} (from {len(preds)} pairs)")
#         return score
#     except Exception as e:
#         print(f"BERTScore error: {e}")
#         return 0.0


# def faithfulness_score(pred_text: str, retrieved_chunks: List[Dict]) -> float:
#     """Check semantic grounding."""
#     if not pred_text or not retrieved_chunks:
#         return 0.0
    
#     context = " ".join([c["text"] for c in retrieved_chunks])
#     pred_sentences = [s.strip() for s in pred_text.split(".") 
#                       if s.strip() and len(s.strip()) > 15]
    
#     if not pred_sentences:
#         return 0.0
    
#     # Batch encode
#     sent_embs = similarity_model.encode(pred_sentences, normalize_embeddings=True, convert_to_numpy=True)
#     context_emb = similarity_model.encode([context], normalize_embeddings=True, convert_to_numpy=True)
    
#     similarities = cosine_similarity(sent_embs, context_emb).flatten()
#     supported = sum(1 for sim in similarities if sim > 0.3)  # Relaxed threshold
    
#     return supported / len(pred_sentences)


# def hallucination_rate(pred_text: str, retrieved_chunks: List[Dict]) -> float:
#     """Inverse of faithfulness."""
#     return 1.0 - faithfulness_score(pred_text, retrieved_chunks)


# def measure_latency(fn, *args, **kwargs):
#     start = time.time()
#     result = fn(*args, **kwargs)
#     latency = time.time() - start
#     return result, latency


# def load_test_data(path: Path) -> List[Dict]:
#     data = []
#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             data.append(json.loads(line))
#     return data


# def parse_answer(answer: str) -> Dict:
#     """Parse structured answer - LESS STRICT filtering."""
#     problem = ""
#     causes = []
#     actions = []
    
#     current_section = None
    
#     # Remove markdown
#     answer = answer.replace("**", "")
    
#     for line in answer.split("\n"):
#         line = line.strip()
#         if not line:
#             continue
        
#         # Detect headers
#         if re.match(r"^problem:?\s*$", line, re.IGNORECASE):
#             current_section = "problem"
#             continue
#         elif re.match(r"^possible causes:?\s*$", line, re.IGNORECASE):
#             current_section = "causes"
#             continue
#         elif re.match(r"^recommended actions:?\s*$", line, re.IGNORECASE):
#             current_section = "actions"
#             continue
        
#         # Extract content
#         if line.startswith("-"):
#             content = line[1:].strip()
            
#             # Only skip obvious placeholders
#             if ("[" in content and "]" in content) or \
#                content.lower() in ["cause 1", "cause 2", "cause 3", "action 1", "action 2", "action 3"]:
#                 continue
            
#             # Less strict - keep content with section references
#             if current_section == "problem" and not problem:
#                 problem = content
#             elif current_section == "causes" and len(content) > 10:
#                 causes.append(content)
#             elif current_section == "actions" and len(content) > 10:
#                 actions.append(content)
    
#     print(f"  Parsed: problem={bool(problem)}, causes={len(causes)}, actions={len(actions)}")
    
#     return {
#         "problem": problem.lower().strip(),
#         "causes": [c.lower().strip() for c in causes],
#         "actions": [a.lower().strip() for a in actions]
#     }


# def main():
#     test_data = load_test_data(test_path)
#     test_data = test_data[:10]
#     logger = get_logger(__name__)

#     store = FaissVectorStore()
#     store.load()

#     metrics = {
#         "soft_exact_match": [],
#         "soft_cause_f1": [],
#         "soft_action_f1": [],
#         "rouge_l": [],
#         "bert_score": [],
#         "semantic_similarity": [],
#         "faithfulness": [],
#         "hallucination": [],
#         "latency": []
#     }
    
#     all_predictions = []
#     all_references = []

#     for idx, sample in enumerate(test_data):
#         logger.info(f"\n{'='*60}\nSample {idx+1}/{len(test_data)}: {sample['question'][:60]}...\n{'='*60}")
        
#         query = sample["question"]
#         gt = sample["ground_truth"]

#         retrieved_chunks = store.retrieve(query, top_k=5)
#         answer, latency = measure_latency(generate_answer, query, retrieved_chunks)
        
#         parsed = parse_answer(answer)
        
#         # Problem matching
#         print(f"\n--- Problem ---")
#         metrics["soft_exact_match"].append(
#             soft_exact_match(parsed["problem"], gt["problem"])
#         )
        
#         # Cause F1
#         print(f"\n--- Causes ---")
#         metrics["soft_cause_f1"].append(
#             soft_f1_score(parsed["causes"], gt["possible_causes"])
#         )
        
#         # Action F1
#         print(f"\n--- Actions ---")
#         metrics["soft_action_f1"].append(
#             soft_f1_score(parsed["actions"], gt["recommended_actions"])
#         )
        
#         # Full answer comparison
#         pred_full = f"{parsed['problem']}. {' '.join(parsed['causes'])}. {' '.join(parsed['actions'])}"
#         gold_full = f"{gt['problem'].replace('_', ' ')}. {' '.join([c.replace('_', ' ') for c in gt['possible_causes']])}. {' '.join([a.replace('_', ' ') for a in gt['recommended_actions']])}"
        
#         metrics["semantic_similarity"].append(semantic_similarity(pred_full, gold_full))
#         metrics["faithfulness"].append(faithfulness_score(pred_full, retrieved_chunks))
#         metrics["hallucination"].append(hallucination_rate(pred_full, retrieved_chunks))
#         metrics["latency"].append(latency)
        
#         all_predictions.append(pred_full)
#         all_references.append(gold_full)

#     # Batch metrics
#     logger.info(f"\n{'='*60}\nComputing batch metrics...\n{'='*60}")
#     metrics["rouge_l"] = [compute_rouge_l(all_predictions, all_references)]
#     metrics["bert_score"] = [compute_bertscore(all_predictions, all_references)]

#     # Final results
#     logger.info(f"\n{'='*60}")
#     logger.info("FINAL RESULTS")
#     logger.info(f"{'='*60}")
    
#     for k, v in metrics.items():
#         logger.info(f"{k:25s}: {np.mean(v):.4f}")


# if __name__ == "__main__":
#     main()


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
        "latency": []
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
