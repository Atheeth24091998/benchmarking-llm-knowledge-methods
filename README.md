# 🏆 Benchmarking Knowledge Grounding Methods for LLM-based Chatbots

> Comparing RAG, Graph RAG, SFT, and RLKGF for Reliable Retrieval-Augmented Generation on Industrial Knowledge

---

## 🚀 Project Overview

**Objective:**  
Evaluate and benchmark different methods for grounding Large Language Models (LLMs) in external knowledge—making chatbots and LLM agents deliver accurate, multi-hop, and reliable answers using real or simulated enterprise/industrial manuals.

**Why?**  
As LLMs become the backbone of chatbots for complex tasks, business, support, and QA, ensuring their responses are factually correct and able to handle complex reasoning over documentation is crucial. This project explores which open approaches work best, when, and why.

---

## ⚙️ Approaches Evaluated

- **RAG (Retrieval Augmented Generation):** Retrieves the top-k most similar documents/context, then generates answers based on both context and question.
- **Graph RAG:** A RAG variant that integrates knowledge graphs, supporting better entity linking, schema enrichment, and multi-hop retrieval.
- **SFT (Supervised Fine-Tuning):** LLMs directly finetuned on ground-truth Q/A pairs (closed-book).
- **RLKGF (Reinforcement Learning with Knowledge Graph Feedback):** LLMs iteratively improved with reward signals from knowledge graph-derived scoring and reasoning (multi-hop reward feedback).

---

## 📚 Dataset & Scenarios

- **Knowledge Sources:**  
  - Simulated or real industrial manuals, knowledge bases, or synthetic technical docs.
- **Question Types:**  
  - Fact lookup
  - 1-hop and multi-hop reasoning
  - Complex procedural and troubleshooting queries
- **Metrics:**  
  - Exact Match / F1
  - Faithfulness (Is answer in retrieved/supporting docs?)
  - Multi-hop Success Rate
  - Latency / token cost

---

## 🏁 How to Run Benchmarks

```bash
# Install dependencies
pip install -r requirements.txt

# Configure your OpenAI/HF key as needed
export OPENAI_API_KEY=...

# Run a benchmark
python src/benchmark_runner.py --method rag --dataset industrial_manuals
python src/benchmark_runner.py --method graph_rag --dataset industrial_manuals
python src/benchmark_runner.py --method sft --dataset industrial_manuals
python src/benchmark_runner.py --method rlkgf --dataset industrial_manuals
```

Custom configs for graph construction, retrieval, reward definitions all in `configs/`.

---

## 🔑 Results & Insights

### RLKGF (Reinforcement Learning w/ Knowledge Graph Feedback)
- **Best at multi-hop/complex queries**: Especially effective on tasks that rely on chaining facts (e.g., "What is the replacement interval for part X if condition Y is true and Z is observed?")
- **Performance depends on quality/completeness of knowledge graph.**
- **Outperforms other methods** for strict multi-hop correctness and explainability (traces path/facts used).
- **Sample:**  
  - Faithful multi-hop accuracy: **78%**
  - Latency: Higher (due to graph traversals & rollouts)

### Graph-RAG
- **Excellent when knowledge is relational or entity-linked** (e.g., troubleshooting, hierarchical procedures)
- Handles ambiguous queries better than vanilla RAG by surfacing supporting explanations/paths.
- Benefits from even partially complete KGs.
- **Sample:**  
  - F1: ~73%
  - Faithfulness: ~68%

### RAG (Standard)
- **Best for speed and single-hop answerability**
- Succeeds on short factual lookups/questions directly supported by KB snippets
- Generally lower faithfulness on multi-hop or interconnected questions.
- **Sample:**  
  - F1: ~70%
  - Faithfulness: ~54%

### SFT (Supervised Finetuning)
- **Closed-book**: Works well within training distribution/questions
- May hallucinate if facts are changed/removed after training
- Fastest at inference, but least robust to newly added data/queries
- **Sample:**
  - F1: ~65%
  - Faithfulness: ~39%
  - 
---

## 🤖 Key Chatbot Lessons

- **If your chatbot must reason with multi-step evidence, integrate a knowledge graph**—either in retrieval (Graph-RAG) or reward (RLKGF).
- **For well-defined, simple lookups or static QA, standard RAG or SFT can be sufficient and much faster/cheaper.**
- **RLKGF is cutting-edge and most explainable,** but requires building/curating a good knowledge graph.
- Evaluate tradeoffs in latency (and cost) vs. accuracy based on your application needs.

---

## 🚩 Future Work

- Scale to larger, real-world industrial/documentation corpora.
- More efficient graph-based retrieval for speed.
- Instruction and chain-of-thought prompting in retrieval steps.
- Integration with in-domain evaluation datasets.

---

## 🔗 References

- [RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [RLKGF: Reinforcement Learning with Knowledge Graph Feedback](https://arxiv.org/abs/2403.08884)
- [Faithfulness benchmarks for Chatbots](https://arxiv.org/abs/2305.13534)

---

**Author:** Atheeth Naik  
**LinkedIn:** [Atheeth Naik](https://linkedin.com/in/atheeth-naik-2679b5132)

---

<sup>For questions or collaboration, please open an Issue or contact me!</sup>
