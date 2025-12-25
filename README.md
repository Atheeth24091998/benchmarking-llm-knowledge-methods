# Comparative Study of LLM Adaptation Techniques for Technical Manuals
**Master's Thesis**
**Author:** Atheeth Naik
**University:** FAU Erlangen

---

## 📖 Abstract
This research aims to benchmark and evaluate different Large Language Model (LLM) optimization techniques when applied to complex, unstructured industrial documentation. Using the **Omega 740/750 Machine Manual**  as a primary dataset, this study compares the accuracy, hallucination rate, and context retention of the following methodologies:

1.  **RAG (Retrieval-Augmented Generation):** Baseline retrieval using vector similarity.
2.  **GraphRAG:** Knowledge-graph enhanced retrieval to preserve structural relationships between components.
3.  **SFT (Supervised Fine-Tuning):** Domain-specific model tuning on extracted manual sections.
4.  **RLHF (Reinforcement Learning from Human Feedback):** Alignment optimization (future scope).

## 🚀 Current Status
- [x] **Phase 1: Data Extraction** (Implemented)
    - Context-aware parsing using `pymupdf4llm`.
    - Preservation of tables, headers, and hierarchical structure.
- [x] **Phase 2: Data Chunking** (Implemented)
    - Structure-aware chunking for text vs tabular sections
    - Context preservation via overlap and section metadata
    - End-to-end JSONL processing pipeline for embeddings"
- [ ] **Phase 3: Vector Embedding & RAG** (Planned)
- [ ] **Phase 4: Knowledge Graph Construction** (Planned)
- [ ] **Phase 5: Fine-Tuning (SFT) & Evaluation** (Planned)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)[your-username]/thesis-llm-optimization-benchmark.git
   cd thesis-llm-optimization-benchmark