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
4.  **RLKGF (Reinforcement Learning from Knowledge Grapgh Feedback):** Alignment optimization (future scope).

## 🚀 Current Status
- [x] **Phase 1: Data Extraction** (Implemented)
    - Context-aware parsing using `pymupdf4llm`.
    - Preservation of tables, headers, and hierarchical structure.
- [x] **Phase 2: Data Chunking** (Implemented)
    - Structure-aware chunking for text vs tabular sections
    - Context preservation via overlap and section metadata
    - End-to-end JSONL processing pipeline for embeddings"
- [X] **Phase 3: Vector Embedding & RAG** (Implemented)
    - Embedding Generation: Utilizes sentence-transformers/all-MiniLM-L6-v2 for encoding text chunks into 384-dimensional dense vectors
    - Vector Storage: FAISS IndexFlatIP implementation for efficient cosine similarity search via inner product
    - Semantic Retrieval: Top-k retrieval with configurable batch processing and similarity scoring
    - LLM Integration: Llama-2-7b for context-aware answer generation from retrieved chunks
​
End-to-end Pipeline: Automated workflow from query encoding → retrieval → context injection → generation
- [ ] **RAG Eveluation metrics** (Planned)
- [ ] **RAG Documentation** (Planned)
- [ ] **Knowledge Graph Construction** (Planned)
- [ ] **Fine-Tuning (SFT) & Evaluation** (Planned)
- [ ] **RLKGF Implementation and Evaluation** (Planned)
