# Comparative Study of LLM Adaptation Techniques for Technical Manuals
**Master's Thesis**
**Author:** Atheeth Naik
**University:** FAU Erlangen, Germany
**Thesis Mentor** Ludwig Streloke ( FAPS FAU Erlangen, Germany)

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
​- [X] **RAG Eveluation metrics** (Partially done)
    - Evaluation metrics such as semantic_similarity, bert_score, hallucination .etc has been implemnted and tested.
    - The results are displayed in Stremlit application.
    - I see some issues in the metrics being calculated, need to debug and test again.
- [X] **Knowledge Graph Construction** (Partially done)
    - Implemntation of the Grapgh Rag has been done.
    - Visualisation is done in Neo4j
    - Evaluation metrics such as semantic_similarity, bert_score, hallucination .etc has been implemnted and tested.
    - The results are displayed in Stremlit application.
    - I see some issues in the metrics being calculated, need to debug and test again.
- [X] **Fine-Tuning (SFT) & Evaluation** (Done)
    - Implemented and tested the SFT usinbg LORA 
    - Evaluation metrics for the SFT is also integrated and tested 
    - Also results are displyed in the Streamlit application
- [ ] **RLKGF Implementation and Evaluation** (Partially done)
    - Implementation of RLKGF approachh initialy framework is created. However the training seems to give accuracy as zero always. 
    - Debug and fixing the training issue
    - Integrate the logging and test related code 
    - integarte the results into the Streamlit application

- [ ] **RAG Documentation** (Planned)
- [ ] **GraphRAG Documentation** (Planned)
- [ ] **SFT Documentation** (Planned)
- [ ] **RLKGF Documentation** (Planned)