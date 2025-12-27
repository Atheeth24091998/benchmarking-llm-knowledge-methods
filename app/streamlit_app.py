import streamlit as st

from src.rag.vectorstore.faiss_store import FaissVectorStore
from src.rag.generator.generate import generate_answer
from src.rag.evaluation.metrics import run_evaluation
from src.graph_rag.linking.entity_linker import EntityLinker
from src.graph_rag.retrieval.subgraph_retriever import SubgraphRetriever
from src.graph_rag.generator.generate import GraphAnswerGenerator

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Industrial Manual RAG",
    layout="wide"
)

st.title("📘 Industrial Manual Assistant")
st.caption("Ask questions or evaluate the RAG system")

# -------------------------
# Mode selector
# -------------------------
mode = st.radio(
    "Choose mode",
    [
        "🔍 Vector RAG – Ask",
        "🧠 Graph RAG – Ask",
        "🧪 Vector RAG – Evaluate",
        "🧪 Graph RAG – Evaluate"
    ],
    horizontal=True
)

# -------------------------
# Load vector store (cached)
# -------------------------
@st.cache_resource
def load_store():
    store = FaissVectorStore()
    store.load()
    return store

store = load_store()

# =========================================================
# 🔍 MODE 1: Ask a single question
# =========================================================
if mode == "🔍 Ask a Question":

    query = st.text_input(
        "Enter your question",
        placeholder="e.g. What is the maximum wire length?"
    )

    top_k = st.slider("Number of retrieved chunks", 1, 10, 3)

    if st.button("🔍 Ask") and query.strip():

        with st.spinner("Retrieving relevant sections..."):
            retrieved_chunks = store.retrieve(query, top_k=top_k)

        with st.spinner("Generating answer..."):
            answer = generate_answer(query, retrieved_chunks)

        st.subheader("✅ Answer")
        st.write(answer)

        with st.expander("📚 Retrieved context"):
            for i, chunk in enumerate(retrieved_chunks, 1):
                st.markdown(f"**Chunk {i} — {chunk['metadata']['section_full']}**")
                st.write(chunk["text"])
                st.markdown("---")

elif mode == "🧠 Graph RAG – Ask":



    query = st.text_input("Enter your question")

    if st.button("🧠 Ask with Graph RAG"):

        linker = EntityLinker()
        retriever = SubgraphRetriever()
        generator = GraphAnswerGenerator()

        problem, score = linker.link(query)
        subgraph = retriever.retrieve(problem)
        answer = generator.generate(query, subgraph)

        st.subheader("✅ Graph RAG Answer")
        st.write(answer)

        with st.expander("🧠 Graph Facts Used"):
            st.json(subgraph)

elif mode == "🧪 Graph RAG – Evaluate":

    from src.graph_rag.evaluation.metrics import run_graph_evaluation

    test_size = st.slider("Test set size", 1, 100, 10)

    if st.button("▶ Run Graph RAG Evaluation"):

        with st.spinner("Evaluating Graph RAG..."):
            results = run_graph_evaluation(test_size)

        st.success("Graph RAG Evaluation Complete ✅")

        col1, col2, col3 = st.columns(3)

        col1.metric("Semantic Similarity", f"{results['semantic_similarity']:.3f}")
        col1.metric("Faithfulness", f"{results['faithfulness']:.3f}")
        col1.metric("Hallucination", f"{results['hallucination']:.3f}")

        col2.metric("Cause F1", f"{results['cause_f1']:.3f}")
        col2.metric("Action F1", f"{results['action_f1']:.3f}")

        col3.metric("Latency (s)", f"{results['latency']:.2f}")
        col3.metric("rouge",f"{results['rouge']:.3f}")
        col3.metric("bert",f"{results['bert']:.3f}")


# =========================================================
# 🧪 MODE 2: Run evaluation
# =========================================================
else:
    st.subheader("🧪 RAG Evaluation")

    test_size = st.slider(
        "Test set size",
        min_value=1,
        max_value=100,
        value=10,
        step=1
    )

    if st.button("▶ Run Evaluation"):

        with st.spinner("Running evaluation on test set..."):
            results = run_evaluation(test_size)

        st.success("Evaluation complete ✅")

        col1, col2, col3 = st.columns(3)

        col1.metric("Semantic Similarity", f"{results['semantic_similarity']:.3f}")
        col1.metric("Faithfulness", f"{results['faithfulness']:.3f}")
        col1.metric("Hallucination", f"{results['hallucination']:.3f}")

        col2.metric("Cause F1", f"{results['cause_f1']:.3f}")
        col2.metric("Action F1", f"{results['action_f1']:.3f}")

        col3.metric("Latency (s)", f"{results['latency']:.2f}")
        col3.metric("rouge_l",f"{results['rouge_l']:.3f}")
        col3.metric("bert_score",f"{results['bert_score']:.3f}")