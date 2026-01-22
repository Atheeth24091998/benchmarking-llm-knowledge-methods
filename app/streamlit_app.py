import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.rag.vectorstore.faiss_store import FaissVectorStore
from src.rag.generator.generate import generate_answer
from src.rag.evaluation.metrics import run_evaluation
from src.graph_rag.linking.entity_linker import EntityLinker
from src.graph_rag.retrieval.subgraph_retriever import SubgraphRetriever
from src.graph_rag.generator.generate import GraphAnswerGenerator
from src.sft.inference.infer_sft import sft_infer
from src.sft.evaluation.metrics import run_sft_evaluation
from pathlib import Path

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
# Simple History Function
# -------------------------
def save_to_history(method, test_samples, results):
    """Save evaluation results to CSV"""
    try:
        history_file = Path("data/evaluation_history.csv")
        history_file.parent.mkdir(exist_ok=True)
        
        entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'method': method,
            'samples': test_samples,
            'semantic_sim': results.get('semantic_similarity', 0.0),
            'faithfulness': results.get('faithfulness', 0.0),
            'hallucination': results.get('hallucination', 0.0),
            'cause_f1': results.get('cause_f1', 0.0),
            'action_f1': results.get('action_f1', 0.0),
            'latency': results.get('latency', 0.0),
            'rouge': results.get('rouge_l', results.get('rouge', 0.0)),
            'bert': results.get('bert_score', results.get('bert', 0.0))
        }
        
        if history_file.exists():
            df = pd.read_csv(history_file)
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        else:
            df = pd.DataFrame([entry])
        
        df.to_csv(history_file, index=False)
        return True
    except:
        return False


# -------------------------
# Mode selector
# -------------------------
mode = st.radio(
    "Choose mode",
    [
        "🔍 Vector RAG - Ask",
        "🧠 Graph RAG - Ask",
        "🧪 SFT Model - Ask",
        "🧪 Vector RAG - Evaluate",
        "🧪 Graph RAG - Evaluate",
        "🧪 SFT Model - Evaluate",
        "📊 View History"
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
if mode == "🔍 Vector RAG - Ask":

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


elif mode == "🧠 Graph RAG - Ask":

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

elif mode == "🧪 SFT Model - Ask":

    query = st.text_input("Enter your question")

    if st.button("🧠 Ask with SFT Model"):

        answer = sft_infer(query)

        st.subheader("✅ Graph RAG Answer")
        st.write(answer)

elif mode == "🧪 Vector RAG - Evaluate":

    st.subheader("🧪 Vector RAG Evaluation")

    test_size = st.slider(
        "Test set size",
        min_value=1,
        max_value=144,
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
        col3.metric("rouge_l", f"{results['rouge_l']:.3f}")
        col3.metric("bert_score", f"{results['bert_score']:.3f}")
        
        # Save to history
        if save_to_history("Vector RAG", test_size, results):
            st.info("✅ Saved to history")


elif mode == "🧪 Graph RAG - Evaluate":

    from src.graph_rag.evaluation.metrics import run_graph_evaluation

    test_size = st.slider("Test set size", 1, 144, 10)

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
        col3.metric("rouge", f"{results['rouge']:.3f}")
        col3.metric("bert", f"{results['bert']:.3f}")
        
        # Save to history
        if save_to_history("Graph RAG", test_size, results):
            st.info("✅ Saved to history")


elif mode == "🧪 SFT Model - Evaluate":
    
    st.subheader("🧪 SFT Model Evaluation")
    
    pred_path = Path("logs/sft_predictions_all_data.jsonl")
    
    if not pred_path.exists():
        st.warning("⚠️ No predictions found. Run inference first:")
        st.code("python -m src.sft.inference.generate", language="bash")
    else:
        with pred_path.open() as f:
            total_predictions = sum(1 for _ in f)
        
        st.info(f"Found {total_predictions} predictions in logs/sft_predictions.jsonl")
        
        test_size = st.slider(
            "Test set size",
            min_value=1,
            max_value=total_predictions,
            value=min(10, total_predictions),
            step=1
        )
        
        if st.button("▶ Run SFT Evaluation"):
            
            with st.spinner("Evaluating SFT predictions..."):
                results = run_sft_evaluation(test_size)
            
            st.success("SFT Evaluation Complete ✅")
            
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Semantic Similarity", f"{results['semantic_similarity']:.3f}")
            col1.metric("Faithfulness", f"{results['faithfulness']:.3f}")
            col1.metric("Hallucination", f"{results['hallucination']:.3f}")
            
            col2.metric("Cause F1", f"{results['cause_f1']:.3f}")
            col2.metric("Action F1", f"{results['action_f1']:.3f}")
            
            col3.metric("Latency", "N/A (pre-computed)")
            col3.metric("ROUGE-L", f"{results['rouge_l']:.3f}")
            col3.metric("BERTScore", f"{results['bert_score']:.3f}")
            
            # Save to history
            if save_to_history("SFT", test_size, results):
                st.info("✅ Saved to history")


elif mode == "📊 View History":
    
    st.subheader("📊 Evaluation History")
    
    history_file = Path("data/evaluation_history.csv")
    
    if not history_file.exists():
        st.info("No history yet. Run evaluations to populate!")
    else:
        df = pd.read_csv(history_file)
        
        if df.empty:
            st.info("History is empty.")
        else:
            st.dataframe(df, use_container_width=True, height=400)
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Runs", len(df))
            col2.metric("Methods", df['method'].nunique())
            col3.metric("Total Samples", int(df['samples'].sum()))
            
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv,
                file_name=f"history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
