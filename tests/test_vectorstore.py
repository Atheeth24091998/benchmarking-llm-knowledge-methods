# tests/test_vectorstore.py
# Remove the sys.path manipulation - conftest.py handles it now
from rag.vectorstore.faiss_store import FaissVectorStore

def test_retrieval():
    store = FaissVectorStore()
    store.load()
    
    results = store.retrieve("maximum wire length", top_k=3)
    print(results)
    assert len(results) > 0
