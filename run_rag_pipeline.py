from src.rag.vectorstore.faiss_store import FaissVectorStore
from src.rag.generator.generate import generate_answer

def main():
    store = FaissVectorStore()
    store.load()
    print("Vector store loaded.")
    query = "During operation, bubble off center y axis, precision loss and transfer errors occur. Identify the problem and suggest corrective actions." 
    print(f"Query: {query!r}")

    print("Retrieving chunks...")
    retrieved_chunks = store.retrieve(query, top_k=5)

    print("Generating answer...")
    answer = generate_answer(query, retrieved_chunks)

    print("\nAnswer:")
    print(answer)

if __name__ == "__main__":
    main()