from src.rag.vectorstore.faiss_store import FaissVectorStore
from src.rag.generator.generate import generate_answer

def main():
    store = FaissVectorStore()
    store.load()
    print("Vector store loaded.")
    #query = input("Ask a question: ")
    query = "Applied standards are .. ?" 
    print(f"Query: {query!r}")

    print("Retrieving chunks...")
    retrieved_chunks = store.retrieve(query, top_k=5)

    print("Generating answer...")
    answer = generate_answer(query, retrieved_chunks)

    print("\nAnswer:")
    print(answer)

if __name__ == "__main__":
    main()