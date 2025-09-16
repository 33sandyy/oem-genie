from src.embedder import Embedder
from src.vectorstore_chroma import ChromaVectorStore

def test_query(persist_dir="chroma_store_test"):
    embedder = Embedder()
    chroma_store = ChromaVectorStore(persist_dir)
    vectordb = chroma_store.load_collection(embedder.embedder)

    print("📦 Docs in Chroma:", vectordb._collection.count())

    query = "How to wear seatbelt properly?"
    results = vectordb.similarity_search(query, k=3)

    if not results:
        print("⚠️ No relevant chunks found!")
    else:
        for i, doc in enumerate(results, start=1):
            print(f"\n--- Chunk {i} ---")
            print(doc.page_content[:200], "...")
            print("Source:", doc.metadata.get("source", "N/A"), " | Page:", doc.metadata.get("page_number", "N/A"))

if __name__ == "__main__":
    test_query()
