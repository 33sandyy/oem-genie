import os
from langchain.schema import Document
from src.data_ingestion import PDFIngestor
from src.chunker import TextChunker
from src.embedder import Embedder
from src.vectorstore_chroma import ChromaVectorStore

def test_build_index(pdf_folder="data", persist_dir="chroma_store_test"):
    # 1️⃣ Ingest PDFs
    ingestor = PDFIngestor(pdf_folder)
    raw_docs = ingestor.ingest_folder()

    # 2️⃣ Chunk text
    chunker = TextChunker(chunk_size=500, overlap=100)
    chunk_dicts = chunker.create_chunks_from_docs(raw_docs)

    # 3️⃣ Convert to LangChain Documents with safe metadata
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "id": chunk["id"],
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "figures": ", ".join(chunk.get("figures", [])) if chunk.get("figures") else None,
            },
        )
        for chunk in chunk_dicts
    ]

    print(f"✅ Created {len(documents)} chunks from {len(raw_docs)} PDFs.")

    # 4️⃣ Embedding
    embedder = Embedder()

    # 5️⃣ Build test Chroma store
    chroma = ChromaVectorStore(persist_dir)
    vectordb = chroma.create_collection_from_chunks(chunks=documents, embedding=embedder.embedder, persist=True)

    print(f"✅ Test Chroma store built at {persist_dir}")
    print("📦 Docs inside:", vectordb._collection.count())

if __name__ == "__main__":
    test_build_index()