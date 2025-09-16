# build_index.py
import os
import argparse
from dotenv import load_dotenv
from langchain.schema import Document
from src.data_ingestion import PDFIngestor
from src.chunker import TextChunker
from src.embedder import Embedder
from src.vectorstore_chroma import ChromaVectorStore

load_dotenv()

def build(pdf_folder, persist_dir):
    # 1️ Ingest PDFs
    ingestor = PDFIngestor(pdf_folder)
    raw_docs = ingestor.ingest_folder()  # {filename: [ {page_number, text, figures}, ...]}

    # 2️ Chunk text
    chunker = TextChunker(chunk_size=500, overlap=100)
    chunk_dicts = chunker.create_chunks_from_docs(raw_docs)

    # 3️ Convert to LangChain Document objects with clean metadata
    documents = []
    for chunk in chunk_dicts:
        meta = {
            "id": chunk["id"],
            "source": chunk["source"],
            "page_number": chunk["page_number"],
        }
        if chunk.get("figures"):  # only include if not empty
            meta["figures"] = ", ".join(chunk["figures"])

        documents.append(
            Document(
                page_content=chunk["text"],
                metadata=meta
            )
        )

    print(f"✅ Loaded {len(documents)} chunks from {len(raw_docs)} PDFs.")

    # 4️ Create embeddings (consistent with app.py)
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 5️ Build and persist Chroma collection
    chroma_store = ChromaVectorStore(persist_dir)
    vectordb = chroma_store.create_collection_from_chunks(
        chunks=documents,
        embedding=embedder.embedder,
        persist=True
    )

    print(f"✅ Chroma vector store created and persisted at {persist_dir}")
    print("📦 Docs inside:", vectordb._collection.count())
    print("🔎 Sample metadata:", documents[0].metadata if documents else "No docs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_folder", required=True, help="Folder with PDF files")
    parser.add_argument("--persist_dir", default="chroma_store", help="Directory to store Chroma DB")
    args = parser.parse_args()

    build(args.pdf_folder, args.persist_dir)
