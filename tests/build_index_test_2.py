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
    # 1️⃣ Ingest PDFs
    ingestor = PDFIngestor(pdf_folder)
    raw_docs = ingestor.ingest_folder()  # returns {filename: [ {page_number, text, figures}, ...]}

    # 2️⃣ Chunk text
    chunker = TextChunker(chunk_size=1000, overlap=200)
    chunk_dicts = chunker.create_chunks_from_docs(raw_docs)

    # Convert to LangChain Document objects
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "id": chunk["id"],
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "figures": ", ".join(chunk["figures"]) if chunk["figures"] else None ,
            }
        )
        for chunk in chunk_dicts
    ]

    print(f"✅ Loaded {len(documents)} chunks from {len(raw_docs)} PDFs.")

    # 3️⃣ Create embeddings
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4️⃣ Build and persist Chroma collection
    chroma_store = ChromaVectorStore(persist_dir)
    chroma_store.create_collection_from_chunks(chunks=documents, embedding=embedder.embedder, persist=True)
    print(f"✅ Chroma vector store created and persisted at {persist_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_folder", required=True, help="Folder with PDF files")
    parser.add_argument("--persist_dir", default="chroma_store", help="Directory to store Chroma DB")
    args = parser.parse_args()

    build(args.pdf_folder, args.persist_dir)
