import os
import argparse
import glob
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from src.vectorstore_chroma import ChromaVectorStore

load_dotenv()

def load_pdfs(pdf_folder):
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    documents = []
    for file in pdf_files:
        loader = PyPDFLoader(file)
        docs = loader.load()
        documents.extend(docs)
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def build(pdf_folder, persist_dir):
    # Load PDFs
    docs = load_pdfs(pdf_folder)
    print(f"Loaded {len(docs)} documents from PDFs.")

    # Split into chunks
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    # Embeddings
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build Chroma collection
    chroma_store = ChromaVectorStore(persist_dir)
    chroma_store.create_collection_from_chunks(chunks=chunks, embedding=embedder, persist=True)
    print(f"✅ Chroma vector store created and persisted at {persist_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_folder", required=True, help="Folder with PDF files")
    parser.add_argument("--persist_dir", default="chroma_store", help="Directory to store Chroma DB")
    args = parser.parse_args()

    build(args.pdf_folder, args.persist_dir)
