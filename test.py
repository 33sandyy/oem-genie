# build_index.py
import os
import argparse
from dotenv import load_dotenv
from langchain.schema import Document
from src.data_ingestion import PDFVisualOrderExtractor
from src.chunker import TextChunker
from src.embedder import Embedder
from src.vectorstore_chroma import ChromaVectorStore

if __name__=="__main__":
    pdf_folder="data"
    OUTPUT_PATH="outputs/raw_data.json"
    pdf_reader=PDFVisualOrderExtractor()
    raw_docs = pdf_reader.process_folder(pdf_folder,OUTPUT_PATH)
    
    # raw_docs = ingestor.ingest_folder()  
    # print(raw_docs)
    # ingestor.save_to_json(raw_docs,"outputs/raw_data.json")

    # # 2️ Chunk text
    chunker = TextChunker(chunk_size=800, overlap=200)
    chunk_dicts = chunker.create_chunks_from_docs(raw_docs)
    chunker.save_chunks(chunk_dicts,"outputs/chunks.json")