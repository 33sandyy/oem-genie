# src/embedder.py
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings 

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedder = HuggingFaceEmbeddings(model_name=self.model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # LangChain HuggingFaceEmbeddings provides embed_documents()
        return self.embedder.embed_documents(texts)

    def embed_query(self, text: str):
        return self.embedder.embed_query(text)
