import os
from langchain_chroma import Chroma
from langchain.schema import Document

class ChromaVectorStore:
    def __init__(self, persist_dir="chroma_store"):
        self.persist_dir = persist_dir
        self.vectordb = None

    def create_collection_from_chunks(self, chunks, embedding, persist=True):
        # Ensure chunks are Document objects
        documents = [
            chunk if isinstance(chunk, Document) else
            Document(page_content=chunk["page_content"], metadata=chunk.get("metadata", {}))
            for chunk in chunks
        ]

        self.vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=self.persist_dir  # automatically persists
        )

        # ✅ No need to call .persist()
        return self.vectordb

    def load_collection(self, embedding):
        self.vectordb = Chroma(
            embedding_function=embedding,
            persist_directory=self.persist_dir
        )
        return self.vectordb

    def similarity_search(self, query, k=3):
        if self.vectordb is None:
            raise ValueError("Vector store not loaded. Call load_collection() first.")
        return self.vectordb.similarity_search(query, k=k)
