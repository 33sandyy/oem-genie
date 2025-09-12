# src/query_engine.py
from typing import Tuple, List, Dict
from langchain.schema import HumanMessage
import numpy as np

# Use your LLM wrapper of choice; example uses ChatGroq like your prior code.
from langchain_groq import ChatGroq

class OEMQueryEngine:
    def __init__(
        self,
        chroma_collection,
        embedder,           # Embedder instance (langchain embeddings)
        groq_api_key: str,
        llm_model: str = "openai/gpt-oss-120b"
    ):
        self.vectordb = chroma_collection
        self.embedder = embedder
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)

    def retrieve(self, query: str, top_k: int = 3, where: dict = None) -> Tuple[List[Dict], List[Dict]]:
        # Chroma's similarity_search_with_relevance_scores returns Documents; use filter via where if needed
        docs = self.vectordb.similarity_search(query, k=top_k, filter=where or {})
        retrieved_texts = [d.page_content for d in docs]
        retrieved_meta = [d.metadata for d in docs]
        return retrieved_texts, retrieved_meta

    def answer_query(self, query: str, top_k: int = 3, where: dict = None) -> Tuple[str, List[Dict]]:
        retrieved_texts, retrieved_meta = self.retrieve(query, top_k=top_k, where=where)
        context = "\n\n".join(retrieved_texts)

        prompt = f"""You are an automotive service assistant. Answer only using the OEM manual excerpts below. Do not hallucinate. If answer isn't present, say "I couldn't find that in the manuals."

Context:
{context}

Question: {query}

Answer:
"""

        message = HumanMessage(content=prompt)
        resp = self.llm.invoke([message])
        # The returned object is an AIMessage; extract the content string
        # Use .content (most compatible) or .text depending on library version
        answer_text = getattr(resp, "content", None) or getattr(resp, "text", None)
        return answer_text, retrieved_meta
