# src/reranker.py
from typing import List, Tuple
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Tuple]):
        """
        Re-rank docs using cross-encoder relevance scores.
        Input: docs -> list of (Document, score) or (Document, combined_score,...)
        """
        if not docs:
            return []

        pairs = [(query, doc[0].page_content if isinstance(doc, tuple) else doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        reranked = []
        for i, doc in enumerate(docs):
            reranked.append((doc[0], float(scores[i])))

        # Sort descending by relevance
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
