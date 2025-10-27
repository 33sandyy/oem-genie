import os
import json
import streamlit as st
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from io import BytesIO
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder

from src.embedder import Embedder
from src.vectorstore_chroma import ChromaVectorStore

# -----------------------------
# Load API keys (not needed here)
# -----------------------------
load_dotenv()

# -----------------------------
# Initialize Embeddings + Vector DB
# -----------------------------
embedder = Embedder(model_name="sentence-transformers/msmarco-distilbert-base-v4")
chroma_store = ChromaVectorStore("chroma_store")
vectordb = chroma_store.load_collection(embedder.embedder)

# -----------------------------
# Utility Functions
# -----------------------------
LOG_BASE_DIR = "logs"
Path(LOG_BASE_DIR).mkdir(parents=True, exist_ok=True)

def setup_logging_directory():
    today = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path(LOG_BASE_DIR) / today
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def tokenize(text):
    return text.lower().split()

def apply_bm25_reranking(query, docs_with_scores, bm25_weight=0.5):
    if not docs_with_scores:
        return []
    
    docs = [doc for doc, _ in docs_with_scores]
    similarity_scores = [score for _, score in docs_with_scores]

    tokenized_corpus = [tokenize(doc.page_content) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalize scores
    bm25_scores = np.array(bm25_scores)
    bm25_scores_norm = bm25_scores / bm25_scores.max() if bm25_scores.max() > 0 else bm25_scores
    similarity_scores_norm = np.array(similarity_scores) / max(similarity_scores) if max(similarity_scores) > 0 else np.array(similarity_scores)

    combined_scores = bm25_weight * bm25_scores_norm + (1 - bm25_weight) * similarity_scores_norm

    results = [
        (doc, combined_scores[i], similarity_scores[i], bm25_scores_norm[i])
        for i, doc in enumerate(docs)
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def log_query_and_results(query, results, log_dir, score_info=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"query_{timestamp}.json"
    
    formatted_results = []
    for i, item in enumerate(results, start=1):
        if score_info:
            doc, combined_score, sim_score, bm25_score = item
            result_entry = {
                "chunk_number": i,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "scores": {
                    "combined_score": float(combined_score),
                    "similarity_score": float(sim_score),
                    "bm25_score": float(bm25_score)
                }
            }
        else:
            doc = item
            result_entry = {
                "chunk_number": i,
                "content": doc.page_content,
                "metadata": doc.metadata
            }
        formatted_results.append(result_entry)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "retrieved_chunks_count": len(results),
        "retrieved_context": formatted_results
    }

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)
    return log_file

def generate_pdf(results, query):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    pdf.setTitle("Search Results")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Search Results - Retrieved Context")
    y -= 40

    pdf.setFont("Helvetica", 10)
    pdf.drawString(50, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 30

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Search Query:")
    y -= 20
    pdf.setFont("Helvetica", 11)
    for line in query.split("\n"):
        pdf.drawString(50, y, line)
        y -= 15
    y -= 25

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, f"Retrieved Chunks ({len(results)} results):")
    y -= 25
    pdf.setFont("Helvetica", 10)

    for i, item in enumerate(results, start=1):
        doc = item[0] if isinstance(item, tuple) else item
        context_text = f"Chunk {i} | Source: {doc.metadata.get('source', 'N/A')} | Page: {doc.metadata.get('page_number', 'N/A')}"
        pdf.drawString(50, y, context_text)
        y -= 15
        for line in doc.page_content.split("\n"):
            if y < 80:
                pdf.showPage()
                y = height - 50
            pdf.drawString(60, y, line[:100])
            y -= 12
        y -= 15

    pdf.save()
    buffer.seek(0)
    return buffer

# -----------------------------
# Cross-Encoder Re-Ranker
# -----------------------------
class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, docs):
        pairs = [(query, doc[0].page_content if isinstance(doc, tuple) else doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        reranked = [(docs[i][0], float(scores[i])) for i in range(len(docs))]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Knowledge Base Search", page_icon="🔍", layout="wide")
st.title("🔍 Knowledge Base Search with Hybrid + Cross-Encoder Verification")
st.write("Hybrid retrieval using BM25 + Vector embeddings + Cross-Encoder Re-Ranking for semantic filtering.")

query = st.text_input("Enter your search query:")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Retrieval Settings")
    RELEVANCE_THRESHOLD = st.slider("Combined Score Threshold", 0.0, 1.0, 0.4, 0.05)
    BM25_WEIGHT = st.slider("BM25 Weight", 0.0, 1.0, 0.5, 0.1)
    K_RESULTS = st.slider("Top-K Retrieval", 3, 30, 10, 1)
    CROSS_ENCODER_THRESHOLD = st.slider("Cross-Encoder Min Score", 0.0, 1.0, 0.4, 0.05)
    st.caption("Hybrid = Semantic (Vector) + Lexical (BM25). Cross-Encoder = final semantic verification.")

if query:
    log_dir = setup_logging_directory()

    with st.spinner("Searching knowledge base..."):
        # Step 1: Initial vector search
        results_with_scores = vectordb.similarity_search_with_score(query, k=K_RESULTS)

        # Step 2: Hybrid BM25 reranking
        hybrid_results = apply_bm25_reranking(query, results_with_scores, bm25_weight=BM25_WEIGHT)
        filtered_results = [r for r in hybrid_results if r[1] >= RELEVANCE_THRESHOLD]

        # Step 3: Cross-Encoder re-ranking
        if filtered_results:
            st.info("Applying Cross-Encoder semantic verification...")
            reranker = CrossEncoderReranker()
            reranked = reranker.rerank(query, filtered_results)
            final_results = [(doc, score) for doc, score in reranked if score >= CROSS_ENCODER_THRESHOLD]
        else:
            final_results = []

    if not final_results:
        st.warning("⚠️ No relevant context found. Try lowering the thresholds or rephrasing your query.")
    else:
        st.success(f"✅ Retrieved {len(final_results)} semantically verified chunks.")
        log_file = log_query_and_results(query, hybrid_results, log_dir, score_info=True)
        
        pdf_buffer = generate_pdf(final_results, query)
        st.download_button(
            label="📥 Download Results as PDF",
            data=pdf_buffer,
            file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

        st.markdown("---")
        for i, (doc, score) in enumerate(final_results, start=1):
            st.markdown(f"### 📄 Chunk {i} — Relevance Score: `{score:.3f}`")
            st.text_area(f"Chunk {i}", doc.page_content, height=150, key=f"chunk_{i}")
            meta = doc.metadata
            st.caption(f"Source: {meta.get('source', 'N/A')} | Page: {meta.get('page_number', 'N/A')} | Lines: {meta.get('start_line_number', '?')}–{meta.get('end_line_number', '?')}")
            st.markdown("---")
