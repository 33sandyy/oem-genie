import os
import json
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from io import BytesIO
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.embedder import Embedder
from src.vectorstore_chroma import ChromaVectorStore

# -------------------- CONFIG --------------------
load_dotenv()

# Embedding and Chroma
embedder = Embedder(model_name="sentence-transformers/msmarco-distilbert-base-v4")
chroma_store = ChromaVectorStore("chroma_store")
vectordb = chroma_store.load_collection(embedder.embedder)

# Logging
LOG_BASE_DIR = "logs"
Path(LOG_BASE_DIR).mkdir(parents=True, exist_ok=True)

# Fixed constants (as per your request)
COMBINE_SCORE_WEIGHT = 0.6
BM25_WEIGHT = 0.5
TOP_K = 5
CROSS_ENCODER_THRESHOLD = 0.4


# -------------------- CROSS-ENCODER --------------------
@st.cache_resource
def load_cross_encoder():
    # Fixed device to CPU to avoid meta tensor issue
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

cross_encoder = load_cross_encoder()


# -------------------- HELPERS --------------------
def setup_logging_directory():
    today = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path(LOG_BASE_DIR) / today
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def log_query_and_results(query, results, log_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"query_{timestamp}.json"
    formatted = []
    for i, doc in enumerate(results, 1):
        formatted.append({
            "chunk_number": i,
            "content": doc.page_content,
            "metadata": {
                "source": doc.metadata.get("source", "N/A"),
                "page_number": doc.metadata.get("page_number", "N/A"),
                "start_line_number": doc.metadata.get("start_line_number", "N/A"),
                "end_line_number": doc.metadata.get("end_line_number", "N/A"),
            },
        })
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "retrieved_chunks_count": len(results),
            "retrieved_context": formatted,
        }, f, indent=2, ensure_ascii=False)
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

    for i, doc in enumerate(results, 1):
        context = f"Chunk {i} | Source: {doc.metadata.get('source', 'N/A')} | Page: {doc.metadata.get('page_number', 'N/A')}"
        pdf.drawString(50, y, context)
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


def bm25_retrieve(query, corpus):
    tokenized_corpus = [doc.page_content.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())
    return scores


# -------------------- HYBRID SEARCH PIPELINE --------------------
def hybrid_search_with_rerank(query):
    # Step 1: Retrieve from vector DB
    docs_with_scores = vectordb.similarity_search_with_score(query, k=TOP_K)
    docs = [doc for doc, _ in docs_with_scores]

    # Step 2: BM25 lexical match
    bm25_scores = bm25_retrieve(query, docs)

    # Step 3: Normalize & combine scores
    hybrid_scores = []
    for (doc, vec_score), bm25_score in zip(docs_with_scores, bm25_scores):
        combined = (COMBINE_SCORE_WEIGHT * vec_score) + (BM25_WEIGHT * bm25_score)
        hybrid_scores.append((doc, combined))

    # Step 4: Cross-Encoder rerank (semantic verification)
    cross_inputs = [(query, doc.page_content) for doc, _ in hybrid_scores]
    ce_scores = cross_encoder.predict(cross_inputs)

    # Step 5: Filter by threshold
    final_results = []
    for (doc, combined), ce_score in zip(hybrid_scores, ce_scores):
        if ce_score >= CROSS_ENCODER_THRESHOLD:
            final_results.append((doc, ce_score))

    # Sort final results by cross-encoder confidence
    final_results = sorted(final_results, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in final_results[:TOP_K]]


# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Hybrid Knowledge Base Search", page_icon="🔍", layout="wide")
st.title("🚗 Automotive Knowledge Base Search")
st.caption("Hybrid (Semantic + Lexical) Search with Cross-Encoder Re-Ranking")

query = st.text_input("Enter your search query:")

if query:
    log_dir = setup_logging_directory()
    with st.spinner("Retrieving best matching chunks..."):
        final_results = hybrid_search_with_rerank(query)

    if not final_results:
        st.warning("⚠️ No relevant chunks found after hybrid verification.")
    else:
        log_file = log_query_and_results(query, final_results, log_dir)
        st.success(f"✅ Retrieved {len(final_results)} verified chunks. Logged to: {log_file}")

        pdf_buffer = generate_pdf(final_results, query)

        for i, doc in enumerate(final_results, 1):
            st.markdown(f"### 📄 Chunk {i}")
            st.text_area("Content", doc.page_content, height=150, key=f"chunk_{i}")
            cols = st.columns(4)
            cols[0].metric("Source", doc.metadata.get("source", "N/A"))
            cols[1].metric("Page", doc.metadata.get("page_number", "N/A"))
            cols[2].metric("Start Line", doc.metadata.get("start_line_number", "N/A"))
            cols[3].metric("End Line", doc.metadata.get("end_line_number", "N/A"))
            st.markdown("---")
