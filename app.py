# app.py
import os
import json
import streamlit as st
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from io import BytesIO
from pathlib import Path
from sentence_transformers import CrossEncoder

from src.embedder import Embedder
from src.vectorstore_chroma import ChromaVectorStore

# ------------------- Configuration -------------------
TOP_K = 10             
CROSS_ENCODER_THRESHOLD = 0.3

load_dotenv()

#  Load Vector Store 
embedder = Embedder(model_name="sentence-transformers/msmarco-distilbert-base-v4")
chroma_store = ChromaVectorStore("chroma_store")
vectordb = chroma_store.load_collection(embedder.embedder)

# Initialize Cross-Encoder 
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ------------------- Logging -------------------
LOG_BASE_DIR = "logs"
Path(LOG_BASE_DIR).mkdir(parents=True, exist_ok=True)

def setup_logging_directory():
    today = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path(LOG_BASE_DIR) / today
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def log_query_and_results(query, results, log_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"query_{timestamp}.json"

    formatted_results = []
    for i, (doc, score) in enumerate(results, start=1):
        formatted_results.append({
            "chunk_number": i,
            "content": doc.page_content,
            "metadata": {
                "source": doc.metadata.get('source', 'N/A'),
                "page_number": doc.metadata.get('page_number', 'N/A'),
                "start_line_number": doc.metadata.get('start_line_number', 'N/A'),
                "end_line_number": doc.metadata.get('end_line_number', 'N/A')
            },
            "cross_encoder_score": float(score)
        })

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "retrieved_chunks_count": len(results),
        "retrieved_context": formatted_results
    }

    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)
    return log_file

def create_daily_summary(log_dir):
    summary_file = log_dir / "daily_summary.json"
    query_files = sorted(log_dir.glob("query_*.json"))

    summary = {
        "date": log_dir.name,
        "total_queries": len(query_files),
        "queries": []
    }

    for query_file in query_files:
        with open(query_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            summary["queries"].append({
                "timestamp": data["timestamp"],
                "query": data["query"],
                "chunks_retrieved": data["retrieved_chunks_count"],
                "log_file": query_file.name
            })

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary_file

#  PDF Generation 
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

    for i, (doc, score) in enumerate(results, start=1):
        context_text = (
            f"Chunk {i} | Source: {doc.metadata.get('source', 'N/A')} | "
            f"Page: {doc.metadata.get('page_number', 'N/A')} | "
            f"Lines: {doc.metadata.get('start_line_number', 'N/A')}-{doc.metadata.get('end_line_number', 'N/A')} | "
            f"Score: {score:.3f}"
        )
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

# Streamlit UI 
st.set_page_config(page_title="Knowledge Base Search", page_icon="🔍", layout="wide")
st.title("Knowledge Base Search")
st.write("Search and retrieve highly relevant chunks from safety manuals using only the reranker.")

query = st.text_input("Enter your search query:")

if query:
    log_dir = setup_logging_directory()
    with st.spinner("Searching knowledge base..."):
        # Step 1: Vector search (top K)
        results_with_scores = vectordb.similarity_search_with_score(query, k=TOP_K)
        docs = [doc for doc, _ in results_with_scores]

        # Step 2: Cross-encoder reranking
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)

        doc_scores = list(zip(docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Filter by fixed threshold
        filtered_results = [item for item in doc_scores if item[1] >= CROSS_ENCODER_THRESHOLD]

    if not filtered_results:
        st.warning("No highly relevant context found. Try rephrasing your query.")
    else:
        st.subheader(f" Retrieved {len(filtered_results)} Relevant Chunks (Reranked)")

        # Log results
        log_file = log_query_and_results(query, filtered_results, log_dir)
        summary_file = create_daily_summary(log_dir)
        st.success(f"Query logged to: {log_file}")

        # PDF download
        pdf_buffer = generate_pdf(filtered_results, query)
        st.download_button(
            label="Download Results as PDF",
            data=pdf_buffer,
            file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

        # Display results
        for i, (doc, score) in enumerate(filtered_results, start=1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### Chunk {i}")
                with col2:
                    st.metric("Score", f"{score:.3f}")

                st.markdown("**Content:**")
                st.text_area(
                    f"Chunk {i} content",
                    value=doc.page_content,
                    height=150,
                    key=f"chunk_{i}",
                    label_visibility="collapsed"
                )

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Source", doc.metadata.get('source', 'N/A'))
                with col2:
                    st.metric("Page", doc.metadata.get('page_number', 'N/A'))
                with col3:
                    st.metric("Start Line", doc.metadata.get('start_line_number', 'N/A'))
                with col4:
                    st.metric("End Line", doc.metadata.get('end_line_number', 'N/A'))
                st.markdown("---")

# Sidebar logging info
with st.sidebar:
    st.header("Logging Info")
    if Path(LOG_BASE_DIR).exists():
        today = datetime.now().strftime("%Y-%m-%d")
        today_dir = Path(LOG_BASE_DIR) / today
        if today_dir.exists():
            query_count = len(list(today_dir.glob("query_*.json")))
            st.metric("Today's Queries", query_count)
            summary_path = today_dir / "daily_summary.json"
            if summary_path.exists():
                st.download_button(
                    label="Download Today's Summary",
                    data=summary_path.read_text(encoding='utf-8'),
                    file_name=f"summary_{today}.json",
                    mime="application/json"
                )
