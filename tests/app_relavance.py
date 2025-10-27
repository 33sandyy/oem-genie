import os
import json
import streamlit as st
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from io import BytesIO
from pathlib import Path

from src.embedder import Embedder
from src.vectorstore_chroma import ChromaVectorStore

# Load API keys (not needed but keeping for compatibility)
load_dotenv()

# Embeddings + Chroma
embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_store = ChromaVectorStore("chroma_store")
vectordb = chroma_store.load_collection(embedder.embedder)

# Configure logging directory
LOG_BASE_DIR = "logs"
Path(LOG_BASE_DIR).mkdir(parents=True, exist_ok=True)

def setup_logging_directory():
    """Create day-wise logging directory structure."""
    today = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path(LOG_BASE_DIR) / today
    
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        if not log_dir.exists():
            raise OSError(f"Failed to create directory: {log_dir}")
    except Exception as e:
        st.error(f"Error creating log directory: {e}")
        log_dir = Path(".")
    return log_dir

def log_query_and_results(query, results, log_dir):
    """Log query and retrieved context in formatted JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"query_{timestamp}.json"
    
    formatted_results = []
    for i, doc in enumerate(results, start=1):
        formatted_results.append({
            "chunk_number": i,
            "content": doc.page_content,
            "metadata": {
                "source": doc.metadata.get('source', 'N/A'),
                "page_number": doc.metadata.get('page_number', 'N/A'),
                "start_line_number": doc.metadata.get('start_line_number', 'N/A'),
                "end_line_number": doc.metadata.get('end_line_number', 'N/A')
            }
        })
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "retrieved_chunks_count": len(results),
        "retrieved_context": formatted_results
    }
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving log file: {e}")
        return None
    return log_file

def create_daily_summary(log_dir):
    """Create a summary file for the day's queries."""
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

def generate_pdf(results, query):
    """Generate a PDF with the retrieved context."""
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

    for i, doc in enumerate(results, start=1):
        context_text = (
            f"Chunk {i} | Source: {doc.metadata.get('source', 'N/A')} | "
            f"Page: {doc.metadata.get('page_number', 'N/A')} | "
            f"Lines: {doc.metadata.get('start_line_number', 'N/A')}-{doc.metadata.get('end_line_number', 'N/A')}"
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


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Knowledge Base Search", page_icon="🔍", layout="wide")
st.title("Knowledge Base Search")
st.write("Search and retrieve relevant chunks from your safety manuals.")

query = st.text_input("Enter your search query:")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    RELEVANCE_THRESHOLD = st.slider("Relevance Threshold", 0.1, 0.9, 0.35, 0.05)
    st.caption("Higher = stricter relevance filtering")

if query:
    try:
        log_dir = setup_logging_directory()
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"Error setting up logging: {e}")
        log_dir = Path(".")

    with st.spinner("Searching knowledge base..."):
        results_with_scores = vectordb.similarity_search_with_score(query, k=3)

    # ✅ Filter results by similarity score
    filtered_results = [doc for doc, score in results_with_scores if score >= RELEVANCE_THRESHOLD]

    if not filtered_results:
        st.warning("⚠️ No highly relevant context found in manuals. Try rephrasing your query.")
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "retrieved_chunks_count": 0,
            "retrieved_context": [],
            "note": "All retrieved chunks were below the relevance threshold"
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = log_dir / f"query_{timestamp}.json"
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)
            st.info(f"Query logged to: {log_file}")
        except Exception as e:
            st.error(f"Error saving log: {e}")

    else:
        log_file = log_query_and_results(query, filtered_results, log_dir)
        if log_file:
            summary_file = create_daily_summary(log_dir)
            st.success(f"Query logged to: {log_file}")

        st.subheader(f"✅ Retrieved {len(filtered_results)} Highly Relevant Chunks")

        pdf_buffer = generate_pdf(filtered_results, query)

        st.markdown("---")
        for i, doc in enumerate(filtered_results, start=1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### 📄 Chunk {i}")
                with col2:
                    st.caption(f"Relevance Rank: {i}")

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
    st.header("📊 Logging Info")
    if Path(LOG_BASE_DIR).exists():
        today = datetime.now().strftime("%Y-%m-%d")
        today_dir = Path(LOG_BASE_DIR) / today
        
        if today_dir.exists():
            query_count = len(list(today_dir.glob("query_*.json")))
            st.metric("Today's Queries", query_count)
            
            summary_path = today_dir / "daily_summary.json"
            if summary_path.exists():
                st.download_button(
                    label="📄 Download Today's Summary",
                    data=summary_path.read_text(encoding='utf-8'),
                    file_name=f"summary_{today}.json",
                    mime="application/json"
                )
        
        logged_dates = sorted([d.name for d in Path(LOG_BASE_DIR).iterdir() if d.is_dir()], reverse=True)
        if logged_dates:
            st.subheader("📅 Logged Dates")
            for date in logged_dates[:7]:
                date_dir = Path(LOG_BASE_DIR) / date
                query_count = len(list(date_dir.glob("query_*.json")))
                st.text(f"{date}: {query_count} queries")
