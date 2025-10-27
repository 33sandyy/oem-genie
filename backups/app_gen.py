# app.py
import os
import json
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from io import BytesIO
from pathlib import Path

from src.embedder import Embedder
from src.vectorstore_chroma import ChromaVectorStore

# Load API keys
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in .env file")
    st.stop()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Embeddings + Chroma
embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_store = ChromaVectorStore("chroma_store")
vectordb = chroma_store.load_collection(embedder.embedder)

# Configure logging directory
LOG_BASE_DIR = "logs"

# Initialize logging directory at startup
Path(LOG_BASE_DIR).mkdir(parents=True, exist_ok=True)

def setup_logging_directory():
    """Create day-wise logging directory structure."""
    today = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path(LOG_BASE_DIR) / today
    
    # Ensure the directory is created with proper error handling
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Verify the directory was created
        if not log_dir.exists():
            raise OSError(f"Failed to create directory: {log_dir}")
    except Exception as e:
        st.error(f"Error creating log directory: {e}")
        # Fallback to current directory
        log_dir = Path(".")
    
    return log_dir

def log_query_and_results(query, results, answer, log_dir):
    """Log query, retrieved context, and AI answer in formatted JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Added microseconds for uniqueness
    log_file = log_dir / f"query_{timestamp}.json"
    
    # Format retrieved results
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
    
    # Create log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "retrieved_chunks_count": len(results),
        "retrieved_context": formatted_results,
        "ai_answer": answer,
        "model_used": "openai/gpt-oss-120b"
    }
    
    # Save to JSON file with error handling
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
    
    # Get all query logs for the day
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
    
    # Save summary
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return summary_file

def generate_pdf(answer, results, query):
    """Generate a PDF with the AI answer and retrieved context."""
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50
    pdf.setTitle("Customer Support Answer")

    # Header
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Customer Support - Answer Report")
    y -= 40

    # Timestamp
    pdf.setFont("Helvetica", 10)
    pdf.drawString(50, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 30

    # Question
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Question:")
    y -= 20
    pdf.setFont("Helvetica", 11)
    for line in query.split("\n"):
        pdf.drawString(50, y, line)
        y -= 15

    y -= 15

    # AI Answer
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "AI Answer:")
    y -= 20
    pdf.setFont("Helvetica", 11)
    for line in answer.split("\n"):
        if y < 80:  # New page if space runs out
            pdf.showPage()
            y = height - 50
        pdf.drawString(50, y, line)
        y -= 15

    y -= 25

    # Retrieved Context
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Retrieved Context:")
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
            pdf.drawString(60, y, line[:100])  # Wrap long lines
            y -= 12

        y -= 15

    pdf.save()
    buffer.seek(0)
    return buffer

# Streamlit UI
st.set_page_config(page_title="Customer Support", page_icon="🔧", layout="wide")
st.title("Customer Support")
st.write("Ask questions based on your safety manuals.")

query = st.text_input("Enter your question:")

if query:
    # Setup logging - this will create the directory
    try:
        log_dir = setup_logging_directory()
        
        # Verify directory exists
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        
    except Exception as e:
        st.error(f"Error setting up logging: {e}")
        log_dir = Path(".")
    
    with st.spinner("Searching knowledge base..."):
        results = vectordb.similarity_search(query, k=5)

    if not results:
        st.warning("No relevant context found in manuals.")
        
        # Log query with no results
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "retrieved_chunks_count": 0,
            "retrieved_context": [],
            "ai_answer": "No relevant context found in manuals.",
            "model_used": "N/A"
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = log_dir / f"query_{timestamp}.json"
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)
            st.success(f"Query logged to: {log_file}")
        except Exception as e:
            st.error(f"Error saving log: {e}")
            
    else:
        context = "\n".join([doc.page_content for doc in results])

        with st.spinner("Generating response..."):
            chat_completion = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful maintenance assistant. "
                            "Answer strictly using the provided context. "
                            "If the answer is not in context, reply: "
                            "'I couldn't find that in the manuals.'"
                        ),
                    },
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
            )

        answer = chat_completion.choices[0].message.content
        
        # Log query and results
        log_file = log_query_and_results(query, results, answer, log_dir)
        
        if log_file:
            # Update daily summary
            summary_file = create_daily_summary(log_dir)
            st.success(f"Query logged to: {log_file}")
        
        st.subheader("AI Answer")
        st.write(answer)

        # Generate and offer PDF download
        pdf_buffer = generate_pdf(answer, results, query)
        st.download_button(
            label="📥 Download Answer & Context as PDF",
            data=pdf_buffer,
            file_name=f"maintenance_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )

        with st.expander("📖 View Retrieved Context"):
            for i, doc in enumerate(results):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.caption(
                    f"Source: {doc.metadata.get('source', 'N/A')} | "
                    f"Page: {doc.metadata.get('page_number', 'N/A')} | "
                    f"Start Line: {doc.metadata.get('start_line_number', 'N/A')} | "
                    f"End Line: {doc.metadata.get('end_line_number', 'N/A')}"
                )

# Sidebar: Show logging statistics
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
        
        # Show all logged dates
        logged_dates = sorted([d.name for d in Path(LOG_BASE_DIR).iterdir() if d.is_dir()], reverse=True)
        if logged_dates:
            st.subheader("📅 Logged Dates")
            for date in logged_dates[:7]:  # Show last 7 days
                date_dir = Path(LOG_BASE_DIR) / date
                query_count = len(list(date_dir.glob("query_*.json")))
                st.text(f"{date}: {query_count} queries")