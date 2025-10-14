# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from io import BytesIO

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

# Streamlit UI
st.set_page_config(page_title="Maintenance Assistant", page_icon="🔧", layout="wide")
st.title("Customer Suppport")
st.write("Ask questions based on your uploaded manuals.")

query = st.text_input("Enter your question:")

def generate_pdf(answer, results, query):
    """Generate a PDF with the AI answer and retrieved context."""
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50
    pdf.setTitle("Maintenance Assistant Answer")

    # Header
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Maintenance Assistant - Answer Report")
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
            f"Line: {doc.metadata.get('line_number', 'N/A')}"
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

if query:
    with st.spinner("Searching knowledge base..."):
        results = vectordb.similarity_search(query, k=3)

    if not results:
        st.warning("⚠️ No relevant context found in manuals.")
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
                    f"Line: {doc.metadata.get('line_number', 'N/A')}"
                )
