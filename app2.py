# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from src.embedder import Embedder   # use your wrapper (same as build_index2)
from src.vectorstore_chroma import ChromaVectorStore

# Load API keys
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in .env file")
    st.stop()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Use the same embedder as in build_index.py
embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_store = ChromaVectorStore("chroma_store")
vectordb = chroma_store.load_collection(embedder.embedder)

# Streamlit UI
st.set_page_config(page_title="Maintenance Assistant", page_icon="🔧", layout="wide")
st.title("🔧 Maintenance Assistant")
st.write("Ask questions based on your uploaded manuals.")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Searching knowledge base..."):
        results = vectordb.similarity_search(query, k=3)
        st.sidebar.write("Sample search for 'seatbelt':")
        for d in results:
            st.sidebar.write(d.page_content[:150] + "...")

    if not results:
        st.warning("⚠️ No relevant context found in manuals.")
    else:
        # Build context from retrieved chunks
        context = "\n".join([doc.page_content for doc in results])

        with st.spinner("Generating response..."):
            chat_completion = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "You are a helpful maintenance assistant. Answer based strictly on the provided context. If answer isn't in context, reply: 'I couldn't find that in the manuals.'"},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ]
            )

        answer = chat_completion.choices[0].message.content
        st.subheader("✅ AI Answer")
        st.write(answer)

        with st.expander("📖 View retrieved context"):
            for i, doc in enumerate(results):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.caption(f"Source: {doc.metadata.get('source', 'N/A')} | Page: {doc.metadata.get('page_number', 'N/A')}")
