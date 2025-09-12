import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from src.vectorstore_chroma import ChromaVectorStore

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in .env file")
    st.stop()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Embeddings + Chroma
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_store = ChromaVectorStore("chroma_store")
vectordb = chroma_store.load_collection(embedder)

st.set_page_config(page_title="Maintenance Assistant", page_icon="🔧", layout="wide")
st.title("🔧 Maintenance Assistant")
st.write("Ask questions based on your uploaded manuals.")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Searching knowledge base..."):
        results = vectordb.similarity_search(query, k=3)

    context = "\n".join([doc.page_content for doc in results])

    with st.spinner("Generating response..."):
        chat_completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",  # You can change to another Groq-supported model
            messages=[
                {"role": "system", "content": "You are a helpful maintenance assistant. Answer based on context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )

    # answer = chat_completion.choices[0].message["content"]
    answer=chat_completion.choices[0].message.content
    st.subheader("Answer:")
    st.write(answer)

    with st.expander("View retrieved context"):
        for i, doc in enumerate(results):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(doc.page_content)
            st.caption(f"Source: {doc.metadata.get('source', 'N/A')}")
