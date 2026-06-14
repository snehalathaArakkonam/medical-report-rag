# app.py - Medical Report Summarizer with RAG
import streamlit as st
import os
from typing import List
import tempfile

# For PDF handling
from pypdf import PdfReader

# For embeddings and vector store
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# For LangChain-like processing (simple implementation)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# For Gemini API (you'll need to set GOOGLE_API_KEY)
import google.generativeai as genai

st.set_page_config(page_title="Medical Report Summarizer", layout="wide")
st.title("🩺 Medical Report Summarizer (RAG)")

# Sidebar
st.sidebar.header("Instructions")
st.sidebar.info("""
1. Upload your medical report PDF
2. Ask questions like "What is my diagnosis?", "Summarize findings", etc.
3. Get patient-friendly explanations
""")

# API Key
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

api_key = st.sidebar.text_input("Gemini API Key (google.generativeai)", value=st.session_state.gemini_api_key, type="password")
if api_key:
    st.session_state.gemini_api_key = api_key
    genai.configure(api_key=api_key)

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "report_text" not in st.session_state:
    st.session_state.report_text = ""

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file) -> str:
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

# Check if medical report (simple heuristic)
def is_medical_report(text: str) -> bool:
    medical_keywords = ["diagnosis", "patient", "report", "findings", "impression", "lab", "blood", "ct", "mri", "ultrasound"]
    text_lower = text.lower()
    score = sum(1 for kw in medical_keywords if kw in text_lower)
    return score >= 2

# Chunking with overlap
def chunk_text(text: str, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Embed and create FAISS index
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def create_vector_store(chunks: List[str]):
    model = get_embedding_model()
    embeddings = model.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine (after normalization)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, embeddings, model

# Retrieve relevant chunks
def retrieve_relevant_chunks(query: str, top_k=3):
    if st.session_state.vector_store is None:
        return []
    index, _, model = st.session_state.vector_store
    query_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, top_k)
    relevant = [st.session_state.text_chunks[i] for i in indices[0]]
    return relevant

# Generate answer with Gemini
def generate_answer(context: str, question: str) -> str:
    if not st.session_state.gemini_api_key:
        return "Please provide Gemini API key in sidebar."
    
    prompt = f"""You are a helpful medical assistant. Explain the following medical report excerpt in simple, patient-friendly language.
Avoid jargon or explain it. Be empathetic and clear.

Context from report:
{context}

Question: {question}

Provide a concise, easy-to-understand answer:"""

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main UI
uploaded_file = st.file_uploader("Upload Medical Report PDF", type="pdf")

if uploaded_file:
    if st.button("Process Report"):
        with st.spinner("Extracting and processing report..."):
            text = extract_text_from_pdf(uploaded_file)
            if not text:
                st.error("Could not extract text from PDF.")
            elif not is_medical_report(text):
                st.warning("This doesn't appear to be a standard medical report. Proceeding anyway, but results may vary.")
            
            st.session_state.report_text = text
            chunks = chunk_text(text)
            st.session_state.text_chunks = chunks
            
            with st.spinner("Creating embeddings and vector store..."):
                index, embeddings, model = create_vector_store(chunks)
                st.session_state.vector_store = (index, embeddings, model)
            
            st.success(f"Processed {len(chunks)} chunks successfully!")

if st.session_state.report_text:
    st.subheader("Report Preview")
    with st.expander("View extracted text (first 1000 chars)"):
        st.text(st.session_state.report_text[:1000] + "...")

# Query interface
st.subheader("Ask Questions About Your Report")
question = st.text_input("Your question (e.g., 'What is my diagnosis?', 'Summarize key findings')")

if st.button("Get Answer") and question:
    if st.session_state.vector_store is None:
        st.error("Please process a report first.")
    else:
        with st.spinner("Retrieving relevant information..."):
            relevant_chunks = retrieve_relevant_chunks(question, top_k=3)
            context = "\n\n".join(relevant_chunks)
        
        with st.spinner("Generating patient-friendly explanation..."):
            answer = generate_answer(context, question)
        
        st.markdown("### 🤖 Answer")
        st.write(answer)
        
        # Save to history
        st.session_state.query_history.append({"q": question, "a": answer})

# Query History
if st.session_state.query_history:
    st.subheader("Query History")
    for i, entry in enumerate(st.session_state.query_history):
        with st.expander(f"Q: {entry['q'][:80]}..."):
            st.write(entry["a"])

# Download summary
if st.session_state.query_history:
    if st.button("Download Full Summary"):
        summary = "# Medical Report Summary\n\n"
        for entry in st.session_state.query_history:
            summary += f"## Question: {entry['q']}\n\n{entry['a']}\n\n---\n\n"
        
        st.download_button(
            label="📥 Download Summary (Markdown)",
            data=summary,
            file_name="medical_report_summary.md",
            mime="text/markdown"
        )

# Cleanup
if st.button("Reset / Upload New Report"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()