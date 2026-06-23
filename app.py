import streamlit as st
import requests
from typing import List, Dict, Any

# Config
st.set_page_config(
    page_title="MARRA Retrieval Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
import os
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# Inject Custom CSS for Premium Design & Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Title and Header Gradient */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }
    .header-title {
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-weight: 300;
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Styled Source Chunk Cards (Glassmorphism look) */
    .source-card {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .source-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06);
    }
    .source-title {
        font-weight: 600;
        font-size: 0.9rem;
        color: #1e3c72;
        margin-bottom: 0.4rem;
    }
    .source-text {
        font-size: 0.85rem;
        line-height: 1.4;
        color: #333333;
    }
    
    /* Dark Mode Source Card Text overrides */
    @media (prefers-color-scheme: dark) {
        .source-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .source-title {
            color: #64b5f6;
        }
        .source-text {
            color: #e0e0e0;
        }
    }
    
    /* File Type Badge Styling */
    .file-type-container {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 15px;
    }
    .file-type-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-color);
        opacity: 0.85;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .file-badge-list {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        margin-bottom: 8px;
    }
    .file-badge-list:last-child {
        margin-bottom: 0;
    }
    .file-badge {
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.7rem;
        background-color: rgba(128, 128, 128, 0.1);
        color: var(--text-color);
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid rgba(128, 128, 128, 0.15);
        font-weight: bold;
    }
    
    /* Hide default file uploader info text to avoid double info and compression */
    [data-testid="stFileUploader"] small {
        display: none !important;
    }
</style>

""", unsafe_allow_html=True)

# App Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/plasticine/200/brain.png", width=100)
    st.markdown("## MARRA Agent")
    st.markdown("A local-first hybrid RAG reasoning engine driven by LangGraph, Qdrant, and Ollama.")
    
    # Ping Backend Health Check
    st.markdown("---")
    st.markdown("### System Status")
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=2.0)
        if health_resp.status_code == 200:
            st.success("🟢 API Gateway: Connected")
        else:
            st.error(f"🔴 API Gateway: Error {health_resp.status_code}")
    except requests.exceptions.RequestException:
        st.error("🔴 API Gateway: Offline")
        
    st.markdown("---")
    st.markdown("### Document Ingestion")
    
    # Clearly show the accepted file types, grouped by category
    st.markdown("""
    <div class="file-type-container">
        <div style="font-size: 0.8rem; font-weight: 700; margin-bottom: 8px; color: var(--text-color);">
            Supported Formats (Max 200MB)
        </div>
        <div class="file-type-label">📄 Documents</div>
        <div class="file-badge-list">
            <span class="file-badge">TXT</span>
            <span class="file-badge">MD</span>
        </div>
        <div class="file-type-label" style="margin-top: 8px;">🖼️ Images</div>
        <div class="file-badge-list">
            <span class="file-badge">PNG</span>
            <span class="file-badge">JPG</span>
            <span class="file-badge">JPEG</span>
            <span class="file-badge">WEBP</span>
        </div>
        <div class="file-type-label" style="margin-top: 8px;">🎵 Audio</div>
        <div class="file-badge-list">
            <span class="file-badge">MP3</span>
            <span class="file-badge">WAV</span>
        </div>
        <div class="file-type-label" style="margin-top: 8px;">🎥 Video</div>
        <div class="file-badge-list">
            <span class="file-badge">MP4</span>
            <span class="file-badge">MOV</span>
            <span class="file-badge">MPEG</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Document or Media File")
    ingest_mode = st.radio("Ingestion Mode", ["Append", "Overwrite"], index=0)
    
    if st.button("🚀 Ingest Document", use_container_width=True):
        if uploaded_file is not None:
            overwrite_val = (ingest_mode == "Overwrite")
            with st.spinner("Ingesting document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "text/plain")}
                    data = {"overwrite": str(overwrite_val).lower()}
                    response = requests.post(f"{API_URL}/ingest", files=files, data=data, timeout=300.0)
                    if response.status_code == 200:
                        st.success(response.json().get("message", "Success!"))
                    else:
                        st.error(f"Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to server: {e}")
        else:
            st.warning("Please choose a file first.")

    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Header Design
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🧠 MARRA Retrieval Platform</h1>
    <div class="header-subtitle">Phase 4: Decoupled API Service & Streamlit RAG Interface</div>
</div>
""", unsafe_allow_html=True)

# Initialize Session State Messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Messages from Session State
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If assistant message has sources, render them beautifully
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 View Grounded Context Sources"):
                for idx, src in enumerate(msg["sources"]):
                    metadata = src.get("metadata", {})
                    # Format a nice title or fallback to doc index
                    title = metadata.get("file_name", metadata.get("source", f"Document Context #{idx+1}"))
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">{title} (Chunk ID: {metadata.get('chunk_id', 'N/A')})</div>
                        <div class="source-text">{src.get('text', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

# Main Chat Input
if user_query := st.chat_input("Ask MARRA a question..."):
    # Guard: reject empty/whitespace queries
    if not user_query.strip():
        st.warning("⚠️ Please enter a question before submitting.")
        st.stop()
    
    # 1. Render user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # 2. Setup call to FastAPI Backend
    # Format conversational history for backend model (Pydantic ChatRequest history list)
    # We filter messages to only pass role & content (excluding metadata like sources)
    history_payload = []
    for m in st.session_state.messages[:-1]:
        history_payload.append({
            "role": m["role"],
            "content": m["content"]
        })
        
    payload = {
        "query": user_query,
        "history": history_payload
    }
    
    # 3. Call backend inside st.spinner
    answer = None
    sources = []
    
    with st.spinner("Synthesizing research..."):
        try:
            response = requests.post(f"{API_URL}/chat", json=payload, timeout=120.0)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                sources = data.get("sources", [])
            elif response.status_code == 503:
                st.error("⚠️ **Service Unavailable (503)**: The backend could not establish a connection to local models (Ollama) or database (Qdrant). Please verify that Docker, Qdrant, and Ollama are running.")
            else:
                st.error(f"❌ **API Error ({response.status_code})**: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ **Connection Refused**: Cannot connect to the FastAPI server on port 8000. Please ensure the backend is running.")
        except Exception as e:
            st.error(f"❌ **Unexpected Client Error**: {e}")
            
    # 4. Render and record assistant response
    if answer is not None:
        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                with st.expander("📚 View Grounded Context Sources"):
                    for idx, src in enumerate(sources):
                        metadata = src.get("metadata", {})
                        title = metadata.get("file_name", metadata.get("source", f"Document Context #{idx+1}"))
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">{title} (Chunk ID: {metadata.get('chunk_id', 'N/A')})</div>
                            <div class="source-text">{src.get('text', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
