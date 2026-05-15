# MARRA: Hybrid Multi-Agent RAG Research Assistant

MARRA is a privacy-preserving, local-first AI research platform designed to ingest complex documents, retrieve highly relevant context using hybrid search (Semantic + Keyword), and synthesize precise answers via an agentic workflow.

## Key Features
- **100% Local-First:** No data leaves your machine. Operates autonomously without paid cloud APIs.
- **Agentic Orchestration:** Utilizes LangGraph for deterministic state-machine routing (Planner -> Retriever -> Synthesizer).
- **Hybrid Retrieval:** Designed to fuse dense vector search with sparse keyword search (BM25) using Reciprocal Rank Fusion.
- **Strictly Typed:** Adheres to rigorous software engineering standards using Pydantic for all state boundaries.

## Tech Stack
- **Language:** Python 3.11+
- **LLM / Embeddings:** Local Ollama (`llama3.1:8b`, `nomic-embed-text`)
- **Vector Store:** Local Qdrant via Docker
- **Frameworks:** FastAPI, LangGraph, Streamlit
- **Quality & CI:** Pytest, GitHub Actions (Unmocked testing environment)

## Getting Started

### Prerequisites
1. [Docker](https://docs.docker.com/get-docker/)
2. [Ollama](https://ollama.com/)
3. Python 3.11+

### Infrastructure Setup
1. Start Qdrant:
   ```bash
   docker run -d -p 6333:6333 qdrant/qdrant
   ```
2. Start Ollama and pull the required models:
   ```bash
   ollama serve
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

### Installation
1. Clone the repository.
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the infrastructure verification script to ensure your local vector database and LLM engine are reachable:
   ```bash
   python scripts/verify_infra.py
   ```

### Running the Data Ingestion Pipeline
To chunk and embed a text file into Qdrant:
```bash
python scripts/ingest_file.py data/sample.txt
```

### Running Tests
The project relies on strict CI testing without mocking database calls. Ensure your local infrastructure is running.
```bash
pytest tests/
```
