# 🧠 MARRA: Hybrid Multi-Agent RAG Platform

![Version](https://img.shields.io/badge/version-v1.0.0-blue) ![Python](https://img.shields.io/badge/python-3.11-green) ![License](https://img.shields.io/badge/license-MIT-green)

MARRA (Multi-Agent Retrieval Reasoning Assistant) is a production-grade, local-first RAG platform designed to deliver secure, private, and deterministic document intelligence. It features a decoupled microservices architecture, a deterministic agent state machine orchestrated with LangGraph, hybrid search (dense vector + sparse BM25) utilizing Reciprocal Rank Fusion (RRF), and comprehensive OTLP-compliant observability.

---

## 🏗️ Architecture Overview

The system is split into decoupled layers to ensure strict separation of concerns, scalability, and optimal hardware utilization:

1. **Streamlit UI Layer**: A clean, modern interface presenting interactive agent workflows and grounded source visualizations.
2. **FastAPI API Gateway**: Decoupled backend serving agent invocation endpoints over thread-safe pipelines.
3. **LangGraph State Machine**: Orchestrates search planning, retrieval, and response synthesis inside a deterministic, stateful graph.
4. **Hybrid Retrieval Engine**:
   - **Dense Search**: Semantic vector retrieval matching queries against embeddings stored in Qdrant.
   - **Sparse Search**: Keyword-based search matching queries using a BM25 index.
   - **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse retrieval ranks dynamically.
5. **Observability & Tracing (Arize Phoenix)**: OTLP tracing collector capturing span-level latency, execution logs, and detailed sub-query breakdowns.
6. **Ollama LLM Engine**: Runs natively on the host to leverage metal/GPU acceleration (Metal on macOS, CUDA on Windows/Linux).

### System Topology (ASCII)

```
+-----------------------------------------------------------------------------------+
|                                 HOST ENVIRONMENT                                  |
|                                                                                   |
|  +---------------------------+             +-----------------------------------+  |
|  |     Streamlit Container   |             |           Ollama Service          |  |
|  |                           |             |        (Metal/CUDA Native)        |  |
|  |     +---------------+     |             |        +-----------------+        |  |
|  |     |  app.py (UI)  |     |             |        |  nomic-embed    |        |  |
|  |     +-------+-------+     |             |        +-----------------+        |  |
|  |             |             |             |        |  llama3.1:8b    |        |  |
|  +-------------|-------------+             |        +--------^--------+        |  |
|     (Browser)  | API Request               +-----------------|-----------------+  |
|     Port 8501  | Port 8000                                   |                    |
|                v                                             | OLLAMA_HOST        |
|  +---------------------------+                               | (host.docker.internal)
|  |    FastAPI Container      |                               |                    |
|  |                           |                               |                    |
|  |    +-----------------+    |                               |                    |
|  |    |  server.py API  |    |                               |                    |
|  |    +--------+--------+    |                               |                    |
|  |             |             |                               |                    |
|  |   (Invokes State Machine) |                               |                    |
|  |             v             |                               |                    |
|  |    +-----------------+    |                               |                    |
|  |    |    LangGraph    |----+-------------------------------+                    |
|  |    |  State Machine  |    |   LLM & Embeddings Query                           |
|  |    +--------+--------+    |                                                    |
|  |             |             |                                                    |
|  |             +-------------+--------+                                           |
|  |             | Retrieves Context    | Traces Span Metrics                       |
|  |             v                      v                                           |
|  +-------------|----------------------|-------------------------------------------+
|                |                      |
|                | Port 6333            | Port 4318 (OTLP HTTP)
|                v                      v
|  +---------------------------+  +---------------------------+
|  |     Qdrant Container      |  |     Phoenix Container     |
|  |                           |  |                           |
|  |  +---------------------+  |  |  +---------------------+  |
|  |  |   Vector Database   |  |  |  |   Arize Phoenix     |  |
|  |  +---------------------+  |  |  +---------------------+  |
|  +---------------------------+  +---------------------------+
```

---

## 🛠️ Enterprise Tech Stack

- **Orchestration**: LangGraph (Deterministic state routing)
- **API Gateway**: FastAPI + Uvicorn (Lifespan resource management, thread execution pooling)
- **Frontend**: Streamlit (Reactive component architecture with dark/light mode compatibility)
- **Vector Database**: Qdrant (Dense vector store, Dockerized)
- **Sparse Search**: Rank-BM25 (Keyword-based token score matching, persisted to `data/bm25_index.pkl`)
- **Local LLM**: Ollama (`llama3.1:8b` — query planning, runs on host GPU/CPU)
- **Cloud AI**: Google Gemini (`gemini-embedding-2` for 768-dim Matryoshka embeddings, `gemini-3.5-flash` for multimodal synthesis)
- **Observability**: Arize Phoenix + OpenTelemetry (OTLP exporters)
- **DevOps**: Docker Compose, Makefile, GitHub Actions CI

---

## 🚀 Getting Started

### 0. Configure Your Environment (Required)

MARRA requires a Google Gemini API key for embeddings and response synthesis.

```bash
# 1. Copy the example environment file
cp .env.example .env

# 2. Open .env and replace the placeholder with your real key
# Get a key at: https://aistudio.google.com/app/apikey
# GEMINI_API_KEY=your_actual_key_here
```

> ⚠️ **Never commit your `.env` file.** It is already listed in `.gitignore`.

---

### 1. Prerequisites & Host Configurations

Before starting the containers, ensure you have **Docker** and **Ollama** installed on your host system.

#### ⚠️ CRITICAL STEP: Bind Ollama to `0.0.0.0`
By default, Ollama binds to `127.0.0.1`, which prevents the API container from routing queries to it. You must configure Ollama to listen on all interfaces (`0.0.0.0`) so the container network can reach it:

- **Windows**:
  1. Quit Ollama from the Windows Taskbar tray.
  2. Open system env variables config: Start -> search "Environment Variables" -> click "Edit the system environment variables".
  3. Under "User variables", click **New...**, set Variable name to `OLLAMA_HOST` and Variable value to `0.0.0.0`.
  4. Save changes and restart Ollama from the Start Menu.
  *(Alternatively, run `$env:OLLAMA_HOST="0.0.0.0"` followed by `ollama serve` in PowerShell)*.

- **macOS**:
  1. Quit the Ollama Desktop app.
  2. Open Terminal and start Ollama manually with the host environment variable:
     ```bash
     OLLAMA_HOST=0.0.0.0 ollama serve
     ```

- **Linux**:
  1. Edit the systemd service file:
     ```bash
     sudo systemctl edit ollama.service
     ```
  2. Add the environment variable under the `[Service]` section:
     ```ini
     [Service]
     Environment="OLLAMA_HOST=0.0.0.0"
     ```
  3. Reload systemd and restart the service:
     ```bash
     sudo systemctl daemon-reload
     sudo systemctl restart ollama
     ```

#### Pull Required Models
Make sure you have downloaded the necessary Ollama model to your host:
```bash
# The planner LLM — runs entirely locally
ollama pull llama3.1:8b
```

> **Note:** Embeddings are generated via Google Gemini's `gemini-embedding-2` API (cloud). No Ollama embedding model is required.

---

### 2. Quickstart with Makefile

We leverage a `Makefile` to simplify deployment orchestration. Run these commands from the project root:

#### Spin Up Services
Build containers and start them in detached mode:
```bash
make up
```
This spins up four services:
- **Streamlit UI** on [http://localhost:8501](http://localhost:8501)
- **FastAPI API Gateway** on [http://localhost:8000](http://localhost:8000)
- **Qdrant Vector DB** dashboard on [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
- **Arize Phoenix Observability** dashboard on [http://localhost:6006](http://localhost:6006)

#### Tail Service Logs
Observe output across all running containers:
```bash
make logs
```

#### Shut Down Services
Stop and tear down the running containers:
```bash
make down
```

---

### 3. Ingestion Pipeline

To populate the vector database, you can run the local ingestion script to chunk, embed, and index documents.

If you want to run it on your host machine, install requirements locally:
```bash
pip install -r requirements.txt
```

Then run the ingestion script with your target file:
```bash
python scripts/ingest_file.py data/sample.txt
```
This will chunk the text, query Ollama for embeddings, insert dense vectors into Qdrant, and compile/serialize the BM25 sparse index into `data/bm25_index.pkl` (which is shared with the API container via Docker volume mounts).

---

## 🔍 Observability and Tracing

MARRA has full observability built directly into the LangGraph nodes and FastAPI lifespan events.

1. Once the containers are running, navigate to the Arize Phoenix UI: **[http://localhost:6006](http://localhost:6006)**.
2. In the Streamlit UI, send a user query (e.g. "What is MARRA?").
3. Refresh the Phoenix dashboard. You will see a trace tree for the POST request to `/chat`:
   - Expand the tree to view `planner_node`, `retriever_node`, and `synthesizer_node` execution times.
   - Inspect individual LLM generation inputs/outputs and database queries.

---

## 🧪 Testing

The codebase relies on real integration tests that run against the local Qdrant/Ollama endpoints. Ensure your services are running before executing tests.

To run the suite locally:
```bash
pytest tests/
```
