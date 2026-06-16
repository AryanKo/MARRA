# 🧠 MARRA: Hybrid Multi-Agent RAG Platform

![Version](https://img.shields.io/badge/version-v1.1.0-blue) ![Python](https://img.shields.io/badge/python-3.11-green) ![License](https://img.shields.io/badge/license-MIT-green)

MARRA (Multi-Agent Retrieval Reasoning Assistant) is a production-grade, local-first RAG platform designed to deliver secure, private, and deterministic document intelligence. It features a decoupled microservices architecture, a deterministic agent state machine orchestrated with LangGraph, hybrid search (dense vector + sparse BM25) utilizing Reciprocal Rank Fusion (RRF), and comprehensive OTLP-compliant observability.

---

## 🏗️ Architecture Overview

The system is split into decoupled layers to ensure strict separation of concerns, scalability, and optimal hardware utilization:

1. **Streamlit UI Layer**: A clean, modern interface presenting interactive agent workflows, multimodal ingestion controls (in the sidebar), and grounded source visualizations.
2. **FastAPI API Gateway**: Decoupled backend serving agent invocation and ingestion endpoints over thread-safe pipelines.
3. **LangGraph State Machine**: Orchestrates search planning, retrieval, and response synthesis inside a deterministic, stateful graph.
4. **Hybrid Retrieval Engine**:
   - **Dense Search**: Semantic vector retrieval matching queries against embeddings stored in Qdrant.
   - **Sparse Search**: Keyword-based search matching queries using a BM25 index.
   - **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse retrieval ranks dynamically.
5. **Observability & Tracing (Arize Phoenix)**: OTLP tracing collector capturing span-level latency, execution logs, and detailed sub-query breakdowns.
6. **Ollama LLM Engine**: Runs locally to host the local planner node (`llama3.1:8b`) with GPU/CPU acceleration.
7. **Google Gemini Cloud**: Handles cloud-native Matryoshka embeddings (`gemini-embedding-2`) and multimodal response synthesis (`gemini-3.5-flash`), processing text context, multi-turn history, and media (images, audio, video). The synthesizer node dynamically routes media assets, utilizing direct byte injection for small files ($\le$ 4MB) and the Google GenAI File API for larger assets (featuring automated bounded polling and guaranteed resource cleanup).

### System Topology (ASCII)

```
+-----------------------------------------------------------------------------------+
|                                 HOST ENVIRONMENT                                  |
|                                                                                   |
|  +---------------------------+             +-----------------------------------+  |
|  |     Streamlit Container   |             |           Ollama Service          |  |
|  |                           |             |        (Metal/CUDA Native)        |  |
|  |     +---------------+     |             |        +-----------------+        |  |
|  |     |  app.py (UI)  |     |             |        |  llama3.1:8b    |        |  |
|  |     +-------+-------+     |             |        +--------^--------+        |  |
|  |             |             |             +-----------------|-----------------+  |
|  |             |             |                               |                    |
|  +-------------|-------------+                               |                    |
|     (Browser)  | API Request                                 |                    |
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
|  |    |  State Machine  |    |   Local LLM Query (Planning)                       |
|  |    +--------+--------+    |                                                    |
|  |             |             |                                                    |
|  |             |             |             +-----------------------------------+  |
|  |             |             |             |        Google Gemini Cloud        |  |
|  |             |             |             |                                   |  |
|  |             |             |             |     - gemini-embedding-2          |  |
|  |             +-------------+------------>|     - gemini-3.5-flash            |  |
|  |             | Cloud Multimodal Synthesis|                                   |  |
|  |             v                           +-----------------------------------+  |
|  +-------------|------------------------------------------------------------------+
|                |                      
|                | Port 6333            Port 4318 (OTLP HTTP)
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
- **Sparse Search**: Rank-BM25 (Keyword-based token score matching, persisted locally)
- **Local LLM**: Ollama (`llama3.1:8b` — query planning, runs on host GPU/CPU)
- **Cloud AI**: Google Gemini (`gemini-embedding-2` for 768-dim Matryoshka embeddings, `gemini-3.5-flash` for multimodal synthesis)
- **Observability**: Arize Phoenix + OpenTelemetry (OTLP exporters)
- **DevOps**: Docker Compose, Windows run.bat script, GitHub Actions CI

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

Before starting the containers, ensure you have **Docker** and **Ollama** installed on your host system. If running services directly on the host (outside Docker Compose), **FFmpeg** and **FFprobe** must also be installed and added to your system's PATH (these are required for chunking and analyzing incoming audio/video files).

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

### 2. Quickstart with run.bat

We leverage a Windows-native batch macro script `run.bat` to manage the lifecycle of the system. Simply run `run.bat` in your Windows terminal to build, launch, and stream logs simultaneously:

```cmd
run.bat
```

This automated script performs the following operations:
1. Shuts down any pre-existing containers to avoid port collisions.
2. Builds the container images cleanly.
3. Launches the services in detached mode.
4. Automatically streams container logs directly to your terminal.

Once running, the following services are available:
- **Streamlit UI** on [http://localhost:8501](http://localhost:8501)
- **FastAPI API Gateway** on [http://localhost:8000](http://localhost:8000)
- **Qdrant Vector DB** dashboard on [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
- **Arize Phoenix Observability** dashboard on [http://localhost:6006](http://localhost:6006)

To stop the services, run:
```bash
docker compose down
```

---

### 3. Ingestion Pipeline

Ingestion is fully integrated into the Streamlit user interface sidebar. You do not need to run separate command-line scripts to ingest content.

#### How to Ingest Documents:
1. Open the Streamlit UI at [http://localhost:8501](http://localhost:8501).
2. Expand the sidebar panel.
3. Upload files using the secure drag-and-drop area. The ingestion pipeline natively supports:
   - **Text & Markdown documents**: `.txt`, `.md`
   - **Multimodal files**: Images (`.png`, `.jpg`, `.jpeg`, `.webp`), audio (`.mp3`, `.wav`), and video (`.mp4`, `.mov`, `.mpeg`)
4. Choose the ingestion mode:
   - **Append**: Adds the new document vectors and indices to the existing database.
   - **Overwrite**: Purges existing vector collections and local sparse indices before indexing the new files.
5. Click **🚀 Ingest Document** to start the pipeline. The file will be processed, chunked, embedded via `gemini-embedding-2`, and indexed into both Qdrant (dense vectors) and the local BM25 engine.

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

---

## 🆕 Version 1.1.0 Changelog (Omnimodal Upgrade & Hardening)

### 🎙️ Omnimodal Ingestion & Processing
- **Format Extension**: Native support for video files (`.mp4`, `.mov`, `.mpeg`) and audio files (`.mp3`, `.wav`).
- **Secure File Ingestion**: Injected 8-character UUIDs into persistent filenames to prevent file name collisions and accidental media overwrites.
- **Dynamic Media Routing**: Optimized synthesizer node to route media assets dynamically:
  - **Direct Byte Injection**: Files $\le$ 4MB (images) are sent inline via `types.Part.from_bytes` for speed.
  - **Cloud File API**: Files > 4MB (and all audio/video segments) utilize the Google GenAI File API (`client.files.upload`) with bounded polling.

### 🛡️ Production & SRE Hardening
- **Bounded Polling (TV-1)**: Eliminated polling deadlocks by wrapping the File API state check in a bounded loop with a 120-second timeout ceiling, detecting terminal `FAILED` status, and preventing thread exhaustion.
- **FFmpeg Zombie Cleanup (TV-2)**: Encapsulated FFmpeg segmentation in custom `Popen` blocks using process-group isolation (`CREATE_NEW_PROCESS_GROUP` on Windows and `start_new_session=True` on Linux) with a 300s timeout. If timed out, the entire process group is forcefully reaped.
- **Zero-Leak Remote Storage (TV-3)**: Implemented a strict `try...finally` resource manager that guarantees the execution of `client.files.delete()` for all uploaded Google File API objects, ensuring customer data privacy and preventing API quota exhaustion.
- **Strict Token Budgeting (TV-4)**: Overhauled token estimation logic using `ffprobe` with a 10s timeout, falling back on conservative high-end byte-size heuristics (~18,000 tokens/MB for video and ~3,840 tokens/MB for audio) to safely skip files exceeding context thresholds instead of triggering `400 INVALID_ARGUMENT` crashes.
- **Album Art Muxing Crash Fix**: Enforced `-vn -map 0:a` parameters in FFmpeg to strip image metadata (e.g., album art) from audio files, bypassing a known muxing crash.
- **Timeout Adaptations**: Streamlit timeout threshold extended to 300 seconds for ingestion and 120 seconds for chat, permitting handling of large media.
