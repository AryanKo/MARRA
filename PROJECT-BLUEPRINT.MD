# MARRA: Master Project Blueprint
**Project Name:** Hybrid Multi-Agent RAG Research Assistant
**Version:** 1.0.0
**Target Deployment:** Local-First, 100% Free-Tier / Open-Source

## 1. Executive Summary & Objectives
MARRA is a privacy-preserving, local-first AI research platform. It ingests complex documents, retrieves highly relevant context using a hybrid search pipeline (Semantic + Keyword), and synthesizes answers via an agentic workflow. 
**Primary Goal:** Prove advanced AI engineering capability (Agentic Routing, Hybrid RAG, Local LLM Orchestration) without relying on paid APIs like OpenAI or Pinecone.

## 2. Global Architecture
The system is strictly decoupled into four layers:
1. **Frontend:** Streamlit UI for chat and citation display.
2. **API Gateway:** FastAPI (async) for orchestration and session management.
3. **Agent Workflow:** LangGraph state machine managing Planner, Retriever, and Synthesizer nodes.
4. **Data & AI Layer:** - LLM / Embeddings: Local Ollama (`llama3.1:8b` and `nomic-embed-text`).
   - Vector Store: Local Qdrant Docker container.

## 3. Tech Stack Requirements (Strict)
- **Language:** Python 3.11+
- **Frameworks:** FastAPI, LangGraph, LangChain (strictly for chunking/parsing, NOT for agent logic), Streamlit.
- **Database:** Qdrant (Python Client -> localhost:6333).
- **Models:** Ollama (Python Client -> localhost:11434).
- **Quality Tools:** Pytest, Pydantic (Strict typing everywhere).

## 4. Directory Structure
```text
/
├── agent-execution-rules.md    # Immutable rules for the AI Agent
├── PROJECT_BLUEPRINT.md        # This file
├── docker-compose.yml          # Runs Qdrant
├── data/                       # Raw input PDFs/Markdown
├── scripts/                    # Verification and infra tests
├── src/
│   ├── api/                    # FastAPI routers and server
│   ├── agents/                 # LangGraph nodes, edges, state dicts
│   ├── retrieval/              # Qdrant client, BM25, hybrid fusion logic
│   ├── ingestion/              # Document parsing and chunking
│   ├── models/                 # Pydantic schemas ONLY
│   └── utils/                  # Logging, config
└── tests/                      # Pytest suite

```

## 5. Development Roadmap & Checkpoints

**Phase 0: Infrastructure & Schemas**

* *Goal:* Establish connections to Qdrant and Ollama. Define Pydantic base models.
* *Checkpoint:* `scripts/verify_infra.py` runs and confirms 200 OK from databases and models. `test_schemas.py` passes.

**Phase 1: Ingestion Pipeline**

* *Goal:* Parse files, chunk via recursive splitting, embed via `nomic`, push to Qdrant.
* *Checkpoint:* A script can be fed a `.txt` file and successfully populate Qdrant.

**Phase 2: Hybrid Retrieval Engine**

* *Goal:* Execute dense vector search AND sparse BM25 search. Fuse via Reciprocal Rank Fusion (RRF).
* *Checkpoint:* Given a test query, the engine returns the top 3 most mathematically relevant chunks.

**Phase 3: Agent Orchestration (LangGraph)**

* *Goal:* Build a deterministic graph. Planner Node -> Retriever Node -> Synthesizer Node.
* *Checkpoint:* A standalone Python script runs the graph start-to-finish without errors and prints a coherent answer.

**Phase 4: API & Frontend Integration**

* *Goal:* Wrap the graph in a FastAPI POST endpoint. Connect Streamlit chat UI.
* *Checkpoint:* User can type a message in the browser and get a streamed/processed response back.

## 6. Critical Engineering Constraints

* **State Management:** LangGraph `AgentState` must be immutable and heavily typed.
* **No LangChain Chains:** Use LangChain for utilities (like `RecursiveCharacterTextSplitter`), but write the RAG retrieval logic natively using the `qdrant-client`. Do NOT abstract away the database logic.
* **Fail Gracefully:** If Ollama crashes, the FastAPI server must catch the `ConnectionError` and return a clean HTTP 503, not a raw traceback.

```

***

> "You are an Elite AI Systems Architect acting autonomously. Before we begin, read `PROJECT_BLUEPRINT.md` and `agent-execution-rules.md` in the root directory. Summarize the project goals, the strict constraints, and the verification loop you must follow. Do not write any code until you have confirmed you understand the macro-architecture and the anti-hallucination rules."

Once it summarizes it back to you, you say:

> "Excellent. You are now authorized to begin Phase 0 as defined in the Blueprint. Execute the task, run your verification script, read the terminal output, and let me know when it passes."

By establishing this macro-context, the agent knows *why* it is building a specific Pydantic schema, it knows *where* it belongs in the `src/` directory, and it knows *how* it fits into Phase 4 down the line. It stops being a code-generator and becomes a cohesive systems developer.

```