import logging
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import httpx
from qdrant_client.http.exceptions import ResponseHandlingException
import anyio

# Import the compiled LangGraph build function
from src.agents.graph import build_graph

from contextlib import asynccontextmanager
from src.utils.observability import setup_observability

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_observability(app)
    yield

app = FastAPI(title="MARRA Reasoning Engine API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = Field(default_factory=list)

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

# Compile the graph once on startup
graph = build_graph()

@app.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(request: ChatRequest):
    try:
        inputs = {
            "query": request.query,
            "history": request.history,
            "sub_queries": [],
            "retrieved_docs": [],
            "final_answer": ""
        }
        
        # Run graph.invoke in a separate thread to keep the FastAPI event loop responsive
        state = await anyio.to_thread.run_sync(graph.invoke, inputs)
        
        # Format sources
        sources = []
        retrieved_docs = state.get("retrieved_docs", [])
        for doc in retrieved_docs:
            sources.append({
                "text": doc.text,
                "metadata": doc.metadata
            })
            
        return ChatResponse(
            answer=state.get("final_answer", ""),
            sources=sources
        )
        
    except (ConnectionError, httpx.ConnectError, ResponseHandlingException) as e:
        logger.error(f"Service connection failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service Unavailable: Failed to communicate with underlying local models or database."
        )
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
