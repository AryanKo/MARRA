import logging
import os
import shutil
from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
import httpx
from qdrant_client.http.exceptions import ResponseHandlingException
import anyio
import asyncio

from src.ingestion.chunker import load_and_chunk_file
from src.retrieval.gemini_embedder import embed_chunk
from src.ingestion.multimodal_loader import chunk_multimodal_file, cleanup_multimodal_chunks
from src.ingestion.bm25_indexer import BM25Indexer
from src.retrieval.vector_store import VectorStore
from src.models.schema import DocumentChunk
from src.retrieval.bm25_retriever import BM25Retriever


# Import the compiled LangGraph build function
from src.agents.graph import build_graph

from contextlib import asynccontextmanager
from src.utils.observability import setup_observability

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module-level lock for serializing BM25 index writes
_bm25_write_lock = asyncio.Lock()

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

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query must not be empty or whitespace-only.")
        return v.strip()

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

@app.post("/ingest", status_code=status.HTTP_200_OK)
async def ingest_endpoint(file: UploadFile = File(...), overwrite: bool = Form(False)):
    try:
        filename = file.filename
        logger.info(f"Received ingestion request for file: {filename}, overwrite={overwrite}")
        
        # 1. Handle overwrite
        if overwrite:
            logger.info("Overwrite is True. Clearing Vector Collection and deleting BM25 index.")
            # Wipe Qdrant
            store = VectorStore(collection_name="marra_multimodal_768")
            store.clear_collection()
            
            # Delete BM25 index pkl
            bm25_path = "data/bm25_index.pkl"
            if os.path.exists(bm25_path):
                try:
                    os.remove(bm25_path)
                    logger.info(f"Successfully deleted BM25 index at {bm25_path}")
                except Exception as e:
                    logger.error(f"Failed to delete BM25 index file: {e}")
        
        # 2. Save file temporarily
        import uuid
        temp_dir = "/tmp"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}_{filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Saved uploaded file to {file_path}")
        
        # 3. Handle chunking and embedding based on file type
        document_chunks = []
        ext = os.path.splitext(filename)[1].lower()
        is_media = ext in [".jpg", ".jpeg", ".png", ".webp", ".mp3", ".wav", ".mp4"]
        
        try:
            if is_media:
                logger.info(f"Processing multimodal file: {file_path}")
                document_chunks = chunk_multimodal_file(file_path)
                logger.info(f"Generated {len(document_chunks)} media chunks. Generating embeddings...")
                for chunk in document_chunks:
                    chunk.dense_vector = await anyio.to_thread.run_sync(embed_chunk, chunk)
            else:
                logger.info(f"Chunking text file: {file_path}")
                texts = load_and_chunk_file(file_path, chunk_size=500, chunk_overlap=50)
                if not texts:
                    return {"status": "success", "message": f"File {filename} was empty. No chunks generated."}
                
                logger.info(f"Generating embeddings for {len(texts)} text chunks...")
                for i, text in enumerate(texts):
                    chunk = DocumentChunk(
                        text=text,
                        metadata={
                            "source": filename,
                            "file_name": filename,
                            "chunk_id": i,
                            "media_type": "text"
                        }
                    )
                    chunk.dense_vector = await anyio.to_thread.run_sync(embed_chunk, chunk)
                    document_chunks.append(chunk)

            new_file_path = None

            if is_media:
                media_dir = "data/media"
                os.makedirs(media_dir, exist_ok=True)
                name, ext = os.path.splitext(filename)
                collision_guard = uuid.uuid4().hex[:8]
                persistent_name = f"{name}_{collision_guard}{ext}"
                new_file_path = os.path.join(media_dir, persistent_name)
                shutil.copy2(file_path, new_file_path)
                logger.info(f"Staged media file to persistent storage: {new_file_path}")
                
                # Update chunk metadata pathing
                for chunk in document_chunks:
                    chunk.metadata["file_path"] = new_file_path

            # 6. Upsert to Qdrant (Atomic Transaction with Rollback)
            try:
                logger.info("Upserting chunks to Qdrant...")
                store = VectorStore(collection_name="marra_multimodal_768")
                await anyio.to_thread.run_sync(store.upsert_chunks, document_chunks)
            except Exception as qdrant_error:
                # ROLLBACK: Delete the staged media file to prevent orphaned disk bloat
                if new_file_path and os.path.exists(new_file_path):
                    try:
                        os.remove(new_file_path)
                        logger.info(f"ROLLBACK: Deleted orphaned media file {new_file_path}")
                    except OSError as cleanup_err:
                        logger.error(f"ROLLBACK FAILED: Could not delete {new_file_path}: {cleanup_err}")
                raise qdrant_error
            
        finally:
            # Cleanup temp files generated by multimodal slicing
            if is_media:
                cleanup_multimodal_chunks(document_chunks)
            # Cleanup the original temp file upload
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Cleaned up temp file {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {file_path}: {e}")
        
        # 7. Build and save BM25 index (only for text chunks)
        logger.info(f"Updating BM25 index (append={not overwrite})...")
        text_chunks = [c for c in document_chunks if c.metadata.get("media_type", "text") == "text"]
        if text_chunks or overwrite:
            async with _bm25_write_lock:
                bm25_indexer = BM25Indexer()
                await anyio.to_thread.run_sync(bm25_indexer.build_and_save_index, text_chunks, not overwrite)
        
        # 8. Reload BM25 Retriever
        logger.info("Reloading BM25 Retriever in-memory index...")
        BM25Retriever().reload()
        
        
        return {
            "status": "success",
            "message": f"Successfully ingested {len(document_chunks)} chunks from {filename}."
        }
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
