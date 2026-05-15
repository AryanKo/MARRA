import pytest
from pydantic import ValidationError
from src.models.schema import DocumentChunk, AgentState, AppError

def test_document_chunk_valid():
    chunk = DocumentChunk(
        text="Sample text",
        metadata={"source": "test.txt"},
        dense_vector=[0.1, 0.2, 0.3]
    )
    assert chunk.text == "Sample text"
    assert chunk.metadata == {"source": "test.txt"}
    assert chunk.dense_vector == [0.1, 0.2, 0.3]

def test_document_chunk_defaults():
    chunk = DocumentChunk(text="Sample text")
    assert chunk.text == "Sample text"
    assert chunk.metadata == {}
    assert chunk.dense_vector is None

def test_document_chunk_invalid():
    with pytest.raises(ValidationError):
        DocumentChunk(metadata={"source": "test.txt"})  # Missing required 'text'

def test_agent_state():
    state: AgentState = {
        "query": "What is AI?",
        "sub_queries": ["definition of AI"],
        "retrieved_docs": [DocumentChunk(text="AI is artificial intelligence.")],
        "final_answer": "AI stands for artificial intelligence."
    }
    assert state["query"] == "What is AI?"
    assert len(state["sub_queries"]) == 1
    assert len(state["retrieved_docs"]) == 1
    assert state["final_answer"] == "AI stands for artificial intelligence."

def test_app_error():
    error = AppError("Something went wrong")
    assert str(error) == "Something went wrong"
    assert isinstance(error, Exception)
