import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    text: str = Field(..., description="The text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the chunk")
    dense_vector: Optional[List[float]] = Field(default=None, description="Optional dense vector embedding")

class AgentState(TypedDict):
    query: str
    history: List[Dict[str, str]]
    sub_queries: Annotated[List[str], operator.add]
    retrieved_docs: Annotated[List[DocumentChunk], operator.add]
    final_answer: str
    media_filter: str

class AppError(Exception):
    """Custom exception class for application-specific errors."""
    pass
