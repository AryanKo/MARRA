import pytest
from unittest.mock import patch, MagicMock
from rank_bm25 import BM25Okapi
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_search import hybrid_search
from src.models.schema import DocumentChunk

def test_bm25_filtering():
    retriever = BM25Retriever()
    
    # Setup test chunks with enough filler documents to avoid negative BM25 IDF scores
    chunks = [
        DocumentChunk(text="The quick brown fox", metadata={"media_type": "text"}),
        DocumentChunk(text="A plain text document about cats", metadata={"media_type": "text"}),
        DocumentChunk(text="Another text file about dogs", metadata={"media_type": "text"}),
        
        DocumentChunk(text="A video showing a jumping fox", metadata={"media_type": "video"}),
        DocumentChunk(text="A video showing a car race", metadata={"media_type": "video"}),
        DocumentChunk(text="A video tutorial on cooking", metadata={"media_type": "video"}),
        
        DocumentChunk(text="An audio recording of a fox barking", metadata={"media_type": "audio"}),
        DocumentChunk(text="An audio track with music", metadata={"media_type": "audio"}),
        DocumentChunk(text="A podcast about history", metadata={"media_type": "audio"}),
        
        DocumentChunk(text="A photo of a red fox", metadata={"media_type": "image"}),
        DocumentChunk(text="A picture of a sunset", metadata={"media_type": "image"}),
        DocumentChunk(text="A painting of a mountain", metadata={"media_type": "image"}),
    ]
    
    # Tokenize corpus for default model
    tokenized_corpus = [c.text.lower().split(" ") for c in chunks]
    bm25_model = BM25Okapi(tokenized_corpus)
    
    # Inject directly
    retriever.bm25_model = bm25_model
    retriever.chunks = chunks
    
    # 1. Search for "fox" with "all" (should return all 4 matching chunks from BM25)
    results = retriever.search("fox", k=10, media_filter="all")
    assert len(results) == 4
    media_types = {chunk.metadata.get("media_type") for chunk, _ in results}
    assert media_types == {"text", "video", "audio", "image"}
        
    # 2. Search for "fox" with "video" (should only return "video" chunks)
    results = retriever.search("fox", k=10, media_filter="video")
    assert len(results) == 1
    assert results[0][0].metadata.get("media_type") == "video"
    assert "video" in results[0][0].text
    
    # 3. Search for "fox" with "audio" (should only return "audio" chunks)
    results = retriever.search("fox", k=10, media_filter="audio")
    assert len(results) == 1
    assert results[0][0].metadata.get("media_type") == "audio"
    
    # 4. Search for "fox" with "image" (should only return "image" chunks)
    results = retriever.search("fox", k=10, media_filter="image")
    assert len(results) == 1
    assert results[0][0].metadata.get("media_type") == "image"

@patch("src.retrieval.hybrid_search.embed_chunk")
@patch("src.retrieval.hybrid_search.VectorStore")
def test_hybrid_search_filtering(mock_vector_store_cls, mock_embed_chunk):
    # Mock embedding
    mock_embed_chunk.return_value = [0.1] * 768
    
    # Mock Qdrant results
    mock_vector_store = MagicMock()
    mock_vector_store_cls.return_value = mock_vector_store
    
    # We will mock dense_search to return a specific list of ScoredPoints
    class MockScoredPoint:
        def __init__(self, text, media_type, score=0.9):
            self.payload = {"text": text, "media_type": media_type}
            self.score = score
            
    # Mock dynamic dense search return values based on call args
    def mock_dense_search(query_vector, k=3, media_filter="all"):
        if media_filter == "video":
            return [MockScoredPoint("video chunk", "video")]
        elif media_filter == "audio":
            return [MockScoredPoint("audio chunk", "audio")]
        elif media_filter == "text":
            return [MockScoredPoint("text chunk", "text")]
        else:
            return [
                MockScoredPoint("text chunk", "text"),
                MockScoredPoint("video chunk", "video"),
                MockScoredPoint("audio chunk", "audio")
            ]
            
    mock_vector_store.dense_search.side_effect = mock_dense_search

    # Setup BM25 Retriever
    retriever = BM25Retriever()
    chunks = [
        DocumentChunk(text="text chunk", metadata={"media_type": "text"}),
        DocumentChunk(text="video chunk", metadata={"media_type": "video"}),
        DocumentChunk(text="audio chunk", metadata={"media_type": "audio"}),
    ]
    tokenized_corpus = [c.text.lower().split(" ") for c in chunks]
    retriever.bm25_model = BM25Okapi(tokenized_corpus)
    retriever.chunks = chunks

    # 1. Search with media_filter="video"
    results = hybrid_search("chunk", k=10, media_filter="video")
    assert len(results) > 0
    for res in results:
        assert res["metadata"].get("media_type") == "video"

    # 2. Search with media_filter="text"
    results = hybrid_search("chunk", k=10, media_filter="text")
    assert len(results) > 0
    for res in results:
        assert res["metadata"].get("media_type") == "text"

    # 3. Search with media_filter="all" (hybrid search filters out media chunks from sparse results, so dense results can still have media, but sparse is text only)
    results = hybrid_search("chunk", k=10, media_filter="all")
    assert len(results) > 0

