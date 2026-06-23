import pytest
from unittest.mock import patch, MagicMock
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_search import hybrid_search
from src.models.schema import DocumentChunk
from rank_bm25 import BM25Okapi

@pytest.fixture
def memory_qdrant():
    """Provides a patched VectorStore configured with an in-memory Qdrant client."""
    with patch("src.retrieval.vector_store.QdrantClient") as mock_qdrant:
        client = QdrantClient(location=":memory:")
        mock_qdrant.return_value = client
        yield client

def test_vector_store_real_memory_filtering(memory_qdrant):
    # Initialize VectorStore. It will use the in-memory client.
    store = VectorStore(collection_name="test_collection")
    
    # Setup test chunks with dense vectors (length 768)
    vector_dim = 768
    chunks = [
        DocumentChunk(
            text="This is a text about coding",
            metadata={"media_type": "text", "file_name": "doc1.txt"},
            dense_vector=[0.1] * vector_dim
        ),
        DocumentChunk(
            text="This is a video showing gameplay",
            metadata={"media_type": "video", "file_name": "game.mp4"},
            dense_vector=[0.2] * vector_dim
        ),
        DocumentChunk(
            text="This is an audio recording of wind",
            metadata={"media_type": "audio", "file_name": "wind.mp3"},
            dense_vector=[0.3] * vector_dim
        ),
        DocumentChunk(
            text="This is an image of a cat",
            metadata={"media_type": "image", "file_name": "cat.png"},
            dense_vector=[0.4] * vector_dim
        ),
        # Edge Case: Missing metadata field "media_type"
        DocumentChunk(
            text="This chunk is missing media_type entirely",
            metadata={"file_name": "unknown.txt"},
            dense_vector=[0.5] * vector_dim
        ),
        # Edge Case: Metadata is empty dictionary
        DocumentChunk(
            text="This chunk has empty metadata dict",
            metadata={},
            dense_vector=[0.6] * vector_dim
        ),
        # Edge Case: media_type is invalid type (integer)
        DocumentChunk(
            text="This chunk has numeric media_type",
            metadata={"media_type": 12345},
            dense_vector=[0.7] * vector_dim
        ),
    ]
    
    # Index them into our in-memory Qdrant
    store.upsert_chunks(chunks)
    
    # Query vector
    query_vector = [0.15] * vector_dim
    
    # 1. Filter: "video" (should return exactly 1 video chunk)
    res_video = store.dense_search(query_vector, k=10, media_filter="video")
    assert len(res_video) == 1
    assert res_video[0].payload["metadata"]["media_type"] == "video"
    assert "gameplay" in res_video[0].payload["text"]
    
    # 2. Filter: "audio" (should return exactly 1 audio chunk)
    res_audio = store.dense_search(query_vector, k=10, media_filter="audio")
    assert len(res_audio) == 1
    assert res_audio[0].payload["metadata"]["media_type"] == "audio"
    
    # 3. Filter: "text" (should return exactly 1 text chunk)
    res_text = store.dense_search(query_vector, k=10, media_filter="text")
    assert len(res_text) == 1
    assert res_text[0].payload["metadata"]["media_type"] == "text"
    
    # 4. Filter: "image" (should return exactly 1 image chunk)
    res_image = store.dense_search(query_vector, k=10, media_filter="image")
    assert len(res_image) == 1
    assert res_image[0].payload["metadata"]["media_type"] == "image"
    
    # 5. Filter: "all" (should return all 7 chunks)
    res_all = store.dense_search(query_vector, k=10, media_filter="all")
    assert len(res_all) == 7
    
    # 6. Filter: None or empty (should act like "all")
    res_none = store.dense_search(query_vector, k=10, media_filter=None)
    assert len(res_none) == 7
    res_empty = store.dense_search(query_vector, k=10, media_filter="")
    assert len(res_empty) == 7
    
    # 7. Filter: Non-existent media type (should return empty list)
    res_unknown = store.dense_search(query_vector, k=10, media_filter="pdf")
    assert len(res_unknown) == 0

def test_bm25_retriever_edge_cases():
    retriever = BM25Retriever()
    
    # Setup test chunks with enough filler documents to avoid negative BM25 IDF scores
    chunks = [
        DocumentChunk(text="The quick brown fox", metadata={"media_type": "text"}),
        DocumentChunk(text="A plain text document about cats", metadata={"media_type": "text"}),
        DocumentChunk(text="Another text file about dogs", metadata={"media_type": "text"}),
        
        DocumentChunk(text="A video tutorial on cooking pizza", metadata={"media_type": "video"}),
        DocumentChunk(text="A video showing a car race", metadata={"media_type": "video"}),
        DocumentChunk(text="A video tutorial on coding python", metadata={"media_type": "video"}),
        
        DocumentChunk(text="An audio recording of rain falling", metadata={"media_type": "audio"}),
        DocumentChunk(text="An audio recording of wind chimes", metadata={"media_type": "audio"}),
        DocumentChunk(text="An audio recording of thunder", metadata={"media_type": "audio"}),
        
        # Edge Case: Missing media_type key
        DocumentChunk(text="A text file without media type key", metadata={"source": "doc.txt"}),
        # Edge Case: Empty metadata dict
        DocumentChunk(text="Empty metadata document", metadata={}),
    ]
    
    tokenized_corpus = [c.text.lower().split(" ") for c in chunks]
    bm25_model = BM25Okapi(tokenized_corpus)
    
    retriever.bm25_model = bm25_model
    retriever.chunks = chunks
    
    # 1. Filter: "video" (should return 1 chunk matching "pizza")
    res_video = retriever.search("pizza", k=5, media_filter="video")
    assert len(res_video) == 1
    assert res_video[0][0].metadata.get("media_type") == "video"
    
    # 2. Filter: "text" (should return 1 chunk matching "fox")
    res_text = retriever.search("fox", k=5, media_filter="text")
    assert len(res_text) == 1
    assert res_text[0][0].metadata.get("media_type") == "text"
    
    # 3. Filter: Nonexistent type (should return empty list, not crash)
    res_unknown = retriever.search("pizza", k=5, media_filter="pdf")
    assert len(res_unknown) == 0
    
    # 4. Filter: "all" (should search across all chunks)
    res_all = retriever.search("pizza", k=5, media_filter="all")
    assert len(res_all) == 1
    
    # 5. Empty Index Edge Case (should return empty list, not crash)
    retriever.bm25_model = None
    retriever.chunks = []
    res_empty_index = retriever.search("pizza", k=5, media_filter="video")
    assert res_empty_index == []

@patch("src.retrieval.hybrid_search.embed_chunk")
@patch("src.retrieval.hybrid_search.VectorStore")
@patch("src.retrieval.hybrid_search.BM25Retriever")
def test_hybrid_search_integration_edge_cases(mock_bm25_retriever_class, mock_vector_store_class, mock_embed_chunk):
    # Mock embedding to return a vector
    mock_embed_chunk.return_value = [0.1] * 768
    
    # Mock Vector Store dense search
    mock_vector_store = MagicMock()
    mock_vector_store_class.return_value = mock_vector_store
    
    class MockScoredPoint:
        def __init__(self, text, media_type, score=0.8):
            self.payload = {"text": text, "metadata": {"media_type": media_type}}
            self.score = score
            
    # Mock BM25 Retriever
    mock_bm25_retriever = MagicMock()
    mock_bm25_retriever_class.return_value = mock_bm25_retriever
    
    # 1. Edge Case: Both systems return nothing
    mock_vector_store.dense_search.return_value = []
    mock_bm25_retriever.search.return_value = []
    
    res = hybrid_search("test query", k=3, media_filter="video")
    assert res == []
    
    # Reset mock since we ran a text query search
    mock_bm25_retriever.search.reset_mock()
    
    # 2. Edge Case: Multimodal query (skip BM25)
    # If query is image/video/audio path or matches multimedia format
    mock_vector_store.dense_search.return_value = [MockScoredPoint("video response", "video")]
    
    res_mm = hybrid_search("test query", query_media_path="some/file.png", k=3, media_filter="video")
    assert len(res_mm) == 1
    assert res_mm[0]["text"] == "video response"
    # Verify BM25 was never called for multimodal queries
    mock_bm25_retriever.search.assert_not_called()
    
    # Reset mocks
    mock_bm25_retriever.search.reset_mock()
    
    # 3. Filtering validation when media_filter is "all" vs specific
    # In "all", sparse_results should be filtered to keep only text chunks.
    # Let's verify hybrid_search logic filters sparse results.
    mock_vector_store.dense_search.return_value = [
        MockScoredPoint("dense audio chunk", "audio", 0.9),
        MockScoredPoint("dense text chunk", "text", 0.8)
    ]
    
    sparse_text_chunk = DocumentChunk(text="sparse text chunk", metadata={"media_type": "text"})
    sparse_video_chunk = DocumentChunk(text="sparse video chunk", metadata={"media_type": "video"})
    mock_bm25_retriever.search.return_value = [
        (sparse_text_chunk, 0.95),
        (sparse_video_chunk, 0.75)
    ]
    
    # Search with media_filter = "all"
    res_all = hybrid_search("some text", k=5, media_filter="all")
    # Verify that sparse video chunk is filtered out from sparse results and does not appear in final output
    retrieved_texts = [r["text"] for r in res_all]
    assert "sparse video chunk" not in retrieved_texts
    assert "sparse text chunk" in retrieved_texts
    assert "dense audio chunk" in retrieved_texts
    assert "dense text chunk" in retrieved_texts
