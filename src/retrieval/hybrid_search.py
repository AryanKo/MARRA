from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.gemini_embedder import embed_chunk
from src.models.schema import DocumentChunk

def _derive_chunk_id(text, metadata):
    """
    Derives a unique chunk identity key.
    For media chunks, uses file_path so distinct media files are never
    collapsed even if their placeholder text is identical.
    For text chunks, uses the text content itself for natural deduplication.
    """
    file_path = metadata.get("file_path") if metadata else None
    if file_path:
        return file_path
    return text

def reciprocal_rank_fusion(dense_results, sparse_results, k_constant=60):
    """
    Fuses dense and sparse results using RRF.
    Dense results: list of Qdrant ScoredPoint objects.
    Sparse results: list of tuples (DocumentChunk, score).
    """
    rrf_scores = {}
    chunk_map = {}
    
    # Process dense results
    for rank, point in enumerate(dense_results):
        text = point.payload.get("text")
        chunk_id = _derive_chunk_id(text, point.payload)
        
        chunk_map[chunk_id] = {
            "text": text,
            "metadata": point.payload,
            "dense_score": point.score,
            "sparse_score": 0.0
        }
        rrf_scores[chunk_id] = 1.0 / (k_constant + rank + 1)
        
    # Process sparse results
    for rank, (chunk, score) in enumerate(sparse_results):
        chunk_id = _derive_chunk_id(chunk.text, chunk.metadata)
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = {
                "text": chunk.text,
                "metadata": chunk.metadata,
                "dense_score": 0.0,
                "sparse_score": score
            }
            rrf_scores[chunk_id] = 1.0 / (k_constant + rank + 1)
        else:
            chunk_map[chunk_id]["sparse_score"] = score
            rrf_scores[chunk_id] += 1.0 / (k_constant + rank + 1)
            
    # Sort by RRF score
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    fused_results = []
    for chunk_id, rrf_score in sorted_chunks:
        data = chunk_map[chunk_id]
        data["rrf_score"] = rrf_score
        fused_results.append(data)
        
    return fused_results

def hybrid_search(query: str, query_media_path: str = None, k: int = 3):
    is_multimodal_query = query_media_path is not None or query.strip() in ["[IMAGE MEDIA PAYLOAD]", "[AUDIO MEDIA PAYLOAD]", "[VIDEO MEDIA PAYLOAD]"]
    
    # 1. Embed query
    if query_media_path:
        chunk = DocumentChunk(text="[IMAGE MEDIA PAYLOAD]", metadata={"file_path": query_media_path, "media_type": "image"})
        query_vector = embed_chunk(chunk)
    else:
        query_vector = embed_chunk(DocumentChunk(text=query))
        
    # 2. Dense search
    vector_store = VectorStore(collection_name="marra_multimodal_768")
    dense_results = vector_store.dense_search(query_vector, k=k)
    
    # 3. Sparse search
    if is_multimodal_query:
        sparse_results = []
    else:
        bm25_retriever = BM25Retriever()
        sparse_results = bm25_retriever.search(query, k=k)
        # Filter out media chunks from sparse results
        sparse_results = [(c, s) for c, s in sparse_results if c.metadata.get("media_type", "text") == "text"]
        
    # 4. Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results, k_constant=60)
    
    return fused_results[:k]
