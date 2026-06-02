from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.ingestion.embedder import embed_texts

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
        chunk_id = text # Use text as unique ID for fusion logic
        
        chunk_map[chunk_id] = {
            "text": text,
            "metadata": point.payload,
            "dense_score": point.score,
            "sparse_score": 0.0
        }
        rrf_scores[chunk_id] = 1.0 / (k_constant + rank + 1)
        
    # Process sparse results
    for rank, (chunk, score) in enumerate(sparse_results):
        chunk_id = chunk.text
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

def hybrid_search(query: str, k: int = 3):
    # 1. Embed query
    query_vector = embed_texts([query])[0]
    
    # 2. Dense search
    vector_store = VectorStore(collection_name="marra_documents")
    dense_results = vector_store.dense_search(query_vector, k=k)
    
    # 3. Sparse search
    bm25_retriever = BM25Retriever()
    sparse_results = bm25_retriever.search(query, k=k)
    
    # 4. Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results, k_constant=60)
    
    return fused_results[:k]
