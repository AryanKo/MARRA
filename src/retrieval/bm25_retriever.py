import pickle
import os
import logging
from src.utils.observability import trace_span as span

class BM25Retriever:
    _instance = None
    
    def __new__(cls, index_path: str = "data/bm25_index.pkl"):
        if cls._instance is None:
            cls._instance = super(BM25Retriever, cls).__new__(cls)
            cls._instance.index_path = index_path
            cls._instance.bm25_model = None
            cls._instance.chunks = []
            cls._instance._load_index()
        return cls._instance

    def _load_index(self):
        if not os.path.exists(self.index_path):
            logging.warning(f"BM25 index not found at {self.index_path}. Sparse search will return empty.")
            return
        
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
                self.bm25_model = data.get("bm25_model")
                self.chunks = data.get("chunks", [])
        except Exception as e:
            logging.error(f"Error loading BM25 index: {e}")
            
    @span(name="bm25_sparse_search")
    def search(self, query: str, k: int = 3):
        if not self.bm25_model or not self.chunks:
            return []
            
        tokenized_query = query.lower().split(" ")
        scores = self.bm25_model.get_scores(tokenized_query)
        
        # Get top k indices
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for i in top_k_indices:
            score = scores[i]
            if score > 0:
                results.append((self.chunks[i], score))
        return results
