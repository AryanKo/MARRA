import pickle
import os
from rank_bm25 import BM25Okapi
from src.models.schema import DocumentChunk

class BM25Indexer:
    def __init__(self, index_path: str = "data/bm25_index.pkl"):
        self.index_path = index_path

    def build_and_save_index(self, document_chunks: list[DocumentChunk]):
        """
        Tokenizes the text chunks, builds the BM25 index, and serializes it to disk
        along with the original chunk references.
        """
        # Simple whitespace tokenizer for demonstration
        tokenized_corpus = [chunk.text.lower().split(" ") for chunk in document_chunks]
        
        bm25 = BM25Okapi(tokenized_corpus)
        
        data_to_save = {
            "bm25_model": bm25,
            "chunks": document_chunks
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        with open(self.index_path, "wb") as f:
            pickle.dump(data_to_save, f)
        
        print(f"BM25 index saved to {self.index_path}")
