import pickle
import os
from rank_bm25 import BM25Okapi
from src.models.schema import DocumentChunk

class BM25Indexer:
    def __init__(self, index_path: str = "data/bm25_index.pkl"):
        self.index_path = index_path

    def build_and_save_index(self, document_chunks: list[DocumentChunk], append: bool = False):
        """
        Tokenizes the text chunks, builds the BM25 index, and serializes it to disk
        along with the original chunk references. Supports appending to an existing index.
        """
        existing_chunks = []
        if append and os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    existing_chunks = data.get("chunks", [])
                    print(f"Loaded {len(existing_chunks)} existing chunks from {self.index_path}")
            except Exception as e:
                print(f"Error loading existing BM25 index from {self.index_path}: {e}")
                
        combined_chunks = existing_chunks + document_chunks

        # Simple whitespace tokenizer for demonstration
        tokenized_corpus = [chunk.text.lower().split(" ") for chunk in combined_chunks]
        
        bm25 = BM25Okapi(tokenized_corpus)
        
        data_to_save = {
            "bm25_model": bm25,
            "chunks": combined_chunks
        }
        
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        with open(self.index_path, "wb") as f:
            pickle.dump(data_to_save, f)
        
        print(f"BM25 index saved to {self.index_path} with {len(combined_chunks)} total chunks")
