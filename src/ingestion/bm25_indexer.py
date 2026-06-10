import pickle
import os
import tempfile
import logging
from rank_bm25 import BM25Okapi
from src.models.schema import DocumentChunk

logger = logging.getLogger(__name__)

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
                    logger.info(f"Loaded {len(existing_chunks)} existing chunks from {self.index_path}")
            except Exception as e:
                logger.error(f"Error loading existing BM25 index from {self.index_path}: {e}")
                
        combined_chunks = existing_chunks + document_chunks

        if len(combined_chunks) == 0:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with open(self.index_path, "wb") as f:
                pickle.dump({"bm25_model": None, "chunks": []}, f)
            logger.info(f"BM25 index saved to {self.index_path} with 0 total chunks (Empty)")
            return

        # Simple whitespace tokenizer for demonstration
        tokenized_corpus = [chunk.text.lower().split(" ") for chunk in combined_chunks]
        
        bm25 = BM25Okapi(tokenized_corpus)
        
        data_to_save = {
            "bm25_model": bm25,
            "chunks": combined_chunks
        }
        
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Atomic write: write to a temp file, then os.replace() to the target path
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=os.path.dirname(self.index_path), delete=False, suffix=".tmp") as tmp_f:
                tmp_path = tmp_f.name
                pickle.dump(data_to_save, tmp_f)
            os.replace(tmp_path, self.index_path)
        except Exception:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        
        logger.info(f"BM25 index saved to {self.index_path} with {len(combined_chunks)} total chunks")
