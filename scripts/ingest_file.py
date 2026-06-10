import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.chunker import load_and_chunk_file
from src.retrieval.gemini_embedder import embed_chunk
from src.ingestion.bm25_indexer import BM25Indexer
from src.retrieval.vector_store import VectorStore
from src.models.schema import DocumentChunk
import logging

logging.basicConfig(level=logging.INFO)

COLLECTION_NAME = "marra_multimodal_768"

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_file.py <file_path>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        sys.exit(1)

    filename = os.path.basename(file_path)
    
    print(f"Reading and chunking {file_path}...")
    texts = load_and_chunk_file(file_path, chunk_size=500, chunk_overlap=50)
    print(f"Generated {len(texts)} chunks.")
    
    print("Generating Gemini embeddings...")
    document_chunks = []
    for i, text in enumerate(texts):
        chunk = DocumentChunk(
            text=text,
            metadata={
                "source": filename,
                "file_name": filename,
                "chunk_id": i,
                "media_type": "text"
            }
        )
        chunk.dense_vector = embed_chunk(chunk)
        document_chunks.append(chunk)
        print(f"  Embedded chunk {i + 1}/{len(texts)}")

    print(f"Upserting {len(document_chunks)} chunks to Qdrant collection '{COLLECTION_NAME}'...")
    try:
        store = VectorStore(collection_name=COLLECTION_NAME)
        store.upsert_chunks(document_chunks)
        print("Qdrant upsert complete.")
        
        print("Building and saving BM25 index...")
        bm25_indexer = BM25Indexer()
        bm25_indexer.build_and_save_index(document_chunks, append=True)
        print("Ingestion complete!")
    except Exception as e:
        logging.error(f"Failed to upsert to Qdrant or build BM25: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default argument for test run
        sys.argv.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample.txt"))
    main()
