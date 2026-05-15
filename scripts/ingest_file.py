import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.chunker import load_and_chunk_file
from src.ingestion.embedder import embed_texts
from src.retrieval.vector_store import VectorStore
from src.models.schema import DocumentChunk

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_file.py <file_path>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        sys.exit(1)
    
    print(f"Reading and chunking {file_path}...")
    texts = load_and_chunk_file(file_path, chunk_size=500, chunk_overlap=50)
    print(f"Generated {len(texts)} chunks.")
    
    print("Embedding chunks using nomic-embed-text...")
    vectors = embed_texts(texts)
    
    print("Preparing DocumentChunk objects...")
    document_chunks = []
    for i, (text, vector) in enumerate(zip(texts, vectors)):
        document_chunks.append(
            DocumentChunk(
                text=text,
                dense_vector=vector,
                metadata={"source": file_path, "chunk_index": i}
            )
        )
        
    print("Upserting to Qdrant...")
    try:
        store = VectorStore(collection_name="marra_documents")
        store.upsert_chunks(document_chunks)
        print("Ingestion complete! Successfully added points to Qdrant.")
    except Exception as e:
        import logging
        logging.error(f"Failed to upsert to Qdrant: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default argument for test run
        sys.argv.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample.txt"))
    main()
