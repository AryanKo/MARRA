import os
import sys
import asyncio
import base64

# Ensure paths are set
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.multimodal_loader import chunk_multimodal_file, cleanup_multimodal_chunks
from src.retrieval.gemini_embedder import embed_chunk
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import hybrid_search
from dotenv import load_dotenv

load_dotenv()

async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Skipping test: GEMINI_API_KEY environment variable is not set. Please set it to test.")
        return

    # Create a tiny valid PNG file to use as a dummy image
    png_data = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==")
    
    # We must save it to a location that exists on Windows if /tmp doesn't exist.
    # Python's tempfile is safer.
    import tempfile
    temp_dir = tempfile.gettempdir()
    test_image_path = os.path.join(temp_dir, "test_image.png")
    
    with open(test_image_path, "wb") as f:
        f.write(png_data)

    print("--- Starting Multimodal Integration Test ---")
    
    # 2. Ingest
    print(f"1. Loading and chunking media: {test_image_path}")
    chunks = chunk_multimodal_file(test_image_path)
    print(f"   Generated {len(chunks)} chunk(s).")
    
    # 3. Embed
    print("2. Generating Gemini 2.0 Embedding (Matryoshka 768-dim)...")
    try:
        for chunk in chunks:
            chunk.dense_vector = embed_chunk(chunk)
            print(f"   Embedding dimension: {len(chunk.dense_vector)}")
            assert len(chunk.dense_vector) == 768, "Dimensionality Rule Violated: Vector is not 768."
            
        # 4. Upsert
        print("3. Upserting to Qdrant collection: marra_multimodal_768")
        store = VectorStore(collection_name="marra_multimodal_768")
        store.upsert_chunks(chunks)

        # 5. Hybrid Search
        print("4. Testing Hybrid Search Retrieval...")
        results = hybrid_search("A red square image")
        print(f"   Retrieved {len(results)} results.")
        for res in results:
            print(f"   -> Result metadata: {res.get('metadata')}, RRF Score: {res.get('rrf_score')}")

    finally:
        # 6. Cleanup
        print("5. Cleaning up temporary multimodal chunks...")
        cleanup_multimodal_chunks(chunks)
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

    print("--- Test Completed Successfully ---")

if __name__ == "__main__":
    asyncio.run(main())
