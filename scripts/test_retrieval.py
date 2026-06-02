import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.hybrid_search import hybrid_search
import logging

def main():
    if len(sys.argv) < 2:
        query = "agentic routing"
    else:
        query = sys.argv[1]
        
    print(f"Running hybrid search for query: '{query}'")
    
    try:
        results = hybrid_search(query, k=3)
        
        print("\n--- TOP 3 FUSED RESULTS ---\n")
        if not results:
            print("No results found.")
            return

        for i, res in enumerate(results):
            print(f"Rank {i+1}:")
            print(f"RRF Score:    {res['rrf_score']:.4f}")
            print(f"Dense Score:  {res['dense_score']:.4f}")
            print(f"Sparse Score: {res['sparse_score']:.4f}")
            snippet = res['text'][:150].replace("\n", " ")
            print(f"Text Snippet: {snippet}...")
            print("-" * 40)
    except Exception as e:
        logging.error(f"Search failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
