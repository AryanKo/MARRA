import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.graph import build_graph

def main():
    if len(sys.argv) < 2:
        query = "How does agentic routing improve retrieval?"
    else:
        query = sys.argv[1]
        
    print(f"--- STARTING GRAPH EXECUTION ---")
    print(f"User Query: '{query}'\n")
    
    app = build_graph()
    
    initial_state = {
        "query": query,
        "sub_queries": [],
        "retrieved_docs": [],
        "final_answer": ""
    }
    
    # Stream the graph execution to trace node transitions
    for output in app.stream(initial_state):
        for node_name, state_update in output.items():
            print(f"--- Node Executed: {node_name} ---")
            if "sub_queries" in state_update:
                print(f"Generated Sub-queries: {state_update['sub_queries']}")
            elif "retrieved_docs" in state_update:
                print(f"Retrieved Documents: {len(state_update['retrieved_docs'])} chunk(s)")
            elif "final_answer" in state_update:
                print(f"Final Answer Generated.")
            print("-" * 40)
            
    # Final state is in the last output
    # Actually stream yields partial states. To get the final state cleanly, we can just print the final answer from the last update
    if "synthesizer" in output:
        print("\n=== FINAL ANSWER ===")
        print(output["synthesizer"].get("final_answer", ""))
        print("====================")

if __name__ == "__main__":
    main()
