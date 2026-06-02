from langgraph.graph import StateGraph, START, END
from src.models.schema import AgentState
from src.agents.nodes import planner_node, retriever_node, synthesizer_node

def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("synthesizer", synthesizer_node)
    
    # Define edges (deterministic linear flow)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()
