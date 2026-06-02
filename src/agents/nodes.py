import json
import logging
from typing import Dict, Any
import ollama
from qdrant_client.http.exceptions import ResponseHandlingException
from src.models.schema import AgentState, DocumentChunk
from src.retrieval.hybrid_search import hybrid_search
from src.utils.observability import trace_span as span

LLM_MODEL = "llama3.1:8b"

@span(name="ollama_generation")
def chat_with_ollama(model: str, messages: list, format: str = None) -> dict:
    if format:
        return ollama.chat(model=model, messages=messages, format=format)
    return ollama.chat(model=model, messages=messages)

def planner_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    history = state.get("history", [])
    
    system_prompt = """You are an expert search planner. Analyze the user's latest query, using the conversational history for context if needed to resolve pronouns or reference.
Break it down into 1 to 3 targeted search queries.
You MUST output valid JSON ONLY with the exact structure: {"sub_queries": ["query1", "query2"]}.
Do not output markdown, explanations, or any other text.
"""
    if history:
        system_prompt += "\nConversational History:\n"
        for msg in history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            system_prompt += f"{role}: {content}\n"
            
    try:
        response = chat_with_ollama(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Query: {query}"}
            ],
            format="json"
        )
        content = response["message"]["content"]
        data = json.loads(content)
        
        sub_queries = data.get("sub_queries", [query])
        if not isinstance(sub_queries, list):
            sub_queries = [query]
            
        if query not in sub_queries:
            sub_queries.append(query)
            
        return {"sub_queries": sub_queries}
    except ConnectionError as e:
        raise e
    except Exception as e:
        logging.error(f"Planner node failed: {e}")
        return {"sub_queries": [query]}



def retriever_node(state: AgentState) -> Dict[str, Any]:
    sub_queries = state.get("sub_queries", [])
    
    existing_docs = state.get("retrieved_docs", [])
    seen_texts = {d.text for d in existing_docs}
    new_docs = []
    
    for sq in sub_queries:
        try:
            results = hybrid_search(sq, k=3)
            
            for res in results:
                text = res["text"]
                if text not in seen_texts:
                    seen_texts.add(text)
                    new_docs.append(
                        DocumentChunk(text=text, metadata=res.get("metadata", {}))
                    )
        except (ConnectionError, ResponseHandlingException) as e:
            raise e
        except Exception as e:
            logging.error(f"Retriever node failed for query '{sq}': {e}")
            
    return {"retrieved_docs": new_docs}


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    
    context_text = "\n\n".join([d.text for d in docs])
    
    prompt = f"""
You are an expert AI assistant. Answer the user's query based ONLY on the provided context.
If the context does not contain the answer, state that you do not know.
Do not use JSON formatting, answer clearly in markdown.

<context>
{context_text}
</context>

User Query: {query}
"""
    
    try:
        response = chat_with_ollama(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"final_answer": response["message"]["content"]}
    except ConnectionError as e:
        raise e
    except Exception as e:
        logging.error(f"Synthesizer node failed: {e}")
        return {"final_answer": "Error: Unable to synthesize answer."}
