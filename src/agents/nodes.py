import os
import json
import logging
from typing import Dict, Any
import ollama
from google import genai
from google.genai import types
from qdrant_client.http.exceptions import ResponseHandlingException
from src.models.schema import AgentState, DocumentChunk
from src.retrieval.hybrid_search import hybrid_search
from src.utils.observability import trace_span as span

LLM_MODEL = "llama3.1:8b"

# --- Context Budget Configuration ---
GEMINI_CONTEXT_BUDGET = 900_000    # Total input token budget for gemini-3.5-flash
RESERVED_SYSTEM_TOKENS = 500       # Reserved for system prompt + user query
MAX_HISTORY_TOKENS = 8_000         # Hard cap on history tokens
MAX_HISTORY_TURNS = 20             # Rolling window: keep last N turns
MAX_MEDIA_BYTES_PER_FILE = 4 * 1024 * 1024   # 4MB per media file
MAX_TOTAL_MEDIA_BYTES = 15 * 1024 * 1024     # 15MB total media budget
MAX_PLANNER_HISTORY_TURNS = 10     # Planner gets fewer turns (local model)

def estimate_token_count(text: str) -> int:
    """Fast heuristic: ~4 chars per token for English text."""
    return max(1, len(text) // 4)

def truncate_history(history: list, max_turns: int, max_tokens: int) -> list:
    """
    Rolling-window truncation: keeps the most recent `max_turns` messages,
    then trims from the oldest until the total token count fits under `max_tokens`.
    Always preserves at least the last 2 messages for conversational coherence.
    """
    if not history:
        return []

    # Phase 1: Windowed truncation — keep last N turns
    truncated = history[-max_turns:]

    # Phase 2: Token-aware trim — remove oldest messages until under budget
    total_tokens = sum(estimate_token_count(m.get("content", "")) for m in truncated)

    while total_tokens > max_tokens and len(truncated) > 2:
        removed = truncated.pop(0)
        total_tokens -= estimate_token_count(removed.get("content", ""))

    return truncated

def budget_media_files(docs: list) -> list:
    """
    Enforces per-file and total media byte budgets.
    Returns a filtered list of (doc, file_path, file_size) tuples that fit the budget.
    """
    approved = []
    total_bytes = 0
    for doc in docs:
        file_path = doc.metadata.get("file_path")
        if not file_path or not os.path.exists(file_path):
            continue
        file_size = os.path.getsize(file_path)
        if file_size > MAX_MEDIA_BYTES_PER_FILE:
            logging.warning(f"Media file {file_path} exceeds per-file budget ({file_size} > {MAX_MEDIA_BYTES_PER_FILE}). Skipping.")
            continue
        if total_bytes + file_size > MAX_TOTAL_MEDIA_BYTES:
            logging.warning(f"Total media budget exceeded. Skipping {file_path}.")
            break
        total_bytes += file_size
        approved.append((doc, file_path, file_size))
    return approved

@span(name="ollama_generation")
def chat_with_ollama(model: str, messages: list, format: str = None) -> dict:
    if format:
        return ollama.chat(model=model, messages=messages, format=format)
    return ollama.chat(model=model, messages=messages)

@span(name="gemini_multimodal_synthesis")
def generate_with_gemini(contents: list) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing.")
        
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=contents
    )
    return response.text

def planner_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    history = state.get("history", [])
    
    # Guard: if query is empty or whitespace, skip planning entirely
    if not query or not query.strip():
        logging.warning("Planner received empty query. Returning empty sub_queries.")
        return {"sub_queries": []}
    
    # Truncate history for local model budget
    history = truncate_history(history, MAX_PLANNER_HISTORY_TURNS, max_tokens=4000)
    
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
    history = state.get("history", [])
    
    # Truncate history for cloud model budget
    history = truncate_history(history, MAX_HISTORY_TURNS, MAX_HISTORY_TOKENS)
    
    contents = []
    
    # 1. Inject Conversational History
    if history:
        contents.append("Conversational History:")
        for msg in history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            contents.append(f"{role}: {content}")
            
    # 2. Inject Retrieved Context / Multimodal Raw Bytes
    contents.append("Retrieved Context:")
    text_docs = [d for d in docs if d.metadata.get("media_type", "text") == "text"]
    media_docs = [d for d in docs if d.metadata.get("media_type", "text") != "text"]
    
    for doc in text_docs:
        contents.append(doc.text)
    
    approved_media = budget_media_files(media_docs)
    for doc, file_path, file_size in approved_media:
        media_type = doc.metadata.get("media_type")
        try:
            with open(file_path, "rb") as f:
                media_bytes = f.read()
                
            mime_type = "image/jpeg"
            if media_type == "image":
                if file_path.endswith(".png"): mime_type = "image/png"
                elif file_path.endswith(".webp"): mime_type = "image/webp"
            elif media_type == "audio":
                if file_path.endswith(".mp3"): mime_type = "audio/mp3"
                elif file_path.endswith(".wav"): mime_type = "audio/wav"
            elif media_type == "video":
                if file_path.endswith(".mp4"): mime_type = "video/mp4"
                
            # Inject direct raw bytes into the payload so Gemini can 'see' or 'hear' it natively
            contents.append(types.Part.from_bytes(data=media_bytes, mime_type=mime_type))
        except Exception as e:
            logging.warning(f"Failed to load media chunk {file_path}: {e}")
                    
    # 3. Inject User Query and Instructions
    prompt = f"""
You are an expert AI assistant. Answer the user's query based ONLY on the provided context.
If the context does not contain the answer, state that you do not know.
Do not use JSON formatting, answer clearly in markdown.

User Query: {query}
"""
    contents.append(prompt)
    
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key or "your_" in api_key:
            raise ValueError("CRITICAL INFRASTRUCTURE ERROR: The API container cannot read a valid GEMINI_API_KEY from the host environment.")
            
        final_answer = generate_with_gemini(contents)
        return {"final_answer": final_answer}
    except ValueError as ve:
        logging.error(str(ve))
        return {"final_answer": f"Error: {str(ve)}"}
    except Exception as e:
        logging.error(f"Synthesizer node failed: {e}")
        return {"final_answer": "Error: Unable to synthesize answer via Gemini Cloud."}
