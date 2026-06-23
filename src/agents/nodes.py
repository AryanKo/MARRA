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

# --- File API Polling Safety ---
MAX_FILE_POLL_SECONDS = 120        # Hard ceiling for File API polling
FILE_POLL_INTERVAL = 2             # Seconds between polls

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

def estimate_media_tokens(file_path: str, media_type: str, file_size: int) -> int:
    """
    Estimates tokens for audio and video files.
    Primary: ffprobe duration extraction (accurate).
    Fallback: Conservative over-estimate to prevent context overflow.
    """
    if media_type == "image":
        return 258
        
    try:
        import subprocess
        import json
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe exited with code {result.returncode}")
        duration = float(json.loads(result.stdout)["format"]["duration"])
        if media_type == "video":
            return int(duration * 300)
        else:
            return int(duration * 32)
    except Exception:
        # FALLBACK: ffprobe failed — use a conservative OVER-estimate.
        # Over-estimating causes files to be skipped by budget_media_files() (safe).
        # Under-estimating causes 400 INVALID_ARGUMENT crashes (unsafe).
        size_in_mb = file_size / (1024 * 1024)
        if media_type == "video":
            estimated = int(size_in_mb * 18_000)
        else:
            estimated = int(size_in_mb * 3_840)
        logging.warning(
            f"ffprobe failed for {file_path}. Using conservative fallback estimate: "
            f"{estimated} tokens for {size_in_mb:.1f}MB {media_type}. "
            f"This file may be skipped by budget checks. Install ffmpeg to fix."
        )
        return estimated

def budget_media_files(docs: list) -> list:
    """
    Enforces per-file and total media token budgets.
    Returns a filtered list of (doc, file_path, file_size) tuples that fit the budget.
    """
    approved = []
    total_bytes = 0
    total_media_tokens = 0
    for doc in docs:
        file_path = doc.metadata.get("file_path")
        media_type = doc.metadata.get("media_type", "image")
        if not file_path or not os.path.exists(file_path):
            continue
            
        file_size = os.path.getsize(file_path)
        estimated_tokens = estimate_media_tokens(file_path, media_type, file_size)
        
        if total_media_tokens + estimated_tokens > (GEMINI_CONTEXT_BUDGET - MAX_HISTORY_TOKENS - RESERVED_SYSTEM_TOKENS):
            logging.warning(f"Media token budget exceeded. Skipping {file_path}.")
            break
            
        total_bytes += file_size
        total_media_tokens += estimated_tokens
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

You MUST also classify the media filter type based on these strict guidelines:
- If the query contains keywords relating to video (e.g., watch, play, clip, video, show), classify as "video".
- If the query contains keywords relating to audio (e.g., sound, listen, hear, voice, audio, music), classify as "audio".
- If the query contains keywords relating to images (e.g., photo, picture, image, drawing, diagram), classify as "image".
- If the query contains keywords relating to documents/PDFs (e.g., read, text, doc, pdf, book), classify as "text".
- If the query is ambiguous, generic, or does not specify a media format, classify as "all".

You MUST output valid JSON ONLY with the exact structure:
{
  "sub_queries": ["query1", "query2"],
  "media_filter": "video" | "audio" | "image" | "text" | "all"
}
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
            
        media_filter = data.get("media_filter", "all")
        if media_filter not in ["video", "audio", "image", "text", "all"]:
            media_filter = "all"
            
        return {"sub_queries": sub_queries, "media_filter": media_filter}
    except ConnectionError as e:
        raise e
    except Exception as e:
        logging.error(f"Planner node failed: {e}")
        return {"sub_queries": [query], "media_filter": "all"}



def retriever_node(state: AgentState) -> Dict[str, Any]:
    sub_queries = state.get("sub_queries", [])
    media_filter = state.get("media_filter", "all")
    
    existing_docs = state.get("retrieved_docs", [])
    seen_texts = {d.text for d in existing_docs}
    new_docs = []
    
    for sq in sub_queries:
        try:
            results = hybrid_search(sq, k=3, media_filter=media_filter)
            
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
    _remote_files = []  # Track remote file handles for guaranteed cleanup (TV-3)
    
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
            mime_type = "image/jpeg"
            if media_type == "image":
                if file_path.endswith(".png"): mime_type = "image/png"
                elif file_path.endswith(".webp"): mime_type = "image/webp"
            elif media_type == "audio":
                if file_path.endswith(".mp3"): mime_type = "audio/mp3"
                elif file_path.endswith(".wav"): mime_type = "audio/wav"
            elif media_type == "video":
                if file_path.endswith(".mp4"): mime_type = "video/mp4"
                elif file_path.endswith(".mov"): mime_type = "video/quicktime"
                elif file_path.endswith(".mpeg"): mime_type = "video/mpeg"
                
            if media_type == "image" and file_size <= 4 * 1024 * 1024:
                with open(file_path, "rb") as f:
                    media_bytes = f.read()
                # Inject direct raw bytes into the payload so Gemini can 'see' natively
                contents.append(types.Part.from_bytes(data=media_bytes, mime_type=mime_type))
            else:
                import time
                api_key = os.environ.get("GEMINI_API_KEY")
                client = genai.Client(api_key=api_key)
                uploaded_file = client.files.upload(file=file_path)
                _remote_files.append((client, uploaded_file.name))

                # Bounded polling with FAILED-state detection (TV-1)
                elapsed = 0
                while uploaded_file.state.name == "PROCESSING":
                    if elapsed >= MAX_FILE_POLL_SECONDS:
                        try:
                            client.files.delete(name=uploaded_file.name)
                        except Exception:
                            pass
                        raise TimeoutError(
                            f"Google File API polling timed out after {MAX_FILE_POLL_SECONDS}s "
                            f"for file '{uploaded_file.name}'. State stuck at PROCESSING."
                        )
                    time.sleep(FILE_POLL_INTERVAL)
                    elapsed += FILE_POLL_INTERVAL
                    uploaded_file = client.files.get(name=uploaded_file.name)
                    logging.info(
                        f"File API poll: {uploaded_file.name} -> {uploaded_file.state.name} "
                        f"({elapsed}/{MAX_FILE_POLL_SECONDS}s)"
                    )

                if uploaded_file.state.name == "FAILED":
                    try:
                        client.files.delete(name=uploaded_file.name)
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Google File API processing FAILED for '{uploaded_file.name}'. "
                        f"The file may be corrupted or use an unsupported codec."
                    )

                contents.append(uploaded_file)
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
    finally:
        # CRITICAL (TV-3): Purge all remote files regardless of success/failure
        for client, file_name in _remote_files:
            try:
                client.files.delete(name=file_name)
                logging.info(f"Cleaned up remote file: {file_name}")
            except Exception as cleanup_err:
                logging.error(
                    f"REMOTE CLEANUP FAILED for {file_name}: {cleanup_err}. "
                    f"File will persist in Google Cloud for up to 48 hours."
                )
