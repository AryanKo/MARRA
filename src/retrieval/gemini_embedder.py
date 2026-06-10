import os
from google import genai
from google.genai import types
import tenacity
from src.models.schema import DocumentChunk

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
    stop=tenacity.stop_after_attempt(5),
    retry=tenacity.retry_if_exception_type(Exception)
)
def embed_chunk(chunk: DocumentChunk) -> list[float]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing.")
        
    client = genai.Client(api_key=api_key)
    
    media_type = chunk.metadata.get("media_type", "text")
    file_path = chunk.metadata.get("file_path")
    
    if media_type == "text":
        content = chunk.text
    else:
        if not file_path or not os.path.exists(file_path):
            raise ValueError(f"File path missing or invalid for media_type {media_type}")
            
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
            
        content = types.Content(
            parts=[
                types.Part.from_bytes(data=media_bytes, mime_type=mime_type)
            ]
        )
        
    response = client.models.embed_content(
        model='gemini-embedding-2',
        contents=content,
        config=types.EmbedContentConfig(output_dimensionality=768)
    )
    
    # Do NOT use Python array slicing. The API handles truncation via output_dimensionality.
    return response.embeddings[0].values
