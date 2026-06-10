import os
import subprocess
import glob
from typing import List
from src.models.schema import DocumentChunk

def chunk_multimodal_file(file_path: str) -> List[DocumentChunk]:
    ext = os.path.splitext(file_path)[1].lower()
    chunks = []
    
    if ext in [".jpg", ".jpeg", ".png", ".webp"]:
        chunks.append(DocumentChunk(
            text="[IMAGE MEDIA PAYLOAD]",
            metadata={
                "file_path": file_path,
                "media_type": "image",
                "start_timestamp": 0,
                "end_timestamp": 0
            }
        ))
    elif ext in [".mp3", ".wav", ".mp4"]:
        media_type = "video" if ext == ".mp4" else "audio"
        
        # Get duration using ffprobe
        import json
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            duration = float(json.loads(result.stdout)["format"]["duration"])
        except Exception:
            duration = 30.0 # Fallback
            
        temp_dir = "/tmp"
        os.makedirs(temp_dir, exist_ok=True)
        
        import uuid
        base_name = f"{uuid.uuid4().hex}"
        output_pattern = os.path.join(temp_dir, f"{base_name}_%03d{ext}")
        
        split_cmd = [
            "ffmpeg", "-i", file_path,
            "-f", "segment", "-segment_time", "30",
            "-c", "copy", "-reset_timestamps", "1",
            "-map", "0", output_pattern
        ]
        
        try:
            subprocess.run(split_cmd, capture_output=True, check=True)
            
            pattern = os.path.join(temp_dir, f"{base_name}_*{ext}")
            segments = sorted(glob.glob(pattern))
            
            for i, seg_path in enumerate(segments):
                start_ts = i * 30.0
                end_ts = min((i + 1) * 30.0, duration)
                chunks.append(DocumentChunk(
                    text=f"[{media_type.upper()} MEDIA PAYLOAD]",
                    metadata={
                        "file_path": seg_path,
                        "media_type": media_type,
                        "start_timestamp": start_ts,
                        "end_timestamp": end_ts
                    }
                ))
        except Exception as e:
            # If slicing fails, we should still try to clean up
            pattern = os.path.join(temp_dir, f"{base_name}_*{ext}")
            for f in glob.glob(pattern):
                try:
                    os.remove(f)
                except OSError:
                    pass
            raise RuntimeError(f"Failed to slice media: {e}")
            
    return chunks

def cleanup_multimodal_chunks(chunks: List[DocumentChunk]):
    """
    Cleans up any generated files in /tmp associated with the chunks.
    This should be called inside a try...finally block after chunks are processed.
    """
    for chunk in chunks:
        if chunk.metadata.get("media_type") in ["audio", "video"]:
            file_path = chunk.metadata.get("file_path")
            if file_path and file_path.startswith("/tmp") and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
