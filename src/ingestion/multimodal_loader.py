import os
import subprocess
import glob
import signal
import sys
from typing import List
from src.models.schema import DocumentChunk

# --- Subprocess Safety Constants ---
FFMPEG_TIMEOUT_SECONDS = 300    # 5 min hard ceiling for segment splitting
FFPROBE_TIMEOUT_SECONDS = 10   # ffprobe should complete in < 1 second

def chunk_multimodal_file(file_path: str) -> List[DocumentChunk]:
    ext = os.path.splitext(file_path)[1].lower()
    chunks = []
    
    if ext in [".jpg", ".jpeg", ".png", ".webp"]:
        chunks.append(DocumentChunk(
            text=f"[IMAGE MEDIA PAYLOAD: {os.path.basename(file_path)}]",
            metadata={
                "file_path": file_path,
                "media_type": "image",
                "start_timestamp": 0,
                "end_timestamp": 0
            }
        ))
    elif ext in [".mp3", ".wav", ".mp4", ".mov", ".mpeg"]:
        media_type = "video" if ext in [".mp4", ".mov", ".mpeg"] else "audio"
        
        # Get duration using ffprobe
        import json
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT_SECONDS)
        try:
            duration = float(json.loads(result.stdout)["format"]["duration"])
        except Exception:
            duration = 30.0 # Fallback
            
        temp_dir = "/tmp"
        os.makedirs(temp_dir, exist_ok=True)
        
        import uuid
        base_name = f"{uuid.uuid4().hex}"
        output_pattern = os.path.join(temp_dir, f"{base_name}_%03d{ext}")
        
        if media_type == "audio":
            split_cmd = [
                "ffmpeg", "-i", file_path,
                "-f", "segment", "-segment_time", "30",
                "-map", "0:a", "-vn", output_pattern
            ]
        else:
            split_cmd = [
                "ffmpeg", "-i", file_path,
                "-f", "segment", "-segment_time", "30",
                "-c", "copy", "-reset_timestamps", "1",
                "-map", "0", output_pattern
            ]
        
        try:
            # Start FFmpeg in a new process group so we can kill the entire tree on timeout.
            popen_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
            if sys.platform != "win32":
                popen_kwargs["start_new_session"] = True
            else:
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

            proc = subprocess.Popen(split_cmd, **popen_kwargs)
            try:
                stdout, stderr = proc.communicate(timeout=FFMPEG_TIMEOUT_SECONDS)
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(
                        proc.returncode, split_cmd, stdout, stderr
                    )
            except subprocess.TimeoutExpired:
                # Kill the entire process group to prevent orphaned ffmpeg workers
                if sys.platform != "win32":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
                proc.wait()  # Reap the zombie
                raise RuntimeError(
                    f"FFmpeg timed out after {FFMPEG_TIMEOUT_SECONDS}s processing {file_path}"
                )
            
            pattern = os.path.join(temp_dir, f"{base_name}_*{ext}")
            segments = sorted(glob.glob(pattern))
            
            for i, seg_path in enumerate(segments):
                start_ts = i * 30.0
                end_ts = min((i + 1) * 30.0, duration)
                chunks.append(DocumentChunk(
                    text=f"[{media_type.upper()} MEDIA PAYLOAD - {base_name}]",
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
