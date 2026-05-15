import urllib.request
import sys
import json

def check_ollama():
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                print(f"[OK] Ollama is running. Models: {len(data.get('models', []))}")
                return True
            else:
                print(f"[FAIL] Ollama returned status {response.status}")
                return False
    except Exception as e:
        print(f"[FAIL] Ollama connection failed: {e}")
        return False

def check_qdrant():
    try:
        req = urllib.request.Request("http://localhost:6333", method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                print(f"[OK] Qdrant is running. Version: {data.get('version', 'unknown')}")
                return True
            else:
                print(f"[FAIL] Qdrant returned status {response.status}")
                return False
    except Exception as e:
        print(f"[FAIL] Qdrant connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Verifying infrastructure...")
    ollama_ok = check_ollama()
    qdrant_ok = check_qdrant()
    
    if ollama_ok and qdrant_ok:
        print("All services are up and running.")
        sys.exit(0)
    else:
        print("Some services are down.")
        sys.exit(1)
