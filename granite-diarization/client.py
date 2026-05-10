import socket
import json
import sys
import time
import threading
from tqdm import tqdm

progress_active = True

def animated_progress():
    """Show animated progress while waiting for server."""
    global progress_active
    with tqdm(desc="Processing", unit="s") as pbar:
        while progress_active:
            time.sleep(1)
            pbar.update(1)

def send_file(file_path, host="localhost", port=8000):
    """Send a file to the transcription server and return the result."""
    global progress_active
    
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host, port))
        
        request = json.dumps({"file_path": file_path})
        client.sendall(request.encode() + b"\n")
        
        print(f"Sending file: {file_path}")
        print("Waiting for server response...")
        
        # Start progress animation in background thread
        progress_thread = threading.Thread(target=animated_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        response = b""
        while True:
            chunk = client.recv(8192)
            if not chunk:
                break
            response += chunk
            if b"\n" in response:
                break
        
        progress_active = False
        client.close()
        
        return json.loads(response.decode().strip())
    except Exception as e:
        progress_active = False
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py <audio_file> [host] [port]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    host = sys.argv[2] if len(sys.argv) > 2 else "localhost"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    
    print("="*60)
    print("Granite ASR Client - Diarized Transcription")
    print("="*60)
    print(f"Server: {host}:{port}")
    print(f"File: {file_path}")
    print("="*60)
    
    start_time = time.time()
    result = send_file(file_path, host, port)
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("TRANSCRIPTION RESULT:")
    print("="*60)
    
    if "error" in result:
        print(f"ERROR: {result['error']}")
    else:
        if "chunks" in result:
            print(f"Processed {result['chunks']} chunks in {elapsed:.1f}s")
            print()
        print(result.get("transcription", "No transcription"))
    
    print("\n" + "="*60)