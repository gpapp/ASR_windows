import os
import subprocess
import argparse
import requests
import json
import base64
from pydub import AudioSegment
from tqdm import tqdm

# We'll use the chat endpoint as it has better support for multimodal content blocks
OLLAMA_URL = "http://localhost:11434/api/chat"

def get_args():
    parser = argparse.ArgumentParser(description="Transcribe and Summarize audio/video files.")
    parser.add_argument("files", nargs="+", help="Path to files")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--model", "-m", default="gemma4:e2b", help="Ollama model name")
    return parser.parse_args()

def extract_audio(input_file):
    base_dir = os.path.dirname(os.path.abspath(input_file))
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    wav_path = os.path.join(base_dir, f"{base_name}_temp.wav")
    
    print(f"-> Extracting audio: {os.path.basename(input_file)}")
    
    # Gemma 4 Technical Specs: 16kHz, Mono, 32-bit float
    try:
        subprocess.run([
            'ffmpeg', '-i', input_file, 
            '-ar', '16000', 
            '-ac', '1', 
            '-c:a', 'pcm_f32le', # Mandatory for 32-bit float in WAV for Gemma 4
            '-y', wav_path
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error: {e.stderr}")
        raise e
        
    return wav_path

def chat_gemma_audio(prompt, model, audio_b64, file_handle):
    # Payload for Ollama's Chat API
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [audio_b64] # Ollama maps multimodal audio to the 'images' field
            }
        ],
        "stream": True,
        "options": {
            "temperature": 1.0,
            "num_predict": 512
        }
    }

    full_response = []
    try:
        # Long timeout because audio encoding is CPU-intensive and can cause lag
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=180) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    text = chunk.get("message", {}).get("content", "")
                    if file_handle:
                        file_handle.write(text)
                        file_handle.flush()
                    full_response.append(text)
    except Exception as e:
        print(f"\n[ERROR] API Call failed: {e}")
        
    return "".join(full_response)

def process_file(file_path, model, manual_output_dir):
    file_path = os.path.abspath(file_path)
    output_dir = manual_output_dir if manual_output_dir else os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    wav_path = extract_audio(file_path)
    audio = AudioSegment.from_wav(wav_path)
    
    # REDUCED CHUNK SIZE: Using 15s to prevent GGML_ASSERT errors/crashes in Ollama runner
    chunk_ms = 60000 
    chunks = [audio[i:i + chunk_ms] for i in range(0, len(audio), chunk_ms)]
    
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    print(f"-> Transcribing {len(chunks)} segments...")
    
    full_transcript = []
    
    # Store temp chunks in the output directory to avoid permission issues
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(tqdm(chunks, desc="Processing")):
            temp_chunk = os.path.join(output_dir, f"tmp_seg_{i}.wav")
            chunk.export(temp_chunk, format="wav")
            
            with open(temp_chunk, "rb") as bf:
                audio_b64 = base64.b64encode(bf.read()).decode('utf-8')
            
            # Refined prompt with <audio> tag for model attention
            prompt = "<audio>Transcribe this speech segment accurately."
            
            response = chat_gemma_audio(prompt, model, audio_b64, f)
            f.write(" ") # Add spacing between transcribed segments
            full_transcript.append(response)
            
            if os.path.exists(temp_chunk):
                os.remove(temp_chunk)

    if "".join(full_transcript).strip():
        # Summary phase
        md_path = os.path.join(output_dir, f"{base_name}.md")
        print(f"-> Summarizing to: {os.path.basename(md_path)}")
        with open(md_path, "w", encoding="utf-8") as f:
            summary_payload = {
                "model": model,
                "messages": [{
                    "role": "user", 
                    "content": f"Summarize the following transcript into professional Markdown with sections for Topics and Actions:\n\n{' '.join(full_transcript)}"
                }],
                "stream": False
            }
            try:
                r = requests.post(OLLAMA_URL, json=summary_payload)
                summary_text = r.json().get("message", {}).get("content", "")
                f.write(summary_text)
            except Exception as e:
                print(f"Summary failed: {e}")
    
    if os.path.exists(wav_path):
        os.remove(wav_path)
    print(f"Finished processing: {base_name}")

if __name__ == "__main__":
    args = get_args()
    for file in args.files:
        if os.path.exists(file):
            process_file(file, args.model, args.output)
        else:
            print(f"File not found: {file}")