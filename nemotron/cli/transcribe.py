import sys
import os
import subprocess
import torch
import librosa
import numpy as np
import re
from tqdm import tqdm
import nemo.collections.asr as nemo_asr

def get_silence_timestamps(file_path):
    """Uses FFmpeg to find gaps of silence longer than 1.5 seconds across the whole file."""
    command = [
        'ffmpeg', '-i', file_path,
        '-af', 'silencedetect=noise=-30dB:d=1.5',
        '-f', 'null', '-'
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stderr
    # Capture the end of a silence period as the start of a new turn
    timestamps = re.findall(r"silence_end: (\d+\.\d+)", output)
    return [float(t) for t in timestamps]

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"

def process_files(file_paths):
    device = torch.device("cpu")
    chunk_duration = 30  
    sample_rate = 16000

    print(">>> Mode: CPU with Independent Turn Detection")
    
    try:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/nemotron-speech-streaming-en-0.6b")
        asr_model.to(device).eval()
    except Exception as e:
        print(f"FAILED TO LOAD MODEL: {e}")
        return

    for file_path in tqdm(file_paths, desc="Batch Progress", unit="file"):
        if not os.path.exists(file_path): continue

        filename = os.path.basename(file_path)
        temp_wav = f"temp_{os.getpid()}.wav"
        
        tqdm.write(f" -> Pre-processing audio structure: {filename}")
        subprocess.run(['ffmpeg', '-y', '-i', file_path, '-ar', str(sample_rate), '-ac', '1', '-vn', '-loglevel', 'error', temp_wav], check=True)
        
        # 1. Get real silence timestamps from the whole file first
        silence_ends = get_silence_timestamps(temp_wav)

        try:
            total_duration = librosa.get_duration(path=temp_wav)
            full_transcript = f"{format_time(0)} SPEAKER: " 
            num_chunks = int(np.ceil(total_duration / chunk_duration))
            
            # Keep track of which silence markers we've already "used"
            used_silence_indices = set()

            for i in tqdm(range(num_chunks), desc="   Transcribing", leave=False):
                offset = i * chunk_duration
                audio_slice, _ = librosa.load(temp_wav, sr=sample_rate, offset=offset, duration=chunk_duration)
                if len(audio_slice) == 0: continue

                audio_signal = torch.tensor(audio_slice).unsqueeze(0).to(device)
                signal_len = torch.tensor([audio_signal.shape[1]]).to(device)
                
                with torch.no_grad():
                    encoded, encoded_len = asr_model.forward(input_signal=audio_signal, input_signal_length=signal_len)
                    best_hyp = asr_model.decoding.rnnt_decoder_predictions_tensor(encoded, encoded_len, return_hypotheses=True)
                    res = best_hyp[0].y_sequence if hasattr(best_hyp[0], 'y_sequence') else best_hyp[0]
                    chunk_text = asr_model.tokenizer.ids_to_text(res) if not isinstance(res, str) else res
                    
                if chunk_text.strip():
                    # Check for "Real" silence within the current chunk duration
                    # We check if a silence_end timestamp falls within [offset, offset + chunk_duration]
                    # and ensure it's not exactly at the very end of the chunk to avoid edge-case overlap
                    current_window_end = offset + chunk_duration
                    
                    found_turn = False
                    for idx, s_end in enumerate(silence_ends):
                        # If a silence end happens IN this chunk AND we haven't used it yet
                        if offset < s_end <= current_window_end and idx not in used_silence_indices:
                            # Avoid triggering on the very boundary of the chunking window
                            # unless it's a significant distance from the edge
                            full_transcript += chunk_text.strip() + f"\n\n{format_time(s_end)} SPEAKER: "
                            used_silence_indices.add(idx)
                            found_turn = True
                            break # Only one turn per 30s chunk to keep it clean
                    
                    if not found_turn:
                        full_transcript += chunk_text.strip() + " "

            # Clean up trailing speaker labels
            final_output = re.sub(r"\[\d+:\d+:\d+\] SPEAKER:$", "", full_transcript.strip()).strip()

            output_txt = os.path.splitext(file_path)[0] + ".txt"
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(final_output)
            
            tqdm.write(f" -> Success! Created flow-aware transcript: {os.path.basename(output_txt)}")
            
        except Exception as e:
            tqdm.write(f" -> Error: {e}")
        finally:
            if os.path.exists(temp_wav):
                try: os.remove(temp_wav)
                except: pass

if __name__ == "__main__":
    process_files(sys.argv[1:])