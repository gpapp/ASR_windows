import sys
import os
import torch
import torch_directml
import directml_patch
import librosa
import numpy as np
import shutil
import nemo.collections.asr as nemo_asr
from omegaconf import DictConfig, open_dict, OmegaConf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import traceback
from contextlib import asynccontextmanager

# --- DIRECTML SETTINGS ---
os.environ["DML_VISIBLE_DEVICES"] = "0"
os.environ["NUMBA_DISABLE_JIT"] = "0"
directml_patch.apply()

ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
ASR_MODEL_NAME = "nvidia/parakeet-tdt_ctc-1.1b"

@asynccontextmanager
async def lifespan(app: FastAPI):
    check_hardware()
    print(">>> Starting Parakeet ASR model load...")
    app.state.asr_model = load_model(ASR_MODEL_NAME)
    print(">>> ASR model load result: ", "Success" if app.state.asr_model else "Failed")
    
    # Warmup models for DirectML stability
    if app.state.asr_model: 
        print(">>> Starting ASR warmup...")
        warmup_asr(app.state.asr_model)
        print(">>> ASR warmup completed.")
    
    yield
    # Cleanup
    if hasattr(app.state, "asr_model"):
        del app.state.asr_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

HW_CONFIG = {
    "device": torch.device("cpu"),
    "dtype": torch.float32,
}

def check_hardware():
    print(">>> Diagnostic: Checking Hardware...")
    if torch.cuda.is_available():
        HW_CONFIG["device"] = torch.device("cuda")
        print(f">>> SUCCESS: CUDA detected.")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        HW_CONFIG["device"] = torch.device("xpu")
        print(">>> SUCCESS: XPU detected.")
    elif torch_directml.is_available():
        HW_CONFIG["device"] = torch_directml.device()
        print(f">>> SUCCESS: DirectML (iGPU) detected: {torch_directml.device_name(0)}")
    else:
        print(">>> INFO: Using CPU.")

def load_model(model_name):
    try:
        print(f">>> Loading Model: {model_name}...")
        
        # Load the model
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        
        model.to(device=HW_CONFIG["device"])
        
        # RNNT decoding on DirectML often triggers version_counter issues.
        # Moving the prediction and joint nets to CPU preserves stability with minimal performance impact.
        if HW_CONFIG["device"].type == "privateuseone":
            if hasattr(model, 'decoder'): 
                print(f">>> {model_name}: Moving prediction net to CPU.")
                model.decoder.to('cpu')
            if hasattr(model, 'joint'): 
                print(f">>> {model_name}: Moving joint net to CPU.")
                model.joint.to('cpu')
            
        model.to(dtype=torch.float32)
        model.eval()
        return model
    except Exception as e:
        print(f"FAILED LOAD {model_name}: {e}")
        traceback.print_exc()
        return None

def warmup_asr(model):
    print(">>> Warming up model on DirectML...")
    try:
        with torch.no_grad():
            dummy_input = torch.randn(1, 16000, device=HW_CONFIG["device"], dtype=torch.float32)
            dummy_len = torch.tensor([16000], device=HW_CONFIG["device"], dtype=torch.long)
            # NeMo RNNT decoding might not support autocast on DML, we use explicit dtype instead
            model.forward(input_signal=dummy_input, input_signal_length=dummy_len)
    except Exception as e:
        print(f">>> Warmup failed (skipping): {e}")

class TranscribeRequest(BaseModel):
    wav_paths: List[str]

@app.get("/health")
async def health_check():
    """Endpoint for clients to verify server status."""
    model_status = "ready" if hasattr(app.state, "asr_model") and app.state.asr_model is not None else "loading"
    return {"status": "online", "model": model_status}

@app.post("/transcribe")
async def transcribe(request: TranscribeRequest):
    model = getattr(app.state, "asr_model", None)
    if model is None: raise HTTPException(status_code=503, detail="ASR Model not initialized")
    
    results = []
    try:
        with torch.no_grad():
            for wav_path in request.wav_paths:
                if not os.path.exists(wav_path):
                    results.append("")
                    continue
                
                # Load audio directly to tensor to bypass NeMo's dataloader logic
                audio_slice, _ = librosa.load(wav_path, sr=16000)
                if len(audio_slice) == 0:
                    results.append("")
                    continue

                audio_signal = torch.tensor(audio_slice).unsqueeze(0).to(HW_CONFIG["device"])
                audio_signal = audio_signal.to(dtype=HW_CONFIG["dtype"])
                
                signal_len = torch.tensor([audio_signal.shape[1]]).to(HW_CONFIG["device"])
                
                # Core low-level inference loop
                encoded, encoded_len = model.forward(input_signal=audio_signal, input_signal_length=signal_len)
                
                # Decoding workaround: Move back to CPU to decode safely
                encoded_cpu = encoded.to('cpu').to(torch.float32)
                encoded_len_cpu = encoded_len.to('cpu')
                
                best_hyp = model.decoding.rnnt_decoder_predictions_tensor(encoded_cpu, encoded_len_cpu, return_hypotheses=True)
                
                # Handle different hypothesis object versions
                res = best_hyp[0].y_sequence if hasattr(best_hyp[0], 'y_sequence') else best_hyp[0]
                chunk_text = model.tokenizer.ids_to_text(res) if not isinstance(res, str) else res
                results.append(chunk_text.strip())
            
            return {"results": results}
    except Exception as e:
        traceback.print_exc()
        return {"results": [], "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
