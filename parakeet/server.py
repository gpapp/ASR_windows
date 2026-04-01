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
from typing import List, Optional
import uvicorn
import traceback
import re
from contextlib import asynccontextmanager


# --- DIRECTML SETTINGS ---
os.environ["DML_VISIBLE_DEVICES"] = "0"
os.environ["NUMBA_DISABLE_JIT"] = "0"
directml_patch.apply()

# Toggle between models here
ASR_MODEL_NAME = "nvidia/parakeet-tdt-1.1b"

HW_CONFIG = {
    "device": torch_directml.device(),
    "dtype": torch.float32  # DML prefers float32 over float16/64
}

def check_hardware():
    print(f">>> Detected DirectML Device: {HW_CONFIG['device']}")

def load_model(model_name):

    try:
        print(f">>> Loading Model: {model_name}...")
        
        # Load initially to CPU
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        
        # Strip float64 tensors while still on CPU to prevent DML crash
        model.to(dtype=torch.float32)
        model.eval() # Set to eval mode BEFORE moving to GPU to freeze weights
        
        # Disable features that cause issues with chunked/streaming inference
        with open_dict(model.cfg):
            model.cfg.use_lhotse = False
            if hasattr(model.cfg, 'preprocessor'):
                model.cfg.preprocessor.dither = 0.0
                model.cfg.preprocessor.pad_to = 0
        
        # Update preprocessor object directly as well
        if hasattr(model, 'preprocessor'):
            model.preprocessor.dither = 0.0
            model.preprocessor.pad_to = 0

        device = HW_CONFIG["device"]
        
        # Moving the whole model is risky on DML if we want to offload parts.
        # Instead, move specific parts.
        if device.type == "privateuseone":
            print(f">>> {model_name}: Selective DML offloading for stability.")
            # Move parameters/buffers to device
            # For AED/Canary models, the decoder is a transformer and usually DML-friendly.
            # In Canary 1b-v2, it is called 'transf_decoder'.
            if "canary" in model_name.lower():
                print(f">>> {model_name}: Moving major components to {device}.")
                if hasattr(model, 'encoder'): model.encoder.to(device)
                if hasattr(model, 'transf_decoder'): model.transf_decoder.to(device)
                if hasattr(model, 'decoder'): model.decoder.to(device)
                if hasattr(model, 'log_softmax'): model.log_softmax.to(device)
            elif "parakeet" in model_name.lower():
                print(f">>> {model_name}: Moving encoder to {device} (stable RNN-T path).")
                if hasattr(model, 'encoder'): model.encoder.to(device)
                # Decoder/Joint on CPU to avoid LSTM/version_counter crashes
                if hasattr(model, 'decoder'): model.decoder.to('cpu')
                if hasattr(model, 'joint'): model.joint.to('cpu')
            else:
                # Generic fallback
                if hasattr(model, 'encoder'): model.encoder.to(device)
                if hasattr(model, 'decoder'): model.decoder.to('cpu')
                if hasattr(model, 'joint'): model.joint.to('cpu')



        else:
            # For CUDA or CPU, just move the whole model
            model.to(device=device)
            
        print(f">>> {model_name} loaded successfully.")
        return model
    except Exception as e:
        print(f"FAILED LOAD {model_name}: {e}")
        traceback.print_exc()
        return None

# Punctuation model - defined here, loaded lazily on first /punctuate call
PUNC_MODEL_NAME = "oliverguhr/fullstop-punctuation-multilang-large"
# Map model output labels to punctuation characters
PUNC_LABEL_MAP = {"PERIOD": ".", "COMMA": ",", "QUESTION": "?", "EXCLAMATION": "!", "0": ""}

def load_punc_model():
    """Lazily load the HF punctuation/capitalization model on first use."""
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
        print(f">>> Loading punctuation model: {PUNC_MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(PUNC_MODEL_NAME)
        model = AutoModelForTokenClassification.from_pretrained(PUNC_MODEL_NAME)
        model.eval()
        # aggregation_strategy="none": we handle grouping manually to correctly
        # assign one label per input word via the SentencePiece word boundary marker.
        pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="none",
            device="cpu",  # Keep on CPU - it's small and fast enough
        )
        print(">>> Punctuation model loaded.")
        return pipe
    except Exception as e:
        print(f"FAILED to load punctuation model: {e}")
        traceback.print_exc()
        return None

def apply_punctuation(pipe, text: str) -> str:
    """Run token-classification and reconstruct punctuated, capitalized text."""
    if not text.strip():
        return text

    # The model works best on chunks of a few hundred words
    MAX_WORDS = 200
    input_words = text.split()

    result_parts = []
    for i in range(0, len(input_words), MAX_WORDS):
        chunk_words = input_words[i:i + MAX_WORDS]
        chunk = " ".join(chunk_words)
        tokens = pipe(chunk)  # aggregation_strategy="none" -> one entry per sub-token

        # SentencePiece marks word-starts with the '▁' prefix (U+2581).
        # We assign the label of the first sub-token of each word to that word.
        word_labels = []
        for tok in tokens:
            word_piece = tok["word"]
            if word_piece.startswith("▁") or not word_labels:
                # First token of a new word
                word_labels.append(tok["entity"])
            # Subsequent sub-tokens: skip (they belong to the previous word)

        # Ensure we have exactly one label per input word
        while len(word_labels) < len(chunk_words):
            word_labels.append("0")

        out = []
        for word, label in zip(chunk_words, word_labels):
            punct = PUNC_LABEL_MAP.get(label, "")
            out.append(word + punct)
        result_parts.append(" ".join(out))

    punctuated = " ".join(result_parts)

    # Capitalize the start of each sentence (after . ? !)
    punctuated = re.sub(r'([.?!]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), punctuated)
    # Capitalize the very first character
    punctuated = punctuated[0].upper() + punctuated[1:] if punctuated else punctuated

    return punctuated


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_hardware()
    app.state.asr_model = load_model(ASR_MODEL_NAME)
    app.state.punc_model = None  # Loaded lazily on first /punctuate call
    yield
    print(">>> Shutting down...")

app = FastAPI(lifespan=lifespan)

class TranscribeRequest(BaseModel):
    wav_paths: List[str]

class PunctuateRequest(BaseModel):
    text: str

@app.get("/health")
async def health_check():
    """Endpoint for clients to verify server status."""
    model_status = "ready" if hasattr(app.state, "asr_model") and app.state.asr_model is not None else "loading"
    punc_status = "loaded" if hasattr(app.state, "punc_model") and app.state.punc_model is not None else "not_loaded"
    return {"status": "online", "model": model_status, "punc_model": punc_status}

@app.post("/punctuate")
async def punctuate(request: PunctuateRequest):
    """Apply punctuation and capitalization to raw ASR output."""
    # Lazy load on first call
    if app.state.punc_model is None:
        app.state.punc_model = load_punc_model()
        if app.state.punc_model is None:
            raise HTTPException(status_code=500, detail="Punctuation model failed to load")
    try:
        result = apply_punctuation(app.state.punc_model, request.text)
        return {"text": result}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe(request: TranscribeRequest):
    model = app.state.asr_model
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # 1. LOAD ALL AUDIOS FOR BATCHING
        audios = []
        for wav_path in request.wav_paths:
            if os.path.exists(wav_path):
                audio, _ = librosa.load(wav_path, sr=16000)
                if len(audio) > 0:
                    audios.append(torch.tensor(audio))
                else:
                    audios.append(torch.zeros(1600)) # 0.1s silence
            else:
                audios.append(torch.zeros(1600))

        if not audios:
            return {"results": []}

        # 2. PADDING AND BATCH CREATION
        # Pad all audios in this batch to the same length
        max_len = max(a.shape[0] for a in audios)
        padded_audios = []
        lengths = []
        for a in audios:
            pad_len = max_len - a.shape[0]
            if pad_len > 0:
                padded_audios.append(torch.nn.functional.pad(a, (0, pad_len)))
            else:
                padded_audios.append(a)
            lengths.append(a.shape[0])

        audio_signal = torch.stack(padded_audios).to(HW_CONFIG["device"])
        signal_len = torch.tensor(lengths).to(HW_CONFIG["device"])

        # 3. BATCHED INFERENCE
        with torch.no_grad():
            # I. PREPROCESSING (Audio -> Mel Spectrogram)
            processed_signal, processed_len = model.preprocessor(
                input_signal=audio_signal, 
                length=signal_len
            )
            
            # II. ENCODER (On GPU/DML)
            try:
                # Modern NeMo models (Canary, Parakeet-TDT)
                encoded, encoded_len = model.encoder(audio_signal=processed_signal, length=processed_len)
            except TypeError:
                # Legacy Parakeet/RNNT models
                encoded, encoded_len = model.encoder(input_signal=processed_signal, input_signal_length=processed_len)

            
            # III. DECODING
            batch_texts = []
            
            if hasattr(model, 'joint'):
                # --- PARAKEET (RNN-T / TDT) PATH ---
                # Move encoded tensors back to CPU for decoding to ensure stability on Parakeet
                encoded_to_use = encoded.detach().cpu().to(torch.float32)
                encoded_len_to_use = encoded_len.detach().cpu()

                if hasattr(model, 'decoding') and hasattr(model.decoding, 'rnnt_decoder_predictions_tensor'):
                    # Multi-sample decoding for RNNT/TDT
                    for b in range(encoded_to_use.shape[0]):
                        hyp = model.decoding.rnnt_decoder_predictions_tensor(
                            encoded_to_use[b:b+1], 
                            encoded_len_to_use[b:b+1], 
                            return_hypotheses=True
                        )
                        res = hyp[0].y_sequence if hasattr(hyp[0], 'y_sequence') else hyp[0]
                        batch_texts.append(model.tokenizer.ids_to_text(res))

                else:
                    for b in range(encoded_to_use.shape[0]):
                        text = model.predict_step(encoded_to_use[b:b+1], encoded_len_to_use[b:b+1])[0]
                        batch_texts.append(text)

            else:
                # --- CANARY (AED) PATH ---
                # Keep tensors on the GPU for Canary decoding
                if encoded.shape[1] > encoded.shape[2]: # (B, D, T)
                    encoded = encoded.transpose(1, 2)
                
                max_time = encoded.shape[1] # (B, T, D)
                encoder_mask = torch.arange(max_time, device=encoded.device).unsqueeze(0) < encoded_len.unsqueeze(1)
                
                res = model.decoding.decode_predictions_tensor(
                    encoder_hidden_states=encoded, 
                    encoder_input_mask=encoder_mask
                )

                
                # Extract texts and clean up
                for text_item in res:
                    text = text_item
                    if hasattr(text, 'text'):
                        text = text.text
                    elif hasattr(text, 'y_sequence'):
                        text = model.tokenizer.ids_to_text(text.y_sequence)
                    elif not isinstance(text, str):
                        text = str(text)
                    
                    # --- CLEANUP LOGIC ---
                    # 1. Strip special NeMo tokens
                    text = re.sub(r"<\|.*?\|>", "", text)
                    # 2. Collapse repeating characters with spaces (D D D, , , ,)
                    text = re.sub(r"\b(\w)\s+\1(\s+\1)+\b", r"\1", text)
                    text = re.sub(r"([,.!?])\s+\1(\s+\1)+", r"\1", text)
                    # 3. Collapse repeating consecutive punctuation
                    text = re.sub(r"([,.!?])\1+", r"\1", text)
                    
                    batch_texts.append(text.strip(" ,.!?"))
        
        return {"results": batch_texts}


    except Exception as e:
        traceback.print_exc()
        return {"results": [], "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


