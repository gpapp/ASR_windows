import os
import logging
import numpy as np
import librosa
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_REPO = "gn64/cohere-transcribe-onnx-int8"
MODEL_DIR = Path(__file__).parent / "models"
N_LAYERS = 8
HEADS = 8
HEAD_DIM = 128
MAX_CTX = 1024
MAX_NEW_TOKENS = 448

state = {}


def ensure_model() -> Path:
    from huggingface_hub import snapshot_download

    needed = [
        "cohere-encoder.int8.onnx",
        "cohere-encoder.int8.onnx.data",
        "cohere-decoder.int8.onnx",
        "tokens.txt",
    ]
    if all((MODEL_DIR / f).exists() for f in needed):
        log.info("Model files already present, skipping download.")
        return MODEL_DIR

    log.info("Downloading model files (~2.9 GB), this may take a while...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        allow_patterns=["*.onnx", "*.onnx.data", "tokens.txt"],
        local_dir=str(MODEL_DIR),
    )
    return MODEL_DIR


def load_models():
    model_dir = ensure_model()

    tokens: dict[int, str] = {}
    with open(model_dir / "tokens.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().rsplit(" ", 1)
            if len(parts) == 2:
                tokens[int(parts[1])] = parts[0]
    token_to_id = {v: k for k, v in tokens.items()}
    log.info(f"Loaded {len(tokens)} tokens")

    available = ort.get_available_providers()
    use_dml = "DmlExecutionProvider" in available
    use_dml=False
    providers = ["DmlExecutionProvider", "CPUExecutionProvider"] if use_dml else ["CPUExecutionProvider"]
    log.info(f"Using providers: {providers}")

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = min(4, max(1, os.cpu_count() or 4))
    opts.intra_op_num_threads = min(4, max(1, os.cpu_count() or 4))

    log.info("Loading encoder...")
    encoder = ort.InferenceSession(str(model_dir / "cohere-encoder.int8.onnx"), opts, providers=providers)
    log.info("Loading decoder...")
    decoder = ort.InferenceSession(str(model_dir / "cohere-decoder.int8.onnx"), opts, providers=providers)

    state["encoder"] = encoder
    state["decoder"] = decoder
    state["tokens"] = tokens
    state["token_to_id"] = token_to_id
    state["use_dml"] = use_dml
    state["status"] = "ready"
    log.info("Model ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    state.clear()
    log.info("Server shut down.")


app = FastAPI(lifespan=lifespan)


class TranscribeRequest(BaseModel):
    wav_paths: List[str]
    language: str = "en"


def transcribe_audio(audio: np.ndarray, language: str = "en") -> str:
    encoder = state["encoder"]
    decoder = state["decoder"]
    tokens = state["tokens"]
    token_to_id = state["token_to_id"]
    device = "dml" if state["use_dml"] else "cpu"

    # Run encoder with IOBinding so cross-attention K/V stay on the accelerator device.
    enc_io = encoder.io_binding()
    enc_io.bind_cpu_input("audio", audio.reshape(1, -1).astype(np.float32))
    enc_io.bind_output("n_layer_cross_k", device)
    enc_io.bind_output("n_layer_cross_v", device)
    encoder.run_with_iobinding(enc_io)
    enc_out = enc_io.get_outputs()
    cross_k_ov = enc_out[0]
    cross_v_ov = enc_out[1]

    lang_token = f"<|{language}|>"
    prompt_tokens = [
        "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
        lang_token, lang_token, "<|pnc|>", "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>",
    ]
    prompt_ids = [token_to_id[t] for t in prompt_tokens if t in token_to_id]
    eos_id = token_to_id.get("<|endoftext|>", -1)

    # Allocate KV cache on the accelerator device.
    self_k_ov = ort.OrtValue.ortvalue_from_numpy(
        np.zeros((N_LAYERS, 1, HEADS, MAX_CTX, HEAD_DIM), dtype=np.float32), device, 0
    )
    self_v_ov = ort.OrtValue.ortvalue_from_numpy(
        np.zeros((N_LAYERS, 1, HEADS, MAX_CTX, HEAD_DIM), dtype=np.float32), device, 0
    )

    generated = list(prompt_ids)
    current = np.array([prompt_ids], dtype=np.int64)
    offset = np.array(0, dtype=np.int64)

    dec_io = decoder.io_binding()

    for _ in range(MAX_NEW_TOKENS):
        dec_io.bind_cpu_input("tokens", current)
        dec_io.bind_ortvalue_input("in_n_layer_self_k_cache", self_k_ov)
        dec_io.bind_ortvalue_input("in_n_layer_self_v_cache", self_v_ov)
        dec_io.bind_ortvalue_input("n_layer_cross_k", cross_k_ov)
        dec_io.bind_ortvalue_input("n_layer_cross_v", cross_v_ov)
        dec_io.bind_cpu_input("offset", offset)
        dec_io.bind_output("logits", device)
        dec_io.bind_output("out_n_layer_self_k_cache", device)
        dec_io.bind_output("out_n_layer_self_v_cache", device)

        decoder.run_with_iobinding(dec_io)

        dec_out = dec_io.get_outputs()
        logits = dec_out[0].numpy()  # only ~64 KB crosses to CPU
        self_k_ov = dec_out[1]       # updated KV stays on DML
        self_v_ov = dec_out[2]

        next_id = int(np.argmax(logits[0, -1, :]))
        if next_id == eos_id:
            break
        generated.append(next_id)
        offset = np.array(int(offset) + current.shape[1], dtype=np.int64)
        current = np.array([[next_id]], dtype=np.int64)

    text = "".join(
        tokens.get(t, "").replace("\u2581", " ")
        for t in generated[len(prompt_ids):]
        if not tokens.get(t, "").startswith("<|")
    ).strip()
    return text


@app.get("/health")
def health():
    return {"status": "online", "model": state.get("status", "loading")}


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    results = []
    for path in req.wav_paths:
        try:
            audio, _ = librosa.load(path, sr=16000, mono=True)
            text = transcribe_audio(audio, req.language)
            results.append(text)
        except Exception as e:
            log.error(f"Error transcribing {path}: {e}")
            results.append(f"[ERROR: {e}]")
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
