"""
Model loading and initialization for ONNX inference.
"""

import os
import site
from pathlib import Path

import json
import numpy as np
import onnxruntime as ort
import structlog

from settings import Settings
from model_state import state, KVCachePool

log = structlog.get_logger()


# ============================================================================
# iGPU Hardware Target Configuration
# ============================================================================

def get_igpu_session_options(provider_type: str = "DirectML", settings: Settings = Settings()):
    """
    Generates execution provider list and session optimizations for iGPUs.
    
    Optimizes for sequence generation models like Conformer/Transformer with KV-Caching.
    Supports both DirectML (universal Windows) and OpenVINO (Intel-optimized).
    
    Args:
        provider_type: "DirectML" (recommended) or "OpenVINO"
    
    Returns:
        Tuple of (providers_list, provider_options, session_options)
    """
    opts: ort.SessionOptions = ort.SessionOptions()
    
    if provider_type.lower() == "openvino":
        # Optimizations for sequence generation models with KV caching
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.add_session_config_entry("optimization.transformers.onnx_model_type", "encoder_decoder")
        
        # Locate OpenVINO DLL path (for OpenVINO provider fallback)
        site_packages_dirs = site.getsitepackages()
        openvino_dll_path = None
        for packages_dir in site_packages_dirs:
            potential_path = os.path.join(packages_dir, "openvino", "libs")
            if os.path.exists(potential_path):
                openvino_dll_path = potential_path
                break

        if openvino_dll_path:
            log.info("openvino_binaries_found", path=openvino_dll_path)
            os.add_dll_directory(openvino_dll_path)
            os.environ["PATH"] = openvino_dll_path + os.path.pathsep + os.environ["PATH"]

        providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        ov_config = {
            "INFERENCE_PRECISION_HINT": "f16"  # Forces FP16 precision on the GPU plugin
        }

        provider_options = [{"device_type": "GPU","load_config": json.dumps(ov_config)},{}]  
        log.info("igpu_config_created", provider="OpenVINO")
        return providers, provider_options, opts
        
    elif provider_type.lower() == "directml":
        # Global Session configuration optimized for DirectML stability       
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  
        opts.enable_mem_pattern = False 
        # Basic optimizations only to prevent invalid command errors and driver crashes
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        log.info("igpu_config_created", provider="DirectML")
        return providers, None, opts
    elif provider_type.lower() == "cpu":
        ## CPU fallback with optimized threading for multi-core CPUs
        providers = ["CPUExecutionProvider"]
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_mem_pattern = False
        opts.intra_op_num_threads = settings.cpu_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return providers, None, opts

    # Fallback to CPU only
    log.warning("igpu_provider_not_recognized", provider=provider_type)
    return ["CPUExecutionProvider"], None, opts


# ============================================================================
# Model Loading
# ============================================================================

def ensure_model(settings: Settings) -> Path:
    """Download model files if not present."""
    from huggingface_hub import snapshot_download
    
    needed = [
        "tokenizer.json",
        f"onnx/encoder_model{settings.encoder_model_type}.onnx",
        f"onnx/encoder_model{settings.encoder_model_type}.onnx",
        f"onnx/encoder_model{settings.encoder_model_type}.onnx_data",
        f"onnx/encoder_model{settings.encoder_model_type}.onnx_data_1",
        f"onnx/decoder_model_merged{settings.decoder_model_type}.onnx",
        f"onnx/decoder_model_merged{settings.decoder_model_type}.onnx_data"
    ]
    
    if all((settings.model_dir / f).exists() for f in needed):
        log.info("model_files_present", path=str(settings.model_dir))
        return settings.model_dir
    
    log.info("downloading_model", repo=settings.model_repo, size_gb=2.9)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_download(
        repo_id=settings.model_repo,
        allow_patterns=[f"decoder_model_merged{settings.decoder_model_type}.onnx*", f"encoder_model{settings.encoder_model_type}.onnx*", "tokenizer.json"],
        local_dir=str(settings.model_dir),
    )
    
    log.info("model_download_complete")
    return settings.model_dir

def ensure_vad_model(settings: Settings) -> Path:
    """Downloads the Silero VAD ONNX model from HF Hub if not present locally."""
    filename=f"onnx/model{settings.vad_model_type}.onnx"
    local_path = settings.vad_model_dir / filename
    if local_path.exists():
        log.info("vad_model_file_present", path=str(local_path))
        return local_path

    log.info("downloading_vad_model", repo=settings.vad_model_repo, file=filename)
    settings.vad_model_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=settings.vad_model_repo,
            filename=filename,
            local_dir=str(settings.vad_model_dir),
            token=settings.hf_token
        )
        return local_path
    except Exception as e:
        log.error("vad_download_failed", error=str(e))
        raise


def ensure_embedding_model(settings: Settings) -> Path:
    """Downloads the ONNX embedding model from HF Hub if not present locally."""
    local_path = settings.embedding_model_dir / settings.embedding_model_filename
    if local_path.exists():
        log.info("embedding_model_file_present", path=str(local_path))
        return local_path

    log.info("downloading_embedding_model", repo=settings.embedding_model_repo, file=settings.embedding_model_filename)
    settings.embedding_model_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=settings.embedding_model_repo,
            filename=settings.embedding_model_filename,
            local_dir=str(settings.embedding_model_dir),
            token=settings.hf_token
        )
        return local_path
    except Exception as e:
        log.error("embedding_download_failed", error=str(e))
        raise

def load_models(settings: Settings):
    """Load encoder, decoder, and auxiliary models with optimized iGPU execution."""
    
    # 2. Ensure model files are downloaded
    model_dir = ensure_model(settings)
    
    # 3. Load vocabulary
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    token_to_id = tokenizer.get_vocab()
    tokens = {v: k for k, v in token_to_id.items()}
    log.info("vocabulary_loaded", token_count=len(tokens))
    
    # Pre-compute prompt tokens for transcription
    prompt_tokens = [
        "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
        "<|en|>", "<|en|>", "<|pnc|>", "<|noitn|>", "<|timestamp|>", "<|nodiarize|>",
    ]
    pre_computed_prompt_ids = [token_to_id[t] for t in prompt_tokens if t in token_to_id]
    pre_computed_eos_id = token_to_id.get("<|endoftext|>", -1)
    pre_computed_prompt_array = np.array([pre_computed_prompt_ids], dtype=np.int64)
    
    # 4. Configure iGPU execution providers (DirectML recommended for Windows)
    PROVIDER_SELECTION = settings.provider_type
    providers, provider_options, opts = get_igpu_session_options(PROVIDER_SELECTION, settings)
    cpu_providers, cpu_provider_options, cpu_opts =  get_igpu_session_options("CPU", settings)
    
    # 5. Load encoder and decoder models
    log.info("loading_encoder_model", provider=PROVIDER_SELECTION)
    try:
        cohere_encoder = ort.InferenceSession(
            str(model_dir / f"onnx/encoder_model{settings.encoder_model_type}.onnx"),
            providers=providers,
            provider_options=provider_options if provider_options else None,
            sess_options=opts,
        )
        log.info("encoder_model_loaded_successfully")
    except Exception as e:
        log.error("encoder_initialization_failed", error=str(e), fallback="CPU")
        cohere_encoder = ort.InferenceSession(
            str(model_dir / f"onnx/encoder_model{settings.encoder_model_type}.onnx"),
            providers=cpu_providers,
            provider_options=cpu_provider_options if cpu_provider_options else None,
            sess_options=cpu_opts
        )
    

    log.info("loading_decoder_model")
    try:
        # Decoder: Use CPU provider due to DirectML compatibility issues with multi-head attention
        # DirectML has known issues with certain attention patterns in this model configuration
        # CPU execution is still very fast for token-by-token generation

        cohere_decoder = ort.InferenceSession(
            str(model_dir / f"onnx/decoder_model_merged{settings.decoder_model_type}.onnx"),
            providers=cpu_providers,
            provider_options=cpu_provider_options if cpu_provider_options else None,
            sess_options=cpu_opts
        )
        log.info("decoder_model_loaded_successfully", note="Using CPU provider due to DirectML compatibility")
    except Exception as e:
        log.error("decoder_initialization_failed", error=str(e))
        raise
    
    # 6. Update model state
    state.encoder = cohere_encoder
    state.decoder = cohere_decoder
    state.tokens = tokens
    state.token_to_id = token_to_id
    state.pre_computed_prompt_ids = pre_computed_prompt_ids
    state.pre_computed_eos_id = pre_computed_eos_id
    state.pre_computed_prompt_array = pre_computed_prompt_array
    state.kv_pool = KVCachePool(settings)
    
    # 7. Load auxiliary models for diarization
    log.info("loading_vad_model")
    
    # Load ONNX VAD with iGPU using ensure_vad_model
    try:
        vad_onnx_path = str(ensure_vad_model(settings))
        
        # Direct loading of the optimized VAD session
        state.vad_session = ort.InferenceSession(
            vad_onnx_path,
            providers=cpu_providers,
            provider_options=cpu_provider_options if cpu_provider_options else None,
            sess_options=cpu_opts
        )
        log.info("vad_model_loaded", provider="cpu")
    except Exception as e:
        log.error("vad_cpu_load_failed", error=str(e))
        raise
    
    # Load embedding model with iGPU using ensure_embedding_model
    log.info("loading_embedding_model")
    try:
        emb_path = str(ensure_embedding_model(settings))
        state.embedding_session = ort.InferenceSession(
            emb_path,
            sess_options=opts,
            providers=providers,
            provider_options=provider_options if provider_options else None
        )
        log.info("embedding_model_loaded", path=emb_path, provider=PROVIDER_SELECTION)
    except Exception as e:
        log.error("embedding_model_loading_failed", error=str(e), fallback="CPU")
        try:
            state.embedding_session = ort.InferenceSession(emb_path, providers=["CPUExecutionProvider"])
        except Exception as e2:
            log.error("embedding_model_cpu_fallback_failed", error=str(e2))
    
    state.status = "ready"
    log.info("all_models_loaded", provider=PROVIDER_SELECTION)


def reload_encoder_session(settings: Settings, force_cpu: bool = False):
    """
    Recreate the encoder ONNX session to release DirectML GPU memory.

    DirectML's memory arena grows with each encoder call but never shrinks.
    Recreating the session forces DirectML to release all GPU allocations
    associated with the old session.

    When *force_cpu* is True, skip GPU and load directly on CPU — used
    as OOM recovery when GPU memory is exhausted.
    """
    from model_state import state
    import gc

    log.info("reloading_encoder_session", force_cpu=force_cpu)

    cpu_providers, cpu_provider_options, cpu_opts = get_igpu_session_options("CPU", settings)
    model_dir = ensure_model(settings)
    model_path = str(model_dir / f"onnx/encoder_model{settings.encoder_model_type}.onnx")

    old = state.encoder
    state.encoder = None
    del old
    gc.collect()

    if force_cpu:
        state.encoder = ort.InferenceSession(
            model_path,
            providers=cpu_providers,
            provider_options=cpu_provider_options if cpu_provider_options else None,
            sess_options=cpu_opts,
        )
        log.warning("encoder_reloaded_cpu_fallback")
    else:
        providers, provider_options, opts = get_igpu_session_options(settings.provider_type, settings)
        try:
            state.encoder = ort.InferenceSession(
                model_path,
                providers=providers,
                provider_options=provider_options if provider_options else None,
                sess_options=opts,
            )
            log.info("encoder_session_reloaded", provider=settings.provider_type)
        except Exception as e:
            log.error("encoder_reload_failed", error=str(e), fallback="CPU")
            state.encoder = ort.InferenceSession(
                model_path,
                providers=cpu_providers,
                provider_options=cpu_provider_options if cpu_provider_options else None,
                sess_options=cpu_opts,
            )
    gc.collect()


def reload_embedding_session(settings: Settings, force_cpu: bool = False):
    """
    Recreate the embedding ONNX session to release DirectML GPU memory.

    Same rationale as reload_encoder_session. When *force_cpu* is True
    the new session is loaded on CPU only.
    """
    from model_state import state
    import gc

    log.info("reloading_embedding_session", force_cpu=force_cpu)

    emb_path = str(ensure_embedding_model(settings))

    old = state.embedding_session
    state.embedding_session = None
    del old
    gc.collect()

    if force_cpu:
        state.embedding_session = ort.InferenceSession(emb_path, providers=["CPUExecutionProvider"])
        log.warning("embedding_reloaded_cpu_fallback")
    else:
        providers, provider_options, opts = get_igpu_session_options(settings.provider_type, settings)
        try:
            state.embedding_session = ort.InferenceSession(
                emb_path,
                sess_options=opts,
                providers=providers,
                provider_options=provider_options if provider_options else None,
            )
            log.info("embedding_session_reloaded", provider=settings.provider_type)
        except Exception as e:
            log.error("embedding_reload_failed", error=str(e), fallback="CPU")
            try:
                state.embedding_session = ort.InferenceSession(emb_path, providers=["CPUExecutionProvider"])
            except Exception as e2:
                log.error("embedding_reload_cpu_fallback_failed", error=str(e2))
    gc.collect()
