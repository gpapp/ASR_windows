"""
Model loading and initialization for Nemotron ASR model using ONNX Runtime GenAI.
"""

import os
import site
from pathlib import Path

import json
import numpy as np
import onnxruntime as ort
import onnxruntime_genai as og
import structlog

from settings import Settings
from model_state import state

log = structlog.get_logger()


LANG_TO_ID = {
    "en":     0, "en-US":  0, "en-GB":  1,
    "es-ES":  2, "es":     3, "es-US":  3,
    "zh-CN":  4,
    "hi":     6, "hi-IN":  6,
    "ar":     7, "ar-AR":  7,
    "fr":     8, "fr-FR":  8,
    "de":     9, "de-DE":  9,
    "ja":    10, "ja-JP": 10,
    "ru":    11, "ru-RU": 11,
    "pt-BR": 12, "pt":    13, "pt-PT": 13,
    "ko":    14, "ko-KR": 14,
    "it":    15, "it-IT": 15,
    "nl":    16, "nl-NL": 16,
    "pl":    17, "pl-PL": 17,
    "tr":    18, "tr-TR": 18,
    "uk":    19, "uk-UA": 19,
    "ro":    20, "ro-RO": 20,
    "el":    21, "el-GR": 21,
    "cs":    22, "cs-CZ": 22,
    "hu":    23, "hu-HU": 23,
    "sv":    24, "sv-SE": 24,
    "da":    25, "da-DK": 25,
    "fi":    26, "fi-FI": 26,
    "sk":    28, "sk-SK": 28,
    "hr":    29, "hr-HR": 29,
    "bg":    30, "bg-BG": 30,
    "lt":    31, "lt-LT": 31,
    "th":    32, "th-TH": 32,
    "vi":    33, "vi-VN": 33,
    "et":    60, "et-EE": 60,
    "lv":    61, "lv-LV": 61,
    "sl":    62, "sl-SI": 62,
    "he":    64, "he-IL": 64,
    "fr-CA": 100,
    "auto":  101,
    "mt":    102, "mt-MT": 102,
    "nb":    103, "nb-NO": 103,
    "nn":    104, "nn-NO": 104,
}


def get_igpu_session_options(provider_type: str = "DirectML", settings: Settings = Settings()):
    opts: ort.SessionOptions = ort.SessionOptions()

    if provider_type.lower() == "directml":
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_mem_pattern = False
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        log.info("igpu_config_created", provider="DirectML")
        return providers, None, opts
    elif provider_type.lower() == "cpu":
        providers = ["CPUExecutionProvider"]
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_mem_pattern = False
        opts.intra_op_num_threads = settings.cpu_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return providers, None, opts

    log.warning("igpu_provider_not_recognized", provider=provider_type)
    return ["CPUExecutionProvider"], None, opts


def ensure_model(settings: Settings) -> Path:
    """Download model files if not present."""
    needed = [
        "genai_config.json",
        "encoder.onnx",
        "decoder.onnx",
        "joint.onnx",
        "tokenizer.json",
    ]

    if all((settings.model_dir / f).exists() for f in needed):
        log.info("model_files_present", path=str(settings.model_dir))
        return settings.model_dir

    log.info("downloading_model", repo=settings.model_repo)
    settings.model_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=settings.model_repo,
        local_dir=str(settings.model_dir),
        token=settings.hf_token,
    )

    log.info("model_download_complete")
    return settings.model_dir


def ensure_vad_model(settings: Settings) -> Path:
    filename = f"onnx/model{settings.vad_model_type}.onnx"
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
    """Load Nemotron ASR model via ONNX Runtime GenAI."""

    model_dir = ensure_model(settings)

    config = og.Config(str(model_dir))
    ep = settings.provider_type.lower()
    
    # Check for DirectML hybrid fallback
    is_dml = (ep == "directml")
    
    if ep != "follow_config":
        config.clear_providers()
        if is_dml:
            # We want to use DML where stable. 
            # In onnxruntime-genai, append_provider("dml") applies to the whole model group.
            # If the encoder fails on DML, we have to decide if we run the WHOLE Model on CPU
            # or try to force hybrid. 
            # Note: og.Model currently doesn't allow per-component provider overrides.
            # If the Encoder is the only blocker, and we want DML for Decoder,
            # we must use CPU for the whole Model if og.Model is used, 
            # OR wrap it ourselves.
            
            # For now, let's try appending DML and see if we can use a workaround.
            config.append_provider("dml")
        elif ep == "cpu":
            pass # Default is CPU

    log.info("loading_nemotron_model", provider=settings.provider_type)
    model = og.Model(config)
    
    if is_dml:
        try:
            # Proactively verify DML works with a dummy inference step
            # This catches 80070057 which happens during first execution
            dummy_audio = np.zeros(settings.nemotron_chunk_samples, dtype=np.float32)
            processor = og.StreamingProcessor(model)
            processor.set_option("use_vad", "false")
            inputs = processor.process(dummy_audio)
            
            params = og.GeneratorParams(model)
            generator = og.Generator(model, params)
            generator.set_inputs(inputs)
            generator.generate_next_token()
            log.info("nemotron_dml_verified_success")
        except Exception as e:
            if "DmlFusedNode" in str(e) or "80070057" in str(e):
                log.warning("nemotron_dml_verification_failed_falling_back_to_full_cpu", error=str(e))
                # Force settings to CPU so VAD and Embeddings follow suit
                settings.provider_type = "CPU"
                is_dml = False
                config.clear_providers()
                model = og.Model(config)
            else:
                raise e

    state.model = model

    processor = og.StreamingProcessor(model)
    processor.set_option("use_vad", "false")
    state.processor = processor

    tokenizer = og.Tokenizer(model)
    state.tokenizer = tokenizer

    log.info("nemotron_model_loaded", model_type=model.type)

    # VAD model
    log.info("loading_vad_model")
    cpu_providers, cpu_provider_options, cpu_opts = get_igpu_session_options("CPU", settings)
    try:
        vad_onnx_path = str(ensure_vad_model(settings))
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

    # Embedding model
    log.info("loading_embedding_model")
    providers, provider_options, opts = get_igpu_session_options(settings.provider_type, settings)
    try:
        emb_path = str(ensure_embedding_model(settings))
        state.embedding_session = ort.InferenceSession(
            emb_path,
            sess_options=opts,
            providers=providers,
            provider_options=provider_options if provider_options else None
        )
        log.info("embedding_model_loaded", path=emb_path, provider=settings.provider_type)
    except Exception as e:
        log.error("embedding_model_loading_failed", error=str(e), fallback="CPU")
        try:
            state.embedding_session = ort.InferenceSession(emb_path, providers=["CPUExecutionProvider"])
        except Exception as e2:
            log.error("embedding_model_cpu_fallback_failed", error=str(e2))

    state.status = "ready"
    log.info("all_models_loaded", provider=settings.provider_type)


def reload_embedding_session(settings: Settings, force_cpu: bool = False):
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
