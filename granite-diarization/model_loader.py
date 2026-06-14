"""
Model loading for Granite ASR (transformers) and auxiliary ONNX models.
"""

import os
import site
from pathlib import Path

import json
import numpy as np
import onnxruntime as ort
import structlog
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from settings import Settings
from model_state import state

log = structlog.get_logger()


def get_igpu_session_options(provider_type: str = "DirectML", settings: Settings = Settings()):
    opts: ort.SessionOptions = ort.SessionOptions()

    if provider_type.lower() == "openvino":
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.add_session_config_entry("optimization.transformers.onnx_model_type", "encoder_decoder")

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
        ov_config = {"INFERENCE_PRECISION_HINT": "f16"}
        provider_options = [{"device_type": "GPU","load_config": json.dumps(ov_config)},{}]
        log.info("igpu_config_created", provider="OpenVINO")
        return providers, provider_options, opts

    elif provider_type.lower() == "directml":
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
        opts.intra_op_num_threads = settings.cpu_threads if hasattr(settings, 'cpu_threads') else max(1, os.cpu_count() - 1)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return providers, None, opts

    log.warning("igpu_provider_not_recognized", provider=provider_type)
    return ["CPUExecutionProvider"], None, opts


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
    """Load Granite ASR model + Silero VAD + ECAPA-TDNN embedding model."""

    log.info("--- Initialising Granite + Silero VAD + ECAPA-TDNN ---")

    # 1. Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state.device = device
    log.info("device_selected", device=device)

    # 2. Load Granite ASR
    log.info("loading_granite_model", repo=settings.model_repo)
    try:
        processor = AutoProcessor.from_pretrained(settings.model_repo, token=settings.hf_token)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            settings.model_repo,
            device_map=device,
            torch_dtype=dtype,
            token=settings.hf_token,
        )
        model.eval()
        state.processor = processor
        state.model = model
        log.info("granite_model_loaded", device=device, dtype=str(dtype))
    except Exception as e:
        log.error("granite_model_load_failed", error=str(e))
        raise

    # 3. Load VAD ONNX (CPU)
    _, cpu_opts = get_igpu_session_options("CPU", settings)[::2]
    log.info("loading_vad_model")
    try:
        vad_onnx_path = str(ensure_vad_model(settings))
        state.vad_session = ort.InferenceSession(
            vad_onnx_path,
            providers=["CPUExecutionProvider"],
            sess_options=cpu_opts if isinstance(cpu_opts, ort.SessionOptions) else None,
        )
        log.info("vad_model_loaded")
    except Exception as e:
        log.error("vad_load_failed", error=str(e))
        raise

    # 4. Load embedding model (DirectML/CPU)
    log.info("loading_embedding_model")
    PROVIDER_SELECTION = settings.provider_type if hasattr(settings, 'provider_type') else "CPU"
    providers, provider_options, opts = get_igpu_session_options(PROVIDER_SELECTION, settings)
    try:
        emb_path = str(ensure_embedding_model(settings))
        state.embedding_session = ort.InferenceSession(
            emb_path,
            sess_options=opts,
            providers=providers,
            provider_options=provider_options if provider_options else None
        )
        log.info("embedding_model_loaded", provider=PROVIDER_SELECTION)
    except Exception as e:
        log.error("embedding_model_load_failed", error=str(e), fallback="CPU")
        try:
            state.embedding_session = ort.InferenceSession(emb_path, providers=["CPUExecutionProvider"])
        except Exception as e2:
            log.error("embedding_model_cpu_fallback_failed", error=str(e2))

    state.status = "ready"
    log.info("all_models_loaded")
