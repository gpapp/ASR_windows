import numpy as np
import structlog

from model_state import state

log = structlog.get_logger()

NEG_INF = np.finfo(np.float32).min
AUDIO_PLACEHOLDER_ID = 100352


def _build_causal_mask(N: int) -> np.ndarray:
    return np.triu(np.full((1, 1, N, N), NEG_INF, dtype=np.float32), k=1)


def _make_position_ids(N: int) -> np.ndarray:
    return np.arange(N, dtype=np.int64).reshape(1, N)


def _prepare_inputs(audio: np.ndarray, prompt_text: str) -> tuple:
    """
    Use the GraniteSpeechProcessor to prepare model inputs.
    The processor tokenizes the prompt (replicating <|audio|> tokens one per
    audio embedding frame) and extracts the audio features.
    Returns (combined_embeds, input_ids) ready for prompt_encode.
    For text-only prompts (audio=None), returns the text-only embedding.
    """
    proc_out = state.processor(
        text=prompt_text,
        audio=audio,
        return_tensors="pt",
        sampling_rate=16000,
    )

    input_ids = proc_out["input_ids"].numpy()
    text_embeds = state.embed_tokens_session.run(
        None, {"input_ids": input_ids}
    )[0]

    audio_positions = np.where(input_ids[0] == AUDIO_PLACEHOLDER_ID)[0]

    if audio is not None and len(audio_positions) > 0:
        input_features = proc_out["input_features"]
        if hasattr(input_features, "numpy"):
            input_features = input_features.numpy()

        # Encode audio features
        audio_embeds, audio_embed_sizes = state.encoder_session.run(
            None, {"input_features": input_features}
        )

        # Replace each <|audio|> token embedding with the corresponding
        # audio embedding frame (one-to-one, in order of appearance)
        if len(audio_positions) != audio_embeds.shape[1]:
            log.warning("audio_token_count_mismatch",
                n_tokens=len(audio_positions),
                n_frames=audio_embeds.shape[1])
        n = min(len(audio_positions), audio_embeds.shape[1])
        for i in range(n):
            text_embeds[0, audio_positions[i]] = audio_embeds[0, i]

    return text_embeds, input_ids


def _transcribe_onnx(
    audio: np.ndarray,
    input_ids: np.ndarray,
    combined: np.ndarray,
    max_new_tokens: int = 400,
) -> str:
    prompt_enc = state.prompt_encode_session
    dec_step = state.decode_step_session
    emb_tok = state.embed_tokens_session
    tokenizer = state.tokenizer
    eos_id = tokenizer.eos_token_id or 100257

    N = combined.shape[1]
    position_ids = _make_position_ids(N)
    attention_mask = _build_causal_mask(N)

    prompt_outputs = prompt_enc.run(None, {
        "inputs_embeds": combined,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    })
    logits = prompt_outputs[0]
    kv_cache = prompt_outputs[1:]

    next_token = int(np.argmax(logits[0, -1, :]))
    generated = [next_token]

    T_past = N

    for _ in range(max_new_tokens - 1):
        if next_token == eos_id:
            break

        token_input = np.array([[next_token]], dtype=np.int64)
        token_emb = emb_tok.run(None, {"input_ids": token_input})[0]

        new_pos = np.array([[T_past]], dtype=np.int64)
        dec_mask = np.zeros((1, 1, 1, T_past + 1), dtype=np.float32)

        feed_dict = {
            "inputs_embeds": token_emb,
            "position_ids": new_pos,
            "attention_mask": dec_mask,
        }
        for i in range(40):
            feed_dict[f"past_key_values.{i}.key"] = kv_cache[i * 2]
            feed_dict[f"past_key_values.{i}.value"] = kv_cache[i * 2 + 1]

        dec_outputs = dec_step.run(None, feed_dict)
        logits = dec_outputs[0]
        kv_cache = dec_outputs[1:]

        next_token = int(np.argmax(logits[0, 0, :]))
        generated.append(next_token)
        T_past += 1

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text


def transcribe_audio(
    audio: np.ndarray,
    prompt_text: str,
    max_new_tokens: int = 400,
) -> str:
    combined, input_ids = _prepare_inputs(audio, prompt_text)
    return _transcribe_onnx(audio, input_ids, combined, max_new_tokens)



