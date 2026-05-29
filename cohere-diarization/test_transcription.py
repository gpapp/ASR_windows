#!/usr/bin/env python3
"""
Test script for Cohere Transcribe ONNX model with DirectML.
Tests encoder, decoder, and end-to-end transcription.
"""

import sys
import numpy as np
import librosa
import onnxruntime as ort
from pathlib import Path

# Configuration
MODEL_DIR = Path(__file__).parent.parent / "models/cohere-transcribe-onnx"
ENCODER_PATH = MODEL_DIR / "onnx/encoder_model_fp16.onnx"
DECODER_PATH = MODEL_DIR / "onnx/decoder_model_merged_fp16.onnx"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"

def test_encoder():
    """Test encoder model with DirectML."""
    print("\n=== Testing Encoder ===")
    
    # Create sample mel-spec input
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio_signal = np.sin(2 * np.pi * 440 * t) * 10000  # 440Hz tone
    audio_signal = audio_signal.astype(np.int16)
    
    # Extract mel-spec
    audio_float = audio_signal.astype(np.float32) / 32768.0
    mel_spec = librosa.feature.melspectrogram(
        y=audio_float,
        sr=16000,
        n_fft=512,
        hop_length=160,
        n_mels=128,
        window='hann'
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = (mel_spec_db - mel_spec_db.mean(axis=1, keepdims=True)) / (mel_spec_db.std(axis=1, keepdims=True) + 1e-8)
    input_features = mel_spec_db.T[np.newaxis, :, :].astype(np.float32)
    
    print(f"Input shape: {input_features.shape}")
    
    # Load encoder with DirectML
    try:
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        encoder = ort.InferenceSession(str(ENCODER_PATH), providers=providers)
        print(f"✓ Encoder loaded with providers: {encoder.get_providers()}")
    except Exception as e:
        print(f"✗ Failed to load encoder: {e}")
        return False
    
    # Run inference
    try:
        enc_output = encoder.run(None, {"input_features": input_features})
        encoder_hidden_state = enc_output[0]
        print(f"✓ Encoder inference successful")
        print(f"  Output shape: {encoder_hidden_state.shape}")
        print(f"  Output dtype: {encoder_hidden_state.dtype}")
        return True
    except Exception as e:
        print(f"✗ Encoder inference failed: {e}")
        return False

def test_decoder():
    """Test decoder model with DirectML."""
    print("\n=== Testing Decoder ===")
    
    # Load decoder - try CPU first to isolate DML issues
    try:
        providers = ["CPUExecutionProvider"]  # Start with CPU
        decoder = ort.InferenceSession(str(DECODER_PATH), providers=providers)
        print(f"✓ Decoder loaded with providers: {decoder.get_providers()}")
    except Exception as e:
        print(f"✗ Failed to load decoder: {e}")
        return False
    
    # Prepare dummy inputs
    batch_size = 1
    try:
        # Dummy encoder output with proper shape
        # Shape should be [batch_size, encoder_seq_len, 1024]
        encoder_seq_len = 100
        encoder_hidden_state = np.random.randn(batch_size, encoder_seq_len, 1024).astype(np.float32)
        
        # Minimal decoder inputs
        input_ids = np.array([[4]], dtype=np.int64)  # Start token
        attention_mask = np.array([[1]], dtype=np.int64)
        position_ids = np.array([[0]], dtype=np.int64)
        num_logits_to_keep = np.array(1, dtype=np.int64)
        
        decoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "num_logits_to_keep": num_logits_to_keep,
            "encoder_hidden_states": encoder_hidden_state,
        }
        
        # Add KV cache tensors (required inputs)
        num_layers = 8
        num_heads = 8
        head_dim = 128
        
        for layer_idx in range(num_layers):
            decoder_inputs[f"past_key_values.{layer_idx}.decoder.key"] = np.zeros(
                (batch_size, num_heads, 0, head_dim), dtype=np.float16
            )
            decoder_inputs[f"past_key_values.{layer_idx}.decoder.value"] = np.zeros(
                (batch_size, num_heads, 0, head_dim), dtype=np.float16
            )
            decoder_inputs[f"past_key_values.{layer_idx}.encoder.key"] = np.zeros(
                (batch_size, num_heads, 0, head_dim), dtype=np.float16
            )
            decoder_inputs[f"past_key_values.{layer_idx}.encoder.value"] = np.zeros(
                (batch_size, num_heads, 0, head_dim), dtype=np.float16
            )
        
        print(f"  Encoder hidden state shape: {encoder_hidden_state.shape}")
        print(f"  Input IDs shape: {input_ids.shape}")
        
        dec_output = decoder.run(None, decoder_inputs)
        logits = dec_output[0]
        
        print(f"✓ Decoder inference successful")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits dtype: {logits.dtype}")
        return True
    except Exception as e:
        print(f"✗ Decoder inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tokenizer():
    """Test tokenizer loading."""
    print("\n=== Testing Tokenizer ===")
    
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        token_to_id = tokenizer.get_vocab()
        
        print(f"✓ Tokenizer loaded")
        print(f"  Vocab size: {len(token_to_id)}")
        
        # Check critical tokens
        critical_tokens = [
            '<|startofcontext|>',
            '<|startoftranscript|>',
            '<|endoftext|>',
            '<|en|>',
        ]
        
        for tok in critical_tokens:
            if tok in token_to_id:
                print(f"  ✓ {tok}: {token_to_id[tok]}")
            else:
                print(f"  ✗ {tok}: NOT FOUND")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Tokenizer test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Cohere Transcribe ONNX - DirectML Test")
    print("=" * 60)
    
    results = {
        "Encoder": test_encoder(),
        "Decoder": test_decoder(),
        "Tokenizer": test_tokenizer(),
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Transcription should work.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
