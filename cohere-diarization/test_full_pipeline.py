#!/usr/bin/env python3
"""
Integration test: Full transcription pipeline with real models.
"""

import sys
import os
import numpy as np
import librosa
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent))

def test_full_pipeline():
    """Test the complete transcription pipeline."""
    print("\n" + "=" * 60)
    print("Full Transcription Pipeline Test")
    print("=" * 60)
    
    try:
        # Import after adding to path
        from model_loader import load_models
        from settings import get_settings
        from model_state import state
        from transcriber import transcribe_audio_sync
        
        print("\n1. Initializing models...")
        settings = get_settings()
        load_models(settings)
        print("✓ Models loaded")
        
        print("\n2. Generating test audio (440Hz tone for 2 seconds)...")
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        frequency = 440
        audio_signal = np.sin(2 * np.pi * frequency * t) * 10000
        audio_signal = audio_signal.astype(np.int16)
        print(f"✓ Test audio: {len(audio_signal)} samples ({duration}s)")
        
        print("\n3. Running transcription...")
        result = transcribe_audio_sync(audio_signal, language="en", timeout_sec=300)
        
        print("✓ Transcription complete!")
        print(f"\nResults:")
        print(f"  Text: {result['text']}")
        print(f"  Tokens generated: {result['tokens_generated']}")
        print(f"  Audio duration: {result['audio_duration_sec']:.2f}s")
        print(f"  Inference time: {result['inference_time_sec']:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
