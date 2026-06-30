import numpy as np
import onnxruntime_genai as og
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from settings import get_settings
from model_loader import load_models
from model_state import state

print("Testing CPU execution...")
settings = get_settings()
settings.provider_type = 'CPU'
load_models(settings)

processor = state.processor
tokenizer = state.tokenizer
model = state.model

# Create 1s of silence
audio = np.zeros(16000, dtype=np.float32)
inputs = processor.process(audio)

params = og.GeneratorParams(model)
generator = og.Generator(model, params)
generator.set_inputs(inputs)

print("Running generate_next_token()...")
generator.generate_next_token()
print("SUCCESS: Generated next token on CPU")
