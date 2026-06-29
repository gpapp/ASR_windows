"""Quick test: load models and verify ONNX readiness."""
import sys; sys.path.insert(0, '.')
from model_loader import load_models
from settings import get_settings
from model_state import state

s = get_settings()
s.onnx_model_id = 'smcleod/ibm-granite-speech-4.1-2b-plus-onnx'
s.onnx_precision = 'int8'
load_models(s)

print('onnx_ready:', state.onnx_ready)
print('tokenizer:', type(state.tokenizer).__name__)
print('status:', state.status)
print('encoder:', 'OK' if state.encoder_session else 'MISSING')
print('embed_tokens:', 'OK' if state.embed_tokens_session else 'MISSING')
print('prompt_encode:', 'OK' if state.prompt_encode_session else 'MISSING')
print('decode_step:', 'OK' if state.decode_step_session else 'MISSING')
