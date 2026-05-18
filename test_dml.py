"""
DML Test Script
Tests torch_directml functionality on this Windows system.
"""

import sys
import os

results = []

def test(name, func):
    """Run a test and record the result."""
    try:
        result = func()
        results.append((name, "PASS", result))
        print(f"[PASS] {name}: {result}")
    except Exception as e:
        results.append((name, "FAIL", str(e)))
        print(f"[FAIL] {name}: {e}")

# Test 1: Import torch_directml
def test_import_directml():
    import torch_directml
    import torch
    return f"torch: {torch.__version__}, directml available"

# Test 2: Get DML device
def test_get_dml_device():
    import torch_directml
    dml = torch_directml.device()
    return f"Device: {dml}, Type: {dml.type}"

# Test 3: Basic tensor ops on DML
def test_basic_tensor_ops():
    import torch
    import torch_directml
    dml = torch_directml.device()
    
    t = torch.randn(100, 80).to(dml)
    t2 = t * 2
    t3 = t + t2
    t4 = t3.mean(dim=1)
    return f"Created tensor shape: {t.shape}, device: {t.device}"

# Test 4: CMN (Cepstral Mean Normalization) on DML
def test_cmn_operation():
    import torch
    import torch_directml
    dml = torch_directml.device()
    
    x = torch.randn(10, 80).to(dml)
    x_cmn = x - x.mean(dim=1, keepdim=True)
    return f"Input shape: {x.shape}, CMN result shape: {x_cmn.shape}, device: {x_cmn.device}"

# Test 5: torchaudio kaldi fbank (CPU -> DML)
def test_kaldi_fbank():
    import torch
    import torch_directml
    import torchaudio
    
    dml = torch_directml.device()
    waveform = torch.randn(1, 16000)
    
    specs = torchaudio.compliance.kaldi.fbank(waveform)
    specs_dml = specs.to(dml)
    return f"fbank: CPU->DML shape={specs_dml.shape}, device={specs_dml.device}"

# Test 6: STFT on DML (with return_complex=True)
def test_stft_dml():
    import torch
    import torch_directml
    
    dml = torch_directml.device()
    x = torch.randn(1, 16000).to(dml)
    window = torch.hann_window(400).to(dml)
    
    # PyTorch 2.4+ requires return_complex for real input
    stft_result = torch.stft(x, n_fft=400, hop_length=160, win_length=400, 
                             window=window, return_complex=True)
    return f"STFT shape: {stft_result.shape}, device: {stft_result.device}"

# Test 7: Matrix operations (bmm, transpose)
def test_matrix_ops():
    import torch
    import torch_directml
    
    dml = torch_directml.device()
    a = torch.randn(32, 80, 256).to(dml)
    b = torch.randn(32, 256, 80).to(dml)
    c = torch.bmm(a, b)
    d = c.transpose(1, 2)
    return f"bmm: {c.shape}, transpose: {d.shape}, device: {d.device}"

# Test 8: Softmax and activation functions
def test_activation_functions():
    import torch
    import torch_directml
    
    dml = torch_directml.device()
    x = torch.randn(32, 100).to(dml)
    
    # Test common activations
    relu_out = torch.relu(x)
    softmax_out = torch.softmax(x, dim=1)
    sigmoid_out = torch.sigmoid(x)
    
    return f"relu: {relu_out.shape}, softmax: {softmax_out.shape}, sigmoid: {sigmoid_out.shape}, device: {x.device}"

# Test 9: Load directml_patch
def test_directml_patch():
    import sys
    sys.path.insert(0, r"C:\Users\Gergely_Papp\source\ASR\nemotron_dml")
    import directml_patch
    directml_patch.apply()
    return "directml_patch loaded and applied"

# Test 10: Masked scatter (the operation that DML struggles with)
def test_masked_scatter():
    import torch
    import torch_directml
    
    dml = torch_directml.device()
    x = torch.zeros(10, 80).to(dml)
    mask = torch.tensor([[True, False] * 40]).repeat(10, 1).to(dml)
    source = torch.randn(10, 80).to(dml)
    
    try:
        result = torch.masked_scatter(x, mask, source)
        return f"masked_scatter works: {result.shape}, device: {result.device}"
    except Exception as e:
        return f"masked_scatter failed: {str(e)[:60]}"

# Run tests
print("=" * 60)
print("Testing torch_directml on Windows")
print("=" * 60)

test("import torch_directml", test_import_directml)
test("get DML device", test_get_dml_device)
test("basic tensor ops", test_basic_tensor_ops)
test("CMN operation (x - x.mean)", test_cmn_operation)
test("kaldi fbank (CPU -> DML)", test_kaldi_fbank)
test("STFT on DML", test_stft_dml)
test("matrix ops (bmm, transpose)", test_matrix_ops)
test("activation functions", test_activation_functions)
test("directml_patch apply", test_directml_patch)
test("masked_scatter", test_masked_scatter)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

passed = sum(1 for _, status, _ in results if status == "PASS")
failed = sum(1 for _, status, _ in results if status == "FAIL")

for name, status, detail in results:
    print(f"  {status:4s} | {name}")
    if status == "FAIL":
        print(f"         -> {detail}")

print(f"\nTotal: {passed} passed, {failed} failed")

if failed > 0:
    sys.exit(1)
