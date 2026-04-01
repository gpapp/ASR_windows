"""
directml_patch.py
-----------------
Monkeypatches PyTorch and NeMo for DirectML (Intel Iris Xe) compatibility.
"""

import torch

_original_out_of_place = torch.Tensor.masked_scatter
_original_in_place = torch.Tensor.masked_scatter_

def _dml_safe_scatter_logic(self, mask, source, in_place=False):
    """Common logic for CPU-fallback scatter."""
    is_dml = (self.device.type == "privateuseone")
    
    if not is_dml:
        try:
            if in_place:
                return _original_in_place(self, mask, source)
            else:
                return _original_out_of_place(self, mask, source)
        except RuntimeError as e:
            if "Number of elements of source < number of ones in mask" not in str(e):
                raise e

    self_cpu = self.cpu()
    mask_cpu = mask.cpu()
    source_cpu = source.cpu()
    
    n_mask = int(mask_cpu.sum().item())
    n_source = source_cpu.numel()
    
    if n_source < n_mask:
        idx = mask_cpu.nonzero(as_tuple=True)
        new_mask = torch.zeros_like(mask_cpu)
        limited_idx = tuple(i[:n_source] for i in idx)
        new_mask[limited_idx] = True
        mask_to_use = new_mask
    else:
        mask_to_use = mask_cpu

    if in_place:
        _original_in_place(self_cpu, mask_to_use, source_cpu)
        self.copy_(self_cpu.to(self.device))
        return self
    else:
        res_cpu = _original_out_of_place(self_cpu, mask_to_use, source_cpu)
        return res_cpu.to(self.device, dtype=self.dtype)

def _patched_masked_scatter(self, mask, source):
    return _dml_safe_scatter_logic(self, mask, source, in_place=False)

def _patched_masked_scatter_(self, mask, source):
    return _dml_safe_scatter_logic(self, mask, source, in_place=True)

# --- NeMo Preprocessor Patch ---
def apply_nemo_patch():
    try:
        import nemo.collections.asr.modules.audio_preprocessing as ap
        
        preprocessor_classes = [
            ap.AudioToMelSpectrogramPreprocessor,
            ap.AudioToMFCCPreprocessor
        ]
        
        for cls in preprocessor_classes:
            _orig_forward = cls.forward
            
            def make_patched_forward(orig_f):
                def patched_forward(self, input_signal, length):
                    if input_signal.device.type == "privateuseone":
                        # CPU Fallback for STFT/ComplexFloat support
                        sig_cpu = input_signal.cpu()
                        len_cpu = length.cpu()
                        
                        # Ensure buffers are on CPU
                        self.to('cpu') 
                        
                        # NeMo @typecheck requires keywords
                        res, res_len = orig_f(self, input_signal=sig_cpu, length=len_cpu)
                        return res.to(input_signal.device), res_len.to(input_signal.device)
                    return orig_f(self, input_signal=input_signal, length=length)
                return patched_forward
                
            cls.forward = make_patched_forward(_orig_forward)
            
        print(" > NeMo Preprocessors patched for DirectML (CPU fallback).")
    except (ImportError, AttributeError):
        pass

def apply():
    torch.Tensor.masked_scatter = _patched_masked_scatter
    torch.Tensor.masked_scatter_ = _patched_masked_scatter_
    
    if hasattr(torch, 'masked_scatter'):
        torch.masked_scatter = lambda input, mask, source: _patched_masked_scatter(input, mask, source)

    apply_nemo_patch()
    print(" > DirectML patches applied.")
