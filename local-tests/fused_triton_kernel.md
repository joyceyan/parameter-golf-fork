# Task: Replace the per-op QAT fake-quantize in CastedLinear with a fused Triton kernel

## Context

In `train_gpt_submit.py`, the `CastedLinear.forward` method has a QAT (quantization-aware training) path that simulates int6 quantization during training using the straight-through estimator (STE). When `CastedLinear._qat_enabled` is True and the module is in training mode, it runs ~10 individual PyTorch ops per linear layer:

```python
if CastedLinear._qat_enabled and self.training and w.ndim == 2:
    with torch.no_grad():
        w32 = self.weight.float()
        row_max = w32.abs().amax(dim=1)
        scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
        w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
    w = w + (w_q - w).detach()  # STE
```

With ~67 linear layers in the model, this launches hundreds of small CUDA kernels per forward pass, causing ~28% overhead (84ms vs 66ms per step). The goal is to replace this with a single fused Triton kernel that does the entire quantize→dequantize→STE operation in one read/write pass per weight matrix.

## What the kernel needs to do

For a 2D weight matrix of shape `(out_features, in_features)`:

1. Read each element `w[i, j]` as float32
2. Compute `row_max[i] = max(|w[i, :]|)` (reduction along dim=1)
3. Compute `scale[i] = max(row_max[i] / 31.0, 1.0 / 31.0)`
4. Compute `w_q[i, j] = clamp(round(w[i, j] / scale[i]), -32, 31) * scale[i]`
5. Write `w_q[i, j]` in the output dtype (bfloat16, matching `x.dtype`)

The STE part (`w + (w_q - w).detach()`) happens outside the kernel in normal PyTorch — the kernel just needs to produce `w_q`.

## Implementation plan

### 1. Create a new file `triton_qat.py` in the same directory as `train_gpt_submit.py`

This file should contain:

```python
import triton
import triton.language as tl
import torch
from torch import Tensor


@triton.jit
def _fused_int6_qat_kernel(
    W_ptr,          # input weight matrix, float32
    OUT_ptr,        # output fake-quantized weight, bf16
    M,              # number of rows (out_features)
    N,              # number of cols (in_features)
    stride_w_m,     # stride along rows for W
    stride_w_n,     # stride along cols for W
    stride_o_m,     # stride along rows for OUT
    stride_o_n,     # stride along cols for OUT
    BLOCK_N: tl.constexpr,  # tile width — number of columns processed per program
):
    # Each program handles one full row
    row_idx = tl.program_id(0)
    
    # Phase 1: compute row_max via tiled reduction
    row_max = tl.zeros([], dtype=tl.float32)
    for col_start in range(0, N, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        w = tl.load(W_ptr + row_idx * stride_w_m + col_offsets * stride_w_n, mask=mask, other=0.0)
        w_abs = tl.abs(w)
        row_max = tl.maximum(row_max, tl.max(w_abs, axis=0))
    
    # Phase 2: compute scale
    scale = tl.maximum(row_max / 31.0, 1.0 / 31.0)
    inv_scale = 1.0 / scale
    
    # Phase 3: quantize-dequantize, write output
    for col_start in range(0, N, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        w = tl.load(W_ptr + row_idx * stride_w_m + col_offsets * stride_w_n, mask=mask, other=0.0)
        
        # round-to-nearest, clamp to [-32, 31], dequantize
        q = tl.math.round(w * inv_scale)  # use multiplication instead of division
        q = tl.minimum(tl.maximum(q, -32.0), 31.0)
        w_dq = q * scale
        
        # Write as bf16
        tl.store(OUT_ptr + row_idx * stride_o_m + col_offsets * stride_o_n,
                 w_dq.to(tl.bfloat16), mask=mask)


def fused_int6_fake_quantize(weight: Tensor) -> Tensor:
    """
    Fused int6 per-row fake quantize.
    
    Input: weight matrix (M, N) in float32
    Output: fake-quantized weight (M, N) in bfloat16
    
    Equivalent to:
        row_max = weight.float().abs().amax(dim=1)
        scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
        w_q = (clamp(round(weight / scale[:, None]), -32, 31) * scale[:, None]).bfloat16()
    """
    assert weight.ndim == 2, "Only 2D weight matrices supported"
    assert weight.is_cuda, "Weight must be on CUDA"
    
    M, N = weight.shape
    # Ensure input is float32 for the kernel
    w_f32 = weight.float() if weight.dtype != torch.float32 else weight
    
    out = torch.empty(M, N, dtype=torch.bfloat16, device=weight.device)
    
    # BLOCK_N: tile width for column processing. 
    # Must be power of 2. 1024 is a good default for typical N values (512, 1536).
    # For N <= 1024, one pass through the row. For N=1536, two passes.
    BLOCK_N = min(triton.next_power_of_2(N), 1024)
    
    grid = (M,)
    _fused_int6_qat_kernel[grid](
        w_f32, out,
        M, N,
        w_f32.stride(0), w_f32.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=BLOCK_N,
    )
    return out
```

### 2. Modify `CastedLinear.forward` in `train_gpt_submit.py`

Add the import at the top of the file, guarded so it doesn't break if Triton isn't available:

```python
try:
    from triton_qat import fused_int6_fake_quantize
    _TRITON_QAT_AVAILABLE = True
except ImportError:
    _TRITON_QAT_AVAILABLE = False
```

Then replace the QAT block in `CastedLinear.forward`:

```python
class CastedLinear(nn.Linear):
    _qat_enabled: bool = False

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            if _TRITON_QAT_AVAILABLE:
                # Fused Triton kernel: single read/write pass per matrix
                w_q = fused_int6_fake_quantize(self.weight)
            else:
                # Fallback: original per-op implementation
                with torch.no_grad():
                    w32 = self.weight.float()
                    row_max = w32.abs().amax(dim=1)
                    scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                    w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()  # STE: gradient flows through w, not w_q
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)
```

## Important details

### The STE (straight-through estimator) stays in PyTorch

The `w = w + (w_q - w).detach()` line must remain as-is in PyTorch. This is the trick that makes QAT work with backpropagation: in the forward pass, the effective weight is `w_q` (quantized), but in the backward pass, gradients flow through to the original `w` (because `(w_q - w).detach()` has zero gradient). The Triton kernel only needs to produce `w_q`.

### Output dtype must be bfloat16

The kernel writes bfloat16 output because `x.dtype` is bfloat16 during the autocast forward pass. The original code does `.to(x.dtype)` at the end of the quantize-dequantize block. The Triton kernel bakes this cast into the store operation to avoid an extra kernel launch.

### Input is always float32

`self.weight` is stored in float32 (the `CastedLinear` constructor and `restore_low_dim_params_to_fp32` ensure this). The kernel reads float32, does all arithmetic in float32, and writes bfloat16.

### Kernel launch overhead

With ~67 linear layers, we're launching 67 Triton kernels per forward pass instead of ~670 PyTorch kernels (67 layers × 10 ops each). This alone should be a significant win. Each kernel processes one weight matrix with M program instances (one per row), where M ranges from 256 to 1536 depending on the layer.

### BLOCK_N tuning

The column tile width `BLOCK_N` controls how much of each row is processed per loop iteration inside the kernel. For the model's typical matrix sizes:
- `c_q.weight`: 512 × 512 → BLOCK_N=512, single pass
- `c_k.weight`: 256 × 512 → BLOCK_N=512, single pass  
- `fc.weight`: 1536 × 512 → BLOCK_N=512, single pass
- `proj.weight` (MLP): 512 × 1536 → BLOCK_N=1024, two passes

`triton.next_power_of_2(N)` capped at 1024 should handle all cases well. If profiling shows the kernel is register-bound on larger matrices, try reducing to 512.

### Memory

The kernel allocates one output tensor `(M, N)` in bf16. This is the same memory the original code allocates for `w_q`. No additional memory overhead.

## Testing checklist

1. **Numerical correctness**: Write a test that generates a random (512, 512) float32 matrix, runs both the original PyTorch QAT code and the Triton kernel, and verifies the bf16 outputs match exactly (not approximately — the operations are deterministic, so they should be bit-identical).

2. **Gradient flow**: Verify that with the Triton kernel, gradients still flow through to `self.weight`. Run a small forward+backward pass with QAT enabled and check that `self.weight.grad` is not None and not all zeros.

3. **Edge cases**: Test with the actual matrix sizes in the model: (512, 512), (256, 512), (1536, 512), (512, 1536), (1024, 128). Also test with a matrix where one row is all zeros (scale should clamp to 1/31).

4. **Performance**: Time 100 forward passes of the full model with QAT enabled, comparing Triton vs original. Target: at least 40% reduction in the QAT overhead (from ~18ms overhead down to ~10ms or less). Measure with `torch.cuda.Event` for accuracy.

5. **Fallback**: Verify that setting `_TRITON_QAT_AVAILABLE = False` falls back to the original implementation with identical results.

## What NOT to change

- Do NOT modify the STE mechanism (`w + (w_q - w).detach()`)
- Do NOT change the quantization parameters (31, -32..31 range, per-row scaling)
- Do NOT change the `_qat_enabled` class variable or the conditions that gate QAT
- Do NOT fuse the `F.linear(x, w, bias)` matmul into the kernel — just produce `w_q`
- Do NOT change any other part of the training script
