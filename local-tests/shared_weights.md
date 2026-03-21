# Task: Add paired weight sharing with wider dim and per-iteration conditioning to train_gpt_submit.py

## Overview

Modify the model architecture to use **paired weight sharing** — adjacent layers share the same Block weight matrices but keep unique scalar parameters. This saves ~4 blocks worth of artifact space, which we spend on a **wider model** (dim=576 instead of 512). Additionally, add **per-iteration conditioning embeddings** so the shared block can behave differently at each position.

The goal is 10 effective layers of depth with only 5 unique blocks stored in the artifact, plus a wider model that nobody else can fit in 16MB without sharing.

## Architecture changes

### Current architecture (9 unique blocks)

```
blocks[0] → blocks[1] → blocks[2] → blocks[3]  (encoder, 4 layers)
blocks[4] → blocks[5] → blocks[6] → blocks[7] → blocks[8]  (decoder, 5 layers)
```

9 unique Block instances. Each has its own weight matrices AND scalar params.

### New architecture (5 unique blocks, 10 effective layers)

```
Effective layer:  0    1    2    3    4    5    6    7    8    9
Unique block:     A    A    B    B    C    C    D    D    E    E
Role:             encoder (5 layers)  |  decoder (5 layers)
Skip connections: save save save save save  ←  ←   ←   ←   ←
```

5 unique Block instances, each used twice. 10 effective layers total (5 encoder + 5 decoder). Each effective layer has its OWN scalar parameters (`attn_scale`, `mlp_scale`, `resid_mix`) and its own iteration conditioning embedding, but shares the heavy weight matrices (`c_q`, `c_k`, `c_v`, `proj`, `fc`, `mlp.proj`) with its partner.

### Key design decisions

**Sharing is within encoder and within decoder, never across the boundary.** Block A is used for encoder layers 0 and 1. Block C is the first decoder block. This keeps the U-Net skip structure clean.

**Scalar parameters are unique per effective layer, not per unique block.** The `attn_scale`, `mlp_scale`, and `resid_mix` are small (512-dim vectors at most) and cost almost nothing in the artifact. Keeping them unique per position lets each iteration of a shared block scale its contributions differently. These must be stored as separate parameters, not inside the Block class.

**Skip connections follow effective layer indices.** With 5 encoder and 5 decoder layers, every decoder layer gets a skip from its matching encoder layer (encoder 4 feeds decoder 5, encoder 3 feeds decoder 6, etc.). This is a cleaner 5/5 split compared to the original 4/5.

## Implementation plan

### 1. Add new hyperparameters

```python
# In Hyperparameters class:
share_pairs = bool(int(os.environ.get("SHARE_PAIRS", "0")))  # Enable paired weight sharing
effective_layers = int(os.environ.get("EFFECTIVE_LAYERS", 0))  # 0 = use num_layers as-is
```

When `share_pairs=True` and `effective_layers=10`, the model creates 5 unique blocks and runs each twice.

### 2. Restructure GPT.__init__

When `share_pairs` is enabled:

```python
if self.share_pairs and effective_layers > 0:
    num_unique_blocks = effective_layers // 2
    self.effective_layers = effective_layers
    # Create only the unique blocks
    self.blocks = nn.ModuleList([
        Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
        for _ in range(num_unique_blocks)
    ])
    # Map from effective layer index to unique block index
    # [0,0,1,1,2,2,3,3,4,4] for 10 effective layers
    self.layer_to_block = [i // 2 for i in range(effective_layers)]
    
    # Per-effective-layer scalar parameters (NOT inside Block)
    self.layer_attn_scales = nn.ParameterList([
        nn.Parameter(torch.ones(model_dim, dtype=torch.float32))
        for _ in range(effective_layers)
    ])
    self.layer_mlp_scales = nn.ParameterList([
        nn.Parameter(torch.ones(model_dim, dtype=torch.float32))
        for _ in range(effective_layers)
    ])
    self.layer_resid_mixes = nn.ParameterList([
        nn.Parameter(torch.stack((torch.ones(model_dim), torch.zeros(model_dim))).float())
        for _ in range(effective_layers)
    ])
    
    # Per-iteration conditioning embeddings
    self.iter_embeddings = nn.Parameter(
        torch.zeros(effective_layers, model_dim, dtype=torch.float32)
    )
    
    # Skip connection weights (5 encoder, 5 decoder = 5 skip weights)
    self.num_encoder_layers = effective_layers // 2
    self.num_decoder_layers = effective_layers - self.num_encoder_layers
    self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
    self.skip_weights = nn.Parameter(
        torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32)
    )
else:
    # Original architecture, unchanged
    ...
```

**Important:** When `share_pairs` is enabled, the Block class should NOT contain `attn_scale`, `mlp_scale`, or `resid_mix` — those are moved to the per-layer ParameterLists on GPT. The Block should only contain the heavy weight matrices (attention projections + MLP). The cleanest way to handle this is to modify Block to accept these as arguments to `forward()` rather than storing them as attributes, OR create a separate `SharedBlock` class that only has `attn_norm`, `mlp_norm`, `attn`, and `mlp` (no scalar params).

I'd recommend the separate class approach to avoid breaking the non-sharing codepath:

```python
class SharedBlock(nn.Module):
    """Block variant for weight sharing — scalar params are external."""
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)

    def forward(self, x, x0, attn_scale, mlp_scale, resid_mix):
        mix = resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x
```

### 3. Restructure GPT.forward (and forward_logits)

When `share_pairs` is enabled, the forward loop becomes:

```python
if self.share_pairs:
    x = self.tok_emb(input_ids)
    if self.bigram is not None:
        x = x + self.bigram(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x = self.smear(x)
    x0 = x
    skips: list[Tensor] = []
    
    num_enc = self.num_encoder_layers
    num_dec = self.num_decoder_layers
    
    # Encoder
    for i in range(num_enc):
        # Add iteration conditioning
        x = x + self.iter_embeddings[i].to(dtype=x.dtype)[None, None, :]
        block = self.blocks[self.layer_to_block[i]]
        x = block(x, x0, self.layer_attn_scales[i], self.layer_mlp_scales[i], self.layer_resid_mixes[i])
        skips.append(x)
    
    # Decoder
    for i in range(num_dec):
        eff_idx = num_enc + i
        if skips:
            x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        # Add iteration conditioning
        x = x + self.iter_embeddings[eff_idx].to(dtype=x.dtype)[None, None, :]
        block = self.blocks[self.layer_to_block[eff_idx]]
        x = block(x, x0, self.layer_attn_scales[eff_idx], self.layer_mlp_scales[eff_idx], self.layer_resid_mixes[eff_idx])
    
    x = self.final_norm(x)
    # ... rest of loss computation unchanged
```

**Apply the same changes to `forward_logits`** — it must use the identical forward path, just without loss computation.

### 4. Optimizer setup changes

The optimizer split needs to handle the new parameter structure:

- **Matrix params for Muon:** The `SharedBlock` weight matrices (`c_q.weight`, `c_k.weight`, `c_v.weight`, `proj.weight`, `fc.weight`, `mlp.proj.weight`). These are in `self.blocks` (the 5 unique blocks). Collect from `base_model.blocks.named_parameters()` as before — there are just fewer of them.

- **Scalar params for AdamW:** The per-layer `layer_attn_scales`, `layer_mlp_scales`, `layer_resid_mixes` (from the ParameterLists), plus `skip_weights`, `smear.gate`, `iter_embeddings`, `bigram.scale`. Collect these explicitly.

- **Iteration embeddings for AdamW:** Add `base_model.iter_embeddings` to the scalar params list. These are small (10 × dim = 10 × 576 = 5,760 params) and should use `scalar_lr`.

### 5. Weight initialization

- `SharedBlock` weight init follows the same pattern as `Block` (orthogonal init for large matrices, zero init for proj layers). Use `num_effective_layers` (not `num_unique_blocks`) for the `1/sqrt(2*num_layers)` scaling on proj weights, since the gradient accumulates across both uses of each shared block.
- `iter_embeddings` should be initialized to zeros. The model starts with no iteration conditioning and learns to use it during training.
- Per-layer scalars (`layer_attn_scales`, `layer_mlp_scales`) init to ones, `layer_resid_mixes` init to `[ones, zeros]` — same as the original Block defaults.

### 6. Model dimension change

Change the default:

```python
model_dim = int(os.environ.get("MODEL_DIM", 576))  # was 512
num_heads = int(os.environ.get("NUM_HEADS", 8))     # unchanged; head_dim becomes 72
```

Verify that head_dim = 576 / 8 = 72 is even (required for RoPE). It is.

Also verify that `num_kv_heads=4` still divides `num_heads=8`. It does.

The MLP hidden dim becomes `int(3.0 * 576) = 1728` (was 1536).

### 7. Quantization / serialization

The quantization and serialization code should work without changes — it operates on the state dict, which will simply have fewer block entries (5 instead of 9) but more per-layer scalar entries. The `_classify_param` function routes by name patterns, so `layer_attn_scales`, `layer_mlp_scales`, `layer_resid_mixes`, and `iter_embeddings` will be classified as "other" and stored as passthrough (fp16), which is correct for small tensors.

However, the **eval model construction** at the end of `main()` needs to use `share_pairs=True` and `effective_layers=10` so the dequantized weights load into the right architecture. Make sure the eval model is constructed with the same architecture hyperparameters.

### 8. The non-sharing codepath must still work

When `SHARE_PAIRS=0` (default), the model should behave identically to the current SOTA script. All changes should be gated behind `if self.share_pairs`. This lets you run both configurations from the same script for A/B comparison.

## Environment variables for the run

```bash
SHARE_PAIRS=1
EFFECTIVE_LAYERS=10
MODEL_DIM=576
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=3.0
# Keep all other hyperparameters at their current defaults
```

If the artifact exceeds 16MB at dim=576, fall back to dim=544 (head_dim=68, still even). If dim=576 fits with room to spare, you could try dim=608 (head_dim=76).

## Artifact size budget estimate

At dim=576 with 5 unique SharedBlocks:
- Per SharedBlock: c_q(576×576) + c_k(288×576) + c_v(288×576) + proj(576×576) + fc(1728×576) + mlp.proj(576×1728) = ~2.66M params
- 5 blocks: ~13.3M params → ~8.3MB at int6 + scales
- tok_emb: 1024×576 = 590K → ~369KB at int8
- Per-layer scalars: 10 × (576 + 576 + 1152) = 23K params → ~46KB at fp16
- iter_embeddings: 10 × 576 = 5.8K → ~12KB at fp16
- skip_weights: 5 × 576 = 2.9K → ~6KB at fp16
- bigram: ~524K → ~328KB at int8
- SmearGate: 576 → ~1KB
- Overhead (torch.save metadata, zstd): ~500KB

**Estimated total: ~9.6MB before compression, likely 13-15MB after zstd at level 22.**

This should fit comfortably. If it does, there may be room to try dim=608 in a follow-up run.

## Testing checklist

1. **Non-sharing mode unchanged:** Run with `SHARE_PAIRS=0` and verify output matches the unmodified script exactly (same loss at step 1, same parameter count).

2. **Sharing mode runs:** Run with `SHARE_PAIRS=1 EFFECTIVE_LAYERS=10 MODEL_DIM=576` and verify:
   - No runtime errors through at least 50 training steps
   - Parameter count is lower than the 9-layer dim=512 baseline
   - Gradients flow correctly (no None grads on shared block params)
   - The shared blocks accumulate gradients from both uses (check `.grad` magnitude is roughly 2x a non-shared block)

3. **Artifact size:** After serialization, verify the int6+zstd artifact is under 16MB.

4. **Roundtrip eval works:** The dequantized model loads and produces reasonable val_bpb.

5. **Iteration embeddings are learning:** After 100+ steps, check that `iter_embeddings` is no longer all zeros — the gradients should have moved them.

## What NOT to change

- Do NOT modify the non-sharing codepath (Block class, original GPT.forward when share_pairs=False)
- Do NOT change the Muon optimizer class
- Do NOT change the quantization / serialization functions
- Do NOT change eval_val or eval_val_sliding
- Do NOT change the data loading, tokenizer setup, or distributed training logic
- Do NOT change the training loop structure (warmup, LR schedule, grad clipping, EMA/SWA, wallclock cap)

## Potential issues to watch for

**torch.compile with shared modules:** When the same nn.Module is called multiple times in a forward pass, `torch.compile` with `fullgraph=True` might struggle — it expects a static graph but sees the same module appearing at multiple points. If this causes errors, try `fullgraph=False`, or manually unroll the loop so each call site is syntactically distinct (e.g., store `self.blocks[0]` in a local variable and call it twice with different names, though this likely won't fool the compiler). Worst case, disable `torch.compile` for the sharing mode — the wider dim might compensate for the lost compile speedup.

**Muon gradient scaling:** The shared blocks get ~2x gradient magnitude. Monitor training loss in the first 100 steps — if it's unstable compared to the non-sharing baseline, try reducing `MATRIX_LR` by a factor of 1.5 or 2.
