# Lab Notebook — Parameter Golf (MLX on M2 Pro)

## Record analysis

### NaiveBaseline (1.2244)

- **Config**: 9L, 512dim, 1024 vocab, 8 heads, 4 KV heads, seq=1024, TIED_EMBED_LR=0.05, MATRIX_LR=0.04, SCALAR_LR=0.04, WARMDOWN_ITERS=1200, TRAIN_BATCH_TOKENS=524288
- ~13780 steps in 10 min on H100, ~43.54ms/step
- Pre-quant: 1.2172, post-quant: 1.2244 (penalty: 0.007)
- Artifact: 15,863,489 bytes
- **Code changes**: None — this is the reference baseline.

### LowerLR (1.2230)

- **Config**: Same code as baseline, only env var overrides: MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03
- 14421 steps, step_avg:41.60ms
- **Code changes**: None — identical train_gpt.py to baseline. Pure HP sweep.
- **LR sweep results**: 0.06→1.2445, 0.04→1.2286, 0.025→1.2250, **0.02→1.2230**, 0.015→1.2234

### FP16Embed_WD3600 (1.2197)

- **Config**: WARMDOWN_ITERS=3600, MATRIX_LR=0.06, MLP_HIDDEN=992 (SCALAR_LR and TIED_EMBED_LR unchanged from baseline)
- **Code changes**:
  1. Add `mlp_hidden` env var and plumb through MLP/Block/GPT constructors: `hidden = mlp_hidden if mlp_hidden > 0 else mlp_mult * dim`
  2. In `quantize_state_dict_int8`, change the passthrough condition: `if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL or name == "tok_emb.weight":` — this keeps tied embedding in fp16 instead of int8
- Post-quant penalty drops from ~0.007 to ~0.0005 BPB. Cost: ~500KB extra, offset by MLP_HIDDEN=992.
- **Failed experiments**: SwiGLU, depth recurrence, QAT, lzma, higher embed LR

### LongContextSeq2048 (1.2058)

- **Config**: TRAIN_SEQ_LEN=2048, TIED_EMBED_LR=0.04, MATRIX_LR=0.032, SCALAR_LR=0.032
- **Code changes**: Only default values changed in Hyperparameters class. No architectural changes.
- 11564 steps, step_avg:51.89ms
- 3 seeds: 1.20576, 1.20617, 1.20716 (mean 1.20637)

### TrainingOptSeq4096 (1.2014)

- **Config**: TRAIN_SEQ_LEN=4096, TRAIN_BATCH_TOKENS=393216, TIED_EMBED_LR=0.030, MATRIX_LR=0.020, SCALAR_LR=0.020, MUON_MOMENTUM=0.99, MUON_MOMENTUM_WARMUP_START=0.92, MUON_MOMENTUM_WARMUP_STEPS=1500, WARMDOWN_ITERS=3000
- **Code changes**: Only default values changed. No architectural changes.
- 8394 steps, step_avg:71.47ms. Quant penalty only 0.0034 BPB.

### 10L_MixedPrecision (1.2147)

- **Config**: NUM_LAYERS=10, MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03, INT4_LAYERS=3,4,5,6, INT4_STEP=4
- **Code changes** (post-quantization, after `quantize_state_dict_int8`):
  1. Add `int4_layers` (comma-separated layer indices) and `int4_step` env vars
  2. For layers in `int4_set`, round int8 values to nearest `step`: `((t.float() / step).round() * step).clamp(-127, 127).to(torch.int8)` — this gives 64 effective quantization levels (int6) while still stored as int8, compressing much better with zlib
  3. Optional `prune_ratio` to zero out small int8 values (not used in winning config)
- 10L at dim=512 would be 17.6MB with pure int8. Int6 on middle layers drops to 15.9MB.
- 13100 steps, step_avg:45.78ms

### WarmdownQuantization (1.2154)

- **Config**: WARMDOWN_ITERS=20000, MATRIX_LR=0.06, SCALAR_LR=0.06, TIED_EMBED_LR=0.07, GRAD_CLIP_NORM=1.0, MUON_BACKEND_STEPS=5, EVAL_SEQ_LEN=1408, MLP_HIDDEN=992
- **Code changes** (combines several techniques):
  1. FP16 tied embedding passthrough (same as FP16Embed_WD3600)
  2. MLP_HIDDEN plumbing (same as FP16Embed_WD3600)
  3. **NTK-RoPE extrapolation**: Modified Rotary class to accept `train_seq_len` and dynamically scale RoPE base when `seq_len > train_seq_len`: `base_scaled = base * ((seq_len / train_seq_len) ** (dim / (dim - 2)))`. Eval at 1408 tokens (1.375x train) improves BPB.
  4. **Int6 quantization for all non-passthrough weights**: `quantize_float_tensor(t, bits=6)` — generalized quantizer with `max_val = (2^(bits-1)) - 1`
  5. **Late-K passthrough**: Last 2 layers' `c_k.weight` kept in fp16 instead of quantized
  6. `eval_val` modified to accept `eval_seq_len` override parameter
- Post-quant penalty: 0.014 (baseline) → 0.005 (warmdown alone) → ~0.001 (+ FP16 embed)

### SlidingWindowEval (1.1925)

- **Config**: EVAL_STRIDE=64, EVAL_BATCH_SEQS=1024. Training identical to baseline.
- **Code changes** (eval-time only, ~100 LOC):
  1. Add `forward_logits` method to GPT: same as `forward` but returns logits `(bsz, seq_len, vocab)` instead of computing loss
  2. Add `eval_val_sliding` function:
     - Generate overlapping windows: for each start position in `range(0, total_tokens, stride)`, take a window of `seq_len` tokens ending at `start + seq_len`
     - For each window, only score the rightmost `stride` tokens (except the first window which scores all tokens) — this ensures each token is scored with near-maximum context
     - Batch windows together (`eval_batch_seqs`) for efficient forward passes
     - Accumulate byte-weighted BPB across all scored tokens
     - Distribute windows across GPUs for multi-GPU eval
  3. At end of training, if `eval_stride > 0`, call `eval_val_sliding` instead of `eval_val`
- Pure eval improvement: 0.032 BPB. Eval time: 70s on 8xH100 (vs ~16s baseline).

### LoRA TTT (1.1928)

- **Config**: Training identical to baseline. TTT params: TTT_LORA_RANK=8, TTT_LORA_LR=0.01, TTT_CHUNK_SIZE=256, TTT_EVAL_SEQ_LEN=1024, TTT_BATCH_SIZE=64
- **How it works**: TTT is NOT built into the eval harness — contestants implement it in submission code. At eval time, per document: predict a chunk → score it → one Adam step on LoRA weights → predict next chunk with updated weights. Reset between documents (no cross-document leakage).
- **Code changes** (~200 LOC):
  1. `BatchedLinearLoRA` class: per-batch-element independent A/B matrices. A init: kaiming-uniform (`1/sqrt(in_features)`), B init: zeros. Forward: `x @ A^T @ B^T`.
  2. `BatchedTTTLoRA` class: creates `lm_head_lora` + `q_loras` + `v_loras` (one per block)
  3. Modify `CausalSelfAttention.forward` to accept `q_delta, v_delta` — added to q/v projections before reshape
  4. Modify `Block.forward` to accept `q_delta_fn, v_delta_fn` — computes deltas from attn_norm output
  5. Modify `GPT.forward` to accept `lora` param — passes q/v lora deltas per block, adds `lora.lm_head_lora(x)` to logits. When lora is passed, returns per-token loss `(bsz, seq_len)` instead of mean loss.
  6. `eval_val_ttt_lora` function: find doc boundaries via BOS_ID=1, sort docs by length for batching efficiency, process in batches of 64. Per chunk: forward with grad if training needed, accumulate BPB scores, then one Adam step (betas=0.9/0.95) on non-final chunks. Reset LoRA + optimizer between docs.
- **Ablation**: ~90% of gain from doc isolation + strided eval, ~10% from LoRA TTT itself.

### SOTA (old): SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit (1.1748)

- **Config**: NUM_LAYERS=10, WARMDOWN_ITERS=2500, TIED_EMBED_LR=0.10, MATRIX_LR=0.04, SCALAR_LR=0.04, EVAL_STRIDE=64 (runtime env var). Adam weight_decay=0.01 for tok_emb and scalar params. Muon WD=0.02 (decoupled, applied manually).
- **Code changes** (combines multiple techniques):
  1. **FP16 tied embedding**: Same passthrough as FP16Embed_WD3600 (`if "tok_emb" in name: keep as fp16`)
  2. **Sliding window eval**: Same `forward_logits` + `eval_val_sliding` as SlidingWindowEval, with compiled forward for efficiency. eval_batch_seqs=256.
  3. **NTK-RoPE**: Same dynamic scaling in Rotary as WarmdownQuantization, parameterized by `train_seq_len`
  4. **Decoupled Muon weight decay**: After each training step, manually apply WD to matrix params: `p.mul_(1.0 - 0.02 * optimizer_muon.param_groups[0]["lr"])`. Also `weight_decay=0.01` passed to AdamW for tok_emb and scalar optimizers (changed from Adam to AdamW).
  5. **Overtone spectral embedding init** (after model creation):
     ```python
     U, S, V = torch.linalg.svd(self.tok_emb.weight.data, full_matrices=False)
     target_S = S[0] * (1.0 / torch.arange(1, S.shape[0] + 1, dtype=S.dtype)) ** 0.5
     self.tok_emb.weight.data = (U * target_S[None, :]) @ V
     ```
  6. **Phase-transition residual mixing init** (per block, after model creation):
     ```python
     phase = torch.sigmoid(torch.tensor(3.0 * (i / max(num_layers - 1, 1) - 0.5)))
     block.resid_mix.data[0] = phase * torch.ones(block.resid_mix.shape[1])
     block.resid_mix.data[1] = (1 - phase) * torch.ones(block.resid_mix.shape[1])
     ```
     Note: `resid_mix` is an existing 2xdim learnable parameter in the Block class. It mixes the residual stream with the initial embedding: `x = mix[0] * x + mix[1] * x0`. Early layers trust x0 more, late layers trust the residual.
- 3 seeds: 1.1756, 1.1742, 1.1744 (mean 1.1748). Artifact: ~14.7MB.

### MixedQuant_Int6Int8_SlidingWindow (1.1630) — 2026-03-19

- **Config**: 9L, 512dim, MLP_MULT=3 (hidden=1536), seq=1024, batch=524288, EVAL_STRIDE=64
- **Code changes** (4 orthogonal improvements):
  1. **MLP 3x expansion**: hidden 1024→1536 — largest single contributor. Enabled by int6 compression savings.
  2. **Mixed int6/int8 quantization**: int6 per-row on all 2D block weights (STE-protected), int8 per-row on tok_emb (no STE). Reduces quant penalty from +0.048 to +0.0015 BPB (32x improvement). Int6 stored in int8 containers, zlib compresses zero high bits.
  3. **Seq=1024 + batch=524K**: Shorter sequences = faster steps (48.4ms vs 55.5ms) = more training. 12395 steps × 524K = ~6.5B tokens.
  4. **Sliding window eval stride=64**: ~0.034 BPB free improvement.
- Optimizer: same defaults as baseline (LR=0.04/0.04/0.05).
- 15.35MB artifact. Eval time: 73s.

### Seq2048_FP16Emb_TunedLR (1.1598) — 2026-03-19

- **Config**: 10L, 512dim, MLP_HIDDEN=1344 (2.625x), seq=2048, MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.04, MUON_MOMENTUM=0.99 (warmup 0.92→0.99 over 1500 steps), MUON_WEIGHT_DECAY=0.04, ADAM_WEIGHT_DECAY=0.04, WARMDOWN_ITERS=3000, GRAD_CLIP_NORM=0.3, EVAL_STRIDE=64
- **Code changes**:
  1. **STE int6 QAT**: fake_quantize_int6 on every CastedLinear forward pass. Eliminates quant gap entirely (0.000 BPB penalty).
  2. **Full int6 quantization** on all block weights (layers 0-10).
  3. **zstd-22 compression**: better than zlib for int6 data, saves ~1.5MB.
  4. **FP16 tied embedding passthrough**.
  5. **10 layers** (funded by int6+zstd savings).
  6. **Sliding window eval stride=64**.
- 8319 steps at ~72ms/step. 15.56MB artifact. **QAT overhead: ~28%** (72ms vs 69ms without).
- **Notable**: WD=0.04 for both Muon and AdamW — confirms 0.04 as optimal.

### smeargate_orthoinit_muonwd (1.1556) — 2026-03-19

- **Config**: 9L, 512dim, MLP_MULT=3, seq=1024, batch=524288, MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03, MUON_MOMENTUM=0.99 (warmup 0.92→0.99 over 1500 steps), MUON_WEIGHT_DECAY=0.01, WARMDOWN_ITERS=3000, EVAL_STRIDE=64
- **Code changes** (introduces 3 new architectural techniques):
  1. **SmearGate**: learned per-dim gate (~512 params) blending current + previous token embedding: `gate = sigmoid(self.gate); output = gate * curr + (1-gate) * prev`. Gate init: sigmoid(3.0)≈0.95 (near-identity). Adds bigram context at embedding layer for free.
  2. **BigramHash(4096, dim=128)**: hash table mapping `(prev*92821+cur)%4096` to 128-dim embeddings, projected to 512. ~524K params. Additive bigram signal complementing SmearGate.
  3. **Orthogonal init**: all CastedLinear weights initialized with `orthogonal_(gain=1.0)`. Output projections scaled by `1/sqrt(2*num_layers)` (muP convention). Uniform gradient flow from step 1.
  4. **Int6 QAT STE** + **zstd-22** + **FP16 embed passthrough** + **sliding window stride=64**.
- 12047 steps, 50ms/step. 15.1MB artifact. Quant gap: ~0.0001 BPB (QAT nearly eliminates it).
- **U-Net skip connections**: 4 encoder + 5 decoder layers with learned skip weights.

### Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA (1.1458) — 2026-03-20

- **Config**: 9L, 512dim, MLP_MULT=3, seq=2048, batch=786432, MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03, MUON_MOMENTUM=0.99 (warmup 0.92→0.99 over 1500 steps), MUON_WEIGHT_DECAY=0.04, ADAM_WEIGHT_DECAY=0.01, WARMDOWN_ITERS=3000, EVAL_STRIDE=64, SWA_EVERY=50, SWA_START_FRAC=0.5, GRAD_CLIP_NORM=0.3
- **Code changes** (builds on smeargate_orthoinit_muonwd + adds SWA + higher WD):
  1. All techniques from smeargate_orthoinit_muonwd (SmearGate, BigramHash(4096), OrthoInit, MLP3x, int6 QAT, zstd-22, FP16 embed, sliding window).
  2. **SWA**: average weights every 50 steps over last 50% of training (~30 checkpoints). Produces smoother weight distributions that quantize better. Swept swa_every from 200 down to 25; optimal at 50.
  3. **Muon WD=0.04** (up from 0.01). AdamW WD stays at 0.01.
- 7379 steps at 81.3ms/step. ~22M params. 15.86MB artifact.
- 3 seeds: 1.1460, 1.1466, 1.1449 (mean 1.1458, std 0.0008).
- Pre-quant: 1.1616. Quant penalty: 0.016 BPB.

### SOTA (current): 10L_Int5MLP_MuonWD04_SWA50 (1.14276) — 2026-03-20

- **Config**: 10L, 512dim, 8 heads, 4 KV heads (GQA), MLP 3x (hidden=1536), relu^2, seq=2048, batch=786K, MATRIX_LR=0.02, MUON_WEIGHT_DECAY=0.04, ADAM_WEIGHT_DECAY=0.04, WARMDOWN_ITERS=3000, WARMUP=20, GRAD_CLIP_NORM=0.3, EVAL_STRIDE=64, SWA_EVERY=50, SWA_START_FRAC=0.4
- **Code changes** (builds on Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA + 3 key additions):
  1. **Int5 quantization for MLP** [-16,15]: MLP weights quantized to int5 (1.88x zstd ratio) vs int6 for attention (1.51x ratio). Saves ~1.86MB vs uniform int6, funding a 10th layer. FP16 for tied embeddings and last-layer key projections.
  2. **BigramHash(10240)**: up from 4096 buckets. Reduces token-pair hash collisions (+0.001 bpb improvement).
  3. **SWA start_frac=0.4** (down from 0.5): collect checkpoints only from last 40% of warmdown (most converged). Quality over quantity: 24 checkpoints averaged every 50 steps.
  4. **AdamW WD=0.04** for embeddings/scalars (up from 0.01). **Validates our independent finding that embedding WD matters!**
  5. **3% magnitude pruning** on weights.
  6. SmearGate, BigramHash(10240), OrthoInit, MLP3x, U-Net skips, tied embeddings.
- 3 seeds: 1.14271, 1.14298, 1.14260 (mean **1.14276**, std 0.00016). ~15.9MB artifact.
- **Ablation**: 9L int6 (1.1485) → +int5 MLP+10L (1.1453, -0.003) → +WD+warmdown (1.1452) → +SWA (1.1446) → +bigram8192 (1.1434) → +bigram10240 (**1.1426**).

### Upstream train_gpt.py changes (2026-03-20)

- **LoRA TTT code removed entirely** from base train_gpt.py. All TTT classes (BatchedLinearLoRA, BatchedTTTLoRA, eval_val_ttt_lora) and related function signatures stripped. Contestants wanting TTT must now implement it fully in their submission code.
- Block.forward, CausalSelfAttention.forward, and GPT.forward simplified — removed q_delta, v_delta, lora parameters.

## Key takeaways and strategy notes

1. **FP16 embeddings** are a near-free win (~0.006-0.007 BPB from reduced quant error)
2. **Extended warmdown** (much larger than actual steps) reduces weight outliers → better compression
3. **10 layers** possible with int6/int5 + zstd compression (10L is now standard in top records)
4. **Sliding window eval** is a huge win (~0.034 BPB) and standard in all top records
5. **Higher LR + longer warmdown** is a productive combo
6. **Lower LR reduces quant penalty** — tradeoff between training quality and compression
7. **Context length**: seq=2048 is now standard in top records (seq=1024 used by some for throughput)
8. **TTT is a contestant-implemented strategy** — LoRA TTT code was removed from upstream train_gpt.py
9. **QAT (STE int6)** eliminates quant gap entirely (~28% overhead in step time, but worth it)
10. **MLP 3x expansion** is the single largest architectural win — enabled by int6/int5 compression savings
11. **SmearGate + BigramHash** provides cheap bigram context at embedding layer
12. **SWA** produces smoother weight distributions that quantize better (start_frac=0.4, every=50 steps)
13. **zstd-22** saves ~5% over zlib-9 on int6 data — critical for fitting more layers/params under 16MB
14. **Int5 for MLP, int6 for attention** — mixed precision quantization saves ~1.86MB
15. **WD=0.04** confirmed optimal for both Muon and AdamW (our embedding WD finding validated!)
16. **Orthogonal init** accelerates convergence with Muon (already-orthogonal → useful updates immediately)
17. **Magnitude pruning (3%)** provides small additional compression

## Ideas queue

Ideas to try in future experiments. Remove when tried or invalidated.

**High priority — proven in SOTA/near-SOTA records (H100 only):**
- QAT (STE int6) — fake quantize during training, eliminates quant gap. ~28% overhead but worth it.
- MLP 3x expansion (hidden=1536) — largest single contributor. Needs int6+zstd to fit under 16MB.
- SmearGate + BigramHash(10240) — cheap bigram context at embedding layer.
- SWA (start_frac=0.4, every=50 steps) — smoother weights → better quantization.
- Int5 for MLP + int6 for attention — mixed quantization saves ~1.86MB, funds 10th layer.
- zstd-22 compression — ~5% better than zlib-9, critical for fitting larger models.
- Orthogonal init — all CastedLinear weights via orthogonal_(gain=1.0), muP-scaled outputs.
- 10 layers — standard in top records, funded by compression savings.
- WD=0.04 for both Muon and AdamW — confirmed optimal. Our finding validated!

**Medium priority — speed optimizations (local-testable, more steps = more learning):**
- ~~Reduce Newton-Schulz iterations 5→3~~ (exp 30 — 0.108 worse, NS quality matters more than speed savings)
- Grad clipping in MLX instead of CPU. Current `clip_grad_tree` converts every gradient to numpy (CPU roundtrip mid-step). Rewrite entirely in mx.array ops to stay on Metal GPU.
- Larger microbatch chunks (`MLX_MAX_MICROBATCH_TOKENS=16384` or `MLX_EAGER_EVAL=0`). Currently 8K chunks = 8 forward/backward passes per microbatch. Fewer chunks = less overhead.
- ~~Fused QKV projection~~ (exp 33 — 0.058 worse, Muon needs separate projections for independent orthogonalization)
- Shorter sequence length (`TRAIN_SEQ_LEN=512`). Attention is O(N²) — halving seq_len gives ~4x cheaper attention per token. More steps in 10 min. Tradeoff: less context per token. **Transfers to H100** (MixedQuant record already used seq=1024 for throughput).
- Fewer layers + wider MLP (e.g. 7L MLP_MULT=3 instead of 9L MLP_MULT=2). Fewer sequential passes, more capacity per layer. MLPs are parallel, depth is sequential. **Transfers to H100.**

**Medium priority — local-testable quality ideas:**
- ~~Int6 quantization (without QAT)~~ (exp 31 — quant penalty 0.173, needs QAT to work. H100 only.)
- NTK-RoPE extrapolation at eval time (EVAL_SEQ_LEN > TRAIN_SEQ_LEN). See WarmdownQuantization record.
- ~~Late-K passthrough~~ (exp 32 — 0.003 worse, int8 quant penalty already low enough)
- Weight snapping (round weights toward quantization grid before final export)
- ~~Label smoothing~~ (exp 29 — 0.344 worse, prevents confident predictions which directly hurts BPB)
- ~~WD on scalar params~~ (exp 24 — 0.008 BPB, kept)
- ~~Embed LR ratio~~ (exp 25 — no change)

**Lower priority — speculative:**
- Depth recurrence (weight tying across layers — halves params for same effective depth)
- Zlib-aware regularization (penalize weight patterns that compress poorly)
- Ensemble eval (average logits from multiple checkpoints at eval time)

**Already tried/invalidated (do not re-try):**
- ~~Sliding window eval~~ (done, exp 21 — stride=512 locally, use stride=64 on H100)
- ~~Overtone spectral embedding init~~ (exp 22 — 0.023 worse)
- ~~Phase-transition residual mixing init~~ (exp 23 — 0.022 worse, needs long training)
- ~~Extended warmdown~~: warmdown=1200 already puts entire training in warmdown on M2 Pro (LR_mul≈0.14). Reducing warmdown to 400 was worse. Current schedule is effectively "always decaying".
- ~~LR above 0.30/0.30/0.35~~: LR=0.40 was worse (exp 14). 0.30 is optimal with current warmdown.
- ~~Muon WD above 0.32~~: WD=0.64 increased quant penalty too much (exp 18). 0.32 is optimal locally.
- ~~FP16 embed~~: Benefit too small (~0.0003) to measure in smoke tests. Standard in all top H100 records.
- ~~10 layers~~: Too slow on M2 Pro (158 vs 175 steps). Standard on H100 — use there.
- ~~Shorter momentum warmup~~: Higher momentum hurt in short-training regime (exp 12).
- ~~Reducing model dim~~: M2-specific artifact (more steps in 10-min cap). Won't transfer to H100. Reverted to dim=512.
- ~~Warmdown 3000~~: Cuts effective LR_mul from 0.14 to 0.057 on M2 (exp 38, 0.108 worse). Designed for H100's 13K steps.
- ~~Disable eager eval~~: 3x slower on 16GB M2 Pro (exp 41). Need 32GB+ unified memory.
- ~~Microbatch 32K~~: Faster step time but slightly worse quality than 16K (exp 40). 16K is optimal.

## Experiment log

(Entries will be appended below as experiments complete)

### Experiment 1: Baseline (2026-03-19 21:36)
- **Config**: 9L, 512dim, 1024 vocab, 65K batch/1x accum, warmup=20, warmdown=1200, LR=0.04/0.04/0.05
- **Result**: roundtrip_val_bpb=2.429389, artifact=7.04MB, 176 steps in 10min, KEEP
- **Training dynamics**: Loss dropped rapidly (6.94→6.69 in first 10 steps), ~3.4s/step, ~19K tok/s. Still decreasing at stop (val_bpb=2.42 at step 176). Lots of room for improvement with more steps on full data.
- **Quant penalty**: pre-quant val_bpb=2.4182 vs post-quant=2.4294, penalty=0.011. Significant — FP16 embeddings could help.
- **Note**: This is smoke-test-only (1 shard, 10-min cap). Absolute numbers are much worse than full-data H100 runs. Use as directional baseline for comparing experiments.
- **Ideas**: FP16 embeddings (records show ~0.006 BPB win), extended warmdown (already at 1200, try much larger), 10 layers (needs size management).

### Experiment 2: FP16 embeddings (2026-03-19 22:52)
- **Hypothesis**: Keeping tok_emb.weight in fp16 instead of int8 during serialization reduces quantization error that compounds through both input embedding and output projection (tied weights). Records show ~0.006 BPB improvement.
- **Config**: Same as baseline, only change is adding "tok_emb" to fp16 passthrough in quantize_state_dict_int8.
- **Result**: roundtrip_val_bpb=2.453442, artifact=7.43MB, 173 steps in ~21min, DISCARDED
- **Training dynamics**: Similar trajectory to baseline (6.94→6.89 in 10 steps), step_avg=3481ms. Pre-quant val_bpb=2.4441 vs baseline's 2.4182.
- **Key insight**: The code change ONLY affects post-training serialization, not training itself. Pre-quant val_bpb difference (0.026) reveals significant run-to-run variance in MLX smoke tests. The quant penalty DID decrease (0.0093 vs baseline 0.0112), confirming FP16 embeddings help with quantization.
- **Conclusion**: FP16 embeddings are still a valid technique (proven by records), but the benefit (~0.002 BPB quant reduction) is drowned by smoke test variance (~0.026). This will help on production H100 runs where pre-quant performance is stable across seeds.
- **Next ideas**: Extended warmdown (much larger WARMDOWN_ITERS), which is a training-time change that should show clearer directional signal.

### Experiment 3: Higher base LR (2026-03-19 23:29)
- **Hypothesis**: With warmdown_iters=1200 and ~3.5s/step on M2 Pro, the entire training runs in warmdown with LR_mul starting at only ~0.14. Higher base LR (0.06/0.06/0.07 from 0.04/0.04/0.05) compensates, matching the FP16Embed_WD3600 record's LR settings.
- **Result**: roundtrip_val_bpb=2.2960 (vs baseline 2.4294), artifact=7.63MB, 175 steps, **KEEP — 0.133 BPB improvement!**
- **Training dynamics**: Loss dropped faster (6.58 at step 10 vs baseline ~6.69). Pre-quant val_bpb=2.2829, quant penalty=0.0131 (slightly higher than baseline's 0.0112, expected with higher LR). Loss still decreasing at step 175.
- **Key insight**: The baseline was severely under-learning with effective LR of only ~0.006 (0.04 * 0.14). Higher base LR dramatically improved learning. This is the most impactful change so far.
- **Next ideas**: Try even higher LR (0.08/0.08/0.10)? Or combine with FP16 embeddings now that we have a better baseline. Also try extended warmdown to reduce the quant penalty.

### Experiment 4: Even higher LR (2026-03-20 00:04)
- **Hypothesis**: LR was the clear bottleneck in exp 3. Push further to 0.08/0.08/0.10.
- **Result**: roundtrip_val_bpb=2.1816 (vs 2.2960), artifact=8.10MB, 176 steps, **KEEP — 0.114 BPB improvement!**
- **Training dynamics**: Step 10 loss 6.51 (vs 6.58 with 0.06 LR). Pre-quant val_bpb=2.1704, quant penalty=0.0112. Loss still decreasing at step 176.
- **Key insight**: LR is still undertuned — more gains from higher LR. Artifact size grew from 7.63→8.10MB (higher LR produces larger weight magnitudes → worse compression ratio). Quant penalty stable at 0.011.
- **Cumulative**: baseline 2.4294 → exp3 2.2960 → exp4 2.1816. Total improvement: 0.248 BPB from LR alone.
- **Next ideas**: Push LR even higher (0.12/0.12/0.15)? The artifact size trend (7.04→7.63→8.10) suggests weights are getting bigger — may eventually hit 16MB. Also: FP16 embed to reduce the 0.011 quant penalty.

### Experiment 5: Push LR higher (2026-03-20 00:39)
- **Hypothesis**: LR trend is clear — keep pushing. Try 0.12/0.12/0.15.
- **Result**: roundtrip_val_bpb=2.0603 (vs 2.1816), artifact=8.85MB, 176 steps, **KEEP — 0.121 BPB improvement!**
- **Training dynamics**: Step 10 loss 6.23 (vs 6.51). Pre-quant val_bpb=2.0527, quant penalty=0.0076 (improved!). Loss still decreasing at step 176.
- **Artifact trend**: 7.04 → 7.63 → 8.10 → 8.85 MB. Growing but still under 16MB.
- **Cumulative**: 2.4294 → 2.2960 → 2.1816 → 2.0603. Total: 0.369 BPB improvement from LR alone.
- **Next ideas**: Keep pushing LR (0.20/0.20/0.25)? Diminishing returns may set in, but no signs yet. Watch artifact size.

### Experiment 6: LR 0.20/0.20/0.25 (2026-03-20 01:14)
- **Hypothesis**: LR trend continuing strong, push to 0.20/0.20/0.25.
- **Result**: roundtrip_val_bpb=2.0063 (vs 2.0603), artifact=9.90MB, 176 steps, **KEEP — 0.054 BPB**
- **Training dynamics**: Step 10 loss 6.00 (vs 6.23). Pre-quant=2.0028, quant penalty=0.0035 (best yet!).
- **Gains slowing**: 0.133 → 0.114 → 0.121 → 0.054. Diminishing returns starting. Artifact: 9.90MB (was 8.85).
- **Artifact concern**: 7.04 → 7.63 → 8.10 → 8.85 → 9.90 MB. At this rate, next LR step may push toward 16MB.
- **Next ideas**: One more LR push (0.30/0.30/0.35) to find the peak? Or try a different axis: grad_clip_norm (records use 1.0), weight decay, or FP16 embed.

### Experiment 7: LR 0.30/0.30/0.35 (2026-03-20 01:48)
- **Hypothesis**: One more LR push to find the ceiling.
- **Result**: roundtrip_val_bpb=2.0009 (vs 2.0063), artifact=10.72MB, 176 steps, **KEEP — 0.005 BPB (marginal)**
- **Training dynamics**: Step 10 loss 5.88 (vs 6.00). Pre-quant=1.9984, quant penalty=0.0025 (best yet).
- **LR diminishing returns confirmed**: Gains: 0.133 → 0.114 → 0.121 → 0.054 → 0.005. LR is near optimal for this warmdown schedule.
- **Artifact**: 10.72MB (growing ~1MB per LR step). Cannot push much further without hitting 16MB.
- **Strategy shift**: LR sweep is done. Time to try different axes:
  1. FP16 embeddings (quant penalty now only 0.0025, less headroom but still worth trying)
  2. Grad clip norm=1.0 (records use this)
  3. Weight decay (Muon WD=0.02 — records show this helps compression)
  4. Smaller warmdown_iters (currently 1200, try reducing to give higher effective LR without bigger base LR)
  5. 10 layers (if WD helps control artifact size)

### Experiment 8: Grad clip norm=1.0 (2026-03-20 02:23)
- **Hypothesis**: Records use grad_clip_norm=1.0. Can stabilize high-LR training and constrain weight magnitudes.
- **Result**: roundtrip_val_bpb=1.9905 (vs 2.0009), artifact=10.60MB, 175 steps, **KEEP — 0.010 BPB**
- **Training dynamics**: Step 10 loss 5.71 (vs 5.88). Pre-quant=1.9884, quant penalty=0.0021 (lowest yet!).
- **Key insight**: Grad clipping helped in two ways: (1) better training (loss lower at step 10), (2) smaller artifact (10.60 vs 10.72MB — clipping constrains weight magnitudes). Quant penalty also decreased.

### Experiment 9: Reduce warmdown_iters to 400 (2026-03-20 02:58)
- **Hypothesis**: Lower warmdown_iters gives higher effective LR multiplier (0.44 vs 0.15), 3x more total learning.
- **Result**: roundtrip_val_bpb=2.0280 (vs 1.9905), artifact=12.87MB, 175 steps, **DISCARDED — 0.038 worse**
- **Training dynamics**: Step 10 loss 5.60 (faster early learning) but final val_bpb worse. Artifact ballooned.
- **Key insight**: Extended warmdown (1200) provides gentle decay producing tighter weight distributions. Reducing it hurt compression heavily (12.87 vs 10.60MB) and final quality. Confirms records: extended warmdown is important for artifact compression.
- **Next ideas**: Try increasing warmdown further? Or try weight decay (Muon WD=0.02), or try reducing base LR back while keeping warmdown=1200 (maybe 0.20 was the sweet spot with the current warmdown). Or try 10 layers.

### Experiment 10: Muon weight decay 0.02 (2026-03-20 03:34)
- **Result**: roundtrip_val_bpb=1.9869 (vs 1.9905), artifact=10.47MB, **KEEP — 0.004 BPB**
- WD shrank artifact (10.47 vs 10.60MB) while slightly improving val_bpb.

### Experiment 11: 10 layers (2026-03-20 04:09)
- **Result**: roundtrip_val_bpb=2.0215 (vs 1.9869), artifact=11.20MB, **DISCARDED — too slow locally (158 steps vs 175)**

### Experiment 12: Shorter muon momentum warmup 500→100 (2026-03-20 04:48)
- **Result**: roundtrip_val_bpb=2.0018 (vs 1.9869), artifact=10.89MB, **DISCARDED — higher momentum hurt**
- Slow momentum warmup acts as implicit regularization in our short-training regime.

**Current best**: val_bpb=1.9869, LR 0.30/0.30/0.35, warmdown=1200, grad_clip=1.0, muon_wd=0.02, 9L/512dim.

### Experiments 13-15: WD sweep + LR push
- **WD=0.04** (exp 13): 1.9844, 10.33MB KEEP
- **LR=0.40 with WD=0.04** (exp 14): 2.0013, 10.75MB DISCARD (LR too high)
- **WD=0.08** (exp 15): 1.9793, 10.04MB KEEP

WD trend: higher WD improves both pre-quant quality AND compression.
| WD | pre-quant | roundtrip | artifact |
|----|-----------|-----------|----------|
| 0 | 1.9884 | 1.9905 | 10.60 |
| 0.02 | 1.9847 | 1.9869 | 10.47 |
| 0.04 | 1.9820 | 1.9844 | 10.33 |
| 0.08 | 1.9767 | 1.9793 | 10.04 |

### WD sweep continued + FP16 embed revisit (exps 16-19)
- **WD=0.16**: 1.9705, 9.49MB KEEP
- **WD=0.32**: 1.9578, 8.51MB KEEP (best!)
- **WD=0.64**: 1.9622, 6.98MB DISCARD (quant penalty 0.010)
- **FP16 embed**: 1.9617, 8.77MB DISCARD (too small vs variance)

WD=0.32 is optimal. FP16 embed too small to measure locally (save for H100 runs).

**Current best**: val_bpb=1.9578, artifact=8.51MB. Config: LR=0.30/0.30/0.35, warmdown=1200, grad_clip=1.0, muon_wd=0.32.
**Progress**: 2.4294 → 1.9578 = 0.472 BPB over 19 experiments.

### Experiment 20: WD on embedding (2026-03-20 09:29)
- **Hypothesis**: Embedding (Adam-optimized) has no WD. Adding decoupled WD should help compress it.
- **Result**: roundtrip_val_bpb=1.9268 (vs 1.9578), artifact=8.46MB, **KEEP — 0.031 BPB! Biggest gain since LR sweep!**
- Pre-quant=1.9223, quant penalty=0.0045. Step 10 loss 5.69 (slightly better).
- **Key insight**: Embed WD is the biggest single non-LR improvement. The tied embedding is used twice (input+output), so regularizing it has outsized impact.

**Current best**: val_bpb=1.9268, artifact=8.46MB.

### Experiment 21: Sliding window eval (2026-03-20 11:39)
- **Result**: roundtrip_val_bpb=1.9216 (stride=512), artifact=8.46MB, **KEEP — 0.005 BPB**
- Eval-only change. H100: use EVAL_STRIDE=64 for ~0.032 BPB.

### Experiment 22: Overtone spectral embedding init (2026-03-20 12:29)
- **Result**: roundtrip_val_bpb=1.9493, **DISCARDED — 0.023 worse**

### Experiment 23: Phase-transition residual mixing init (2026-03-20 13:05)
- **Result**: roundtrip_val_bpb=1.9484, **DISCARDED — 0.022 worse**
- Both SOTA inits hurt with ~170 steps. Save for H100.

### Experiment 24: WD on scalar params (2026-03-20 13:43)
- **Result**: roundtrip_val_bpb=1.9192 (vs 1.9268), artifact=8.30MB, **KEEP — 0.008 BPB**
- WD=0.32 now applies to all param groups (matrix, embed, scalar).

### Experiment 25: Higher embed LR ratio (2026-03-20 14:18)
- **Result**: roundtrip_val_bpb=1.9194 (vs 1.9192), **DISCARDED — no change**

### Experiment 26: Reduce warmup steps 20→5 (2026-03-20 14:55)
- **Result**: roundtrip_val_bpb=1.9158 (vs 1.9192), artifact=8.30MB, **KEEP — 0.003 BPB (borderline)**

### Experiments 27-28: Reduce model dim (2026-03-20 15:30)
- dim=448: 1.8649 (0.051 better), dim=384: 1.7850 (0.080 better) — massive smoke-test improvements.
- **REVERTED**: These gains are M2-specific (more steps in 10-min cap). Won't transfer to H100 with 13K steps. Dim stays at 512.

### Experiment 29: Label smoothing 0.1 (2026-03-20 18:05)
- **Hypothesis**: Label smoothing prevents extreme logits, potentially improving quantization. May also improve generalization.
- **Result**: roundtrip_val_bpb=2.2597 (vs 1.9158), artifact=8.18MB, **DISCARDED — 0.344 worse!**
- Pre-quant val_bpb=2.2541, quant penalty=0.0056. Step 10 loss 7.24 (inflated by smoothing target).
- **Key insight**: Label smoothing at 0.1 is far too aggressive for this task. It prevents confident predictions which directly hurts BPB. The smoothed loss targets essentially cap how much probability mass the model can put on the correct token. In a short-training regime where the model is already under-trained, further penalizing confidence is catastrophic.

### Experiment 30: Newton-Schulz iterations 5→3 (2026-03-20 18:56)
- **Hypothesis**: Fewer NS iterations = faster Muon optimizer step = more training steps in 10-min cap. Approximate orthogonality should be sufficient since gradients refresh each step.
- **Result**: roundtrip_val_bpb=2.0237 (vs 1.9158), artifact=6.35MB, **DISCARDED — 0.108 worse!**
- Step 10 loss 5.83 (vs 5.69). Step_avg=3481ms (vs ~3470ms — barely faster). 173 steps (vs 175).
- **Key insight**: The reduced orthogonalization quality hurt training significantly without saving meaningful compute. Steps 2-3 showed unusual spikes (19.27, 14.96), suggesting instability from poor early orthogonalization. The Muon optimizer in MLX appears to need all 5 NS iterations — unlike PyTorch where 3 may suffice due to different numerics. Not worth pursuing.

### Experiment 31: Int6 post-quantization step=4 (2026-03-20 19:32)
- **Hypothesis**: Snap int8 values to nearest multiple of 4 (effectively int6) for better zlib compression. Used in SOTA records with QAT.
- **Result**: roundtrip_val_bpb=2.0948 (vs 1.9158), artifact=3.52MB, **DISCARDED — 0.179 worse**
- Pre-quant val_bpb=1.9214 (same as baseline). Quant penalty=0.1734 (vs 0.004 with int8).
- **Key insight**: Int6 without QAT is catastrophically lossy. Compression is amazing (3.52MB vs 8.30MB, 58% reduction), proving int6 is the right approach for fitting more capacity. But the model MUST be trained to be robust to int6 quantization (STE/QAT). This is an H100-only change — QAT adds ~28% step time overhead, not worth it in 170-step smoke tests.

### Experiment 32: Late-K passthrough (2026-03-20 20:08)
- **Hypothesis**: Keep last 2 layers' c_k.weight in fp16 instead of int8. Key projections in final layers are most sensitive to quantization error.
- **Result**: roundtrip_val_bpb=1.9187 (vs 1.9158), artifact=8.66MB, **DISCARDED — 0.003 worse**
- Pre-quant=1.9138, quant penalty=0.0049 (vs ~0.004 baseline).
- **Key insight**: Our int8 quant penalty is already very low (~0.004). Late-K passthrough added 0.36MB but didn't reduce the penalty enough. This technique is more valuable with int6 quantization (used in SOTA) where quant penalty is much higher. Not worth the artifact cost at pure int8.

### Experiment 33: Fused QKV projection (2026-03-20 20:43)
- **Hypothesis**: Single c_qkv matmul instead of 3 separate c_q/c_k/c_v. Fewer kernel launches, better hardware utilization, transfers to H100.
- **Result**: roundtrip_val_bpb=1.9734 (vs 1.9158), artifact=8.21MB, **DISCARDED — 0.058 worse!**
- Pre-quant=1.9683, step_avg=3500ms (not faster than 3478ms). 172 steps (fewer).
- **Key insight**: Fused QKV hurts because Muon orthogonalizes each 2D matrix independently. Separate Q/K/V allows the optimizer to learn each projection in its own orthogonal subspace. A single 512x1024 matrix constrains the learning dynamics. This is a known interaction with Muon — don't fuse projections that Muon optimizes separately.

### Experiment 34: Orthogonal init (2026-03-20 21:19)
- **Hypothesis**: Initialize CastedLinear weights with QR-orthogonal matrices. Muon orthogonalizes gradients, so starting from orthogonal weights means productive updates from step 1. Used in SOTA records.
- **Result**: roundtrip_val_bpb=1.9183 (vs 1.9158), artifact=8.42MB, **DISCARDED — 0.003 worse (within noise)**
- Pre-quant=1.9133 (slightly better!), quant penalty=0.005 (slightly worse). Step 10 loss 5.64 (vs 5.69 baseline).
- **Key insight**: Orthogonal init improved pre-quant quality marginally but slightly increased artifact size (8.42 vs 8.30) and quant penalty. The init quality improvement is too small for 170 steps. Promising for H100 with 13K steps where the convergence acceleration compounds. Keep in H100-only queue.

### Experiment 35: Tighter grad clip 1.0→0.3 (2026-03-20 21:55)
- **Hypothesis**: SOTA records use grad_clip=0.3. Tighter clipping may produce smoother weight distributions.
- **Result**: roundtrip_val_bpb=1.9149 (vs 1.9158), artifact=8.29MB, **KEEP — 0.001 BPB (borderline)**
- Pre-quant=1.9105 (0.005 better), quant penalty=0.0044. Step 10 loss 5.73 (slightly slower early learning as expected with tighter clip).
- **Note**: Borderline improvement but aligns with SOTA practice. Pre-quant improvement is more convincing than roundtrip delta.

### Experiment 36: MLP 3x expansion (2026-03-20 22:30)
- **Hypothesis**: MLP 3x (hidden=1536) is the single largest architectural win in SOTA records. More capacity per step should help.
- **Result**: roundtrip_val_bpb=1.9456 (vs 1.9149), artifact=10.10MB, **DISCARDED — 0.031 worse**
- Pre-quant=1.9407. 156 steps (vs 173), step_avg=3869ms (15% slower).
- **Key insight**: MLP 3x needs more training steps to utilize the extra capacity. On M2 with only 156 steps (-10%), the model can't learn enough to use the extra parameters. On H100 with 12K+ steps, this is consistently a huge win. Classic M2 vs H100 divergence — note in H100-only queue.

### Experiment 37: Muon momentum 0.95→0.99 (2026-03-20 23:09)
- **Hypothesis**: SOTA records use momentum=0.99. Higher momentum accumulates more gradient history.
- **Result**: roundtrip_val_bpb=1.9142 (vs 1.9149), artifact=8.35MB, **KEEP — 0.001 BPB (borderline)**
- Pre-quant=1.9091 (0.006 better). 174 steps, step_avg=3459ms.
- **Note**: Borderline, but combined with exp 35 (clip=0.3), we're now matching two key SOTA HP choices: clip=0.3, momentum=0.99.

**Current best**: val_bpb=1.9142, artifact=8.35MB. Config: 9L/512dim, LR=0.30/0.30/0.35, warmdown=1200, grad_clip=0.3, muon_wd=0.32 (all params), warmup=5, momentum=0.99.
**Progress**: 2.4294 → 1.9142 = 0.515 BPB over 37 experiments.

### Experiment 38: Warmdown 1200→3000 (2026-03-21 00:07)
- **Hypothesis**: SOTA records use warmdown=3000. More gradual LR decay should help convergence.
- **Result**: roundtrip_val_bpb=2.0220 (vs 1.9142), artifact=7.72MB, **DISCARDED — 0.108 worse!**
- Pre-quant val_bpb=2.0138. 175 steps, step_avg=3439ms.
- **Key insight**: warmdown=3000 cuts effective LR_mul from 0.14 (warmdown=1200) to 0.057 (warmdown=3000), a ~60% reduction. The model barely learns. This setting is designed for H100 with 13K steps where LR_mul = (13000*43/1000) / 3000 ≈ 0.19 — much healthier. On M2 with 170 steps, even warmdown=1200 puts entire training in warmdown. Going higher makes it strictly worse. Add to "Already tried" list.

### Experiment 39: Larger microbatch chunks 8K→16K (2026-03-21 00:46)
- **Hypothesis**: Fewer sub-batches per step (4 vs 8) means less kernel launch overhead, potentially faster training.
- **Result**: roundtrip_val_bpb=1.8768 (vs 1.9142), artifact=8.35MB, **KEEP — 0.037 BPB improvement!**
- Pre-quant=1.8715, quant penalty=0.005. 177 steps (vs 174), step_avg=3404ms (vs 3459ms — 1.6% faster).
- **Key insight**: Larger sub-batches allow MLX to build more efficient compute graphs. 3 extra steps + better graph optimization = significant quality gain. 16K is the sweet spot.

### Experiment 40: Microbatch 32K tokens (2026-03-21 01:28)
- **Result**: roundtrip_val_bpb=1.8885 (vs 1.8768), artifact=8.38MB, **DISCARDED — 0.012 worse than 16K**
- 179 steps, step_avg=3366ms. Fastest config but slightly worse quality. 16K is optimal.

### Experiment 41: Disable eager eval (2026-03-21 02:06)
- **Result**: KILLED — 3x slower (9.6s/step vs 3.4s). Lazy graph accumulation causes massive memory pressure on 16GB M2 Pro. Not viable.

### Experiment 42: Shorter seq_len 1024→512 (2026-03-21 02:09)
- **Hypothesis**: Halve attention cost (O(N²)), getting more steps in 10 min. seq=1024 used by MixedQuant record for throughput over seq=2048.
- **Result**: roundtrip_val_bpb=1.7532 (vs 1.8768), artifact=8.63MB, **KEEP — 0.124 BPB improvement! Biggest single gain!**
- Pre-quant=1.7492, quant penalty=0.004. 224 steps (vs 177), step_avg=2684ms (21% faster).
- **Key insight**: 27% more steps compensates for less context per token. Sliding window eval at stride=64 still gives full context at eval time. This transfers to H100 — the MixedQuant record chose seq=1024 over 2048 for throughput for exactly this reason.
- **Caution**: Unlike dim reduction (M2-only win), seq_len reduction is validated by top records. MixedQuant went from 2048→1024 and benefited. Our 1024→512 follows the same logic. On H100 with 13K steps, fewer steps matter less, but the throughput gain (more tokens/sec) means more total training tokens in the same wallclock.

**Current best**: val_bpb=1.7532, artifact=8.63MB. Config: 9L/512dim, LR=0.30/0.30/0.35, warmdown=1200, grad_clip=0.3, muon_wd=0.32, warmup=5, momentum=0.99, microbatch=16K, seq=512.
**Progress**: 2.4294 → 1.7532 = 0.676 BPB over 42 experiments.

