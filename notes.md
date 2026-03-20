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

### SOTA: SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit (1.1748)

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

## Key takeaways and strategy notes

1. **FP16 embeddings** are a near-free win (~0.006-0.007 BPB from reduced quant error)
2. **Extended warmdown** (much larger than actual steps) reduces weight outliers → better compression
3. **10 layers** possible if we use weight decay or mixed precision to fit in 16MB
4. **Sliding window eval** is a huge win but requires eval-time code changes
5. **Higher LR + longer warmdown** is a productive combo
6. **Lower LR reduces quant penalty** — there's a tradeoff between training quality and compression
7. **Context length** (2048/4096) helps but costs throughput; local M2 Pro will be slower
8. **TTT is a contestant-implemented strategy**: See LoRA TTT record analysis above for details. Prioritize strided/sliding window eval first (bigger win), then consider adding LoRA TTT on top.

## Ideas queue

Ideas to try in future experiments. Remove when tried or invalidated.

**High priority (proven wins from records):**
- ~~Sliding window eval~~ (done, exp 21)
- Int6 quantization for middle layers — better zlib compression. See 10L_MixedPrecision record: round int8 to nearest step=4, giving 64 levels.
- ~~Overtone spectral embedding init~~ (exp 22 — 0.023 worse)
- ~~Phase-transition residual mixing init~~ (exp 23 — 0.022 worse, needs long training)

**Medium priority:**
- NTK-RoPE extrapolation at eval time (EVAL_SEQ_LEN > TRAIN_SEQ_LEN). See WarmdownQuantization record.
- Late-K passthrough (last 2 layers' c_k.weight in fp16 instead of int8)
- ~~WD on scalar params~~ (exp 24 — 0.008 BPB, kept)
- ~~Embed LR ratio~~ (exp 25 — no change)

**Already tried/invalidated (do not re-try):**
- ~~Extended warmdown~~: warmdown=1200 already puts entire training in warmdown on M2 Pro (LR_mul≈0.14). Reducing warmdown to 400 was worse. Current schedule is effectively "always decaying".
- ~~LR above 0.30/0.30/0.35~~: LR=0.40 was worse (exp 14). 0.30 is optimal with current warmdown.
- ~~Muon WD above 0.32~~: WD=0.64 increased quant penalty too much (exp 18). 0.32 is optimal.
- ~~FP16 embed~~: Benefit too small (~0.0003) to measure in smoke tests. Save for H100 production.
- ~~10 layers~~: Too slow on M2 Pro (158 vs 175 steps). Note for H100 production.
- ~~Shorter momentum warmup~~: Higher momentum hurt in short-training regime (exp 12).

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

**Current best**: val_bpb=1.9192, artifact=8.30MB.
**Progress**: 2.4294 → 1.9192 = 0.510 BPB over 25 experiments.


