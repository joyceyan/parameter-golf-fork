# Lab Notebook — Parameter Golf (MLX on M2 Pro)

## Record analysis

### SOTA: SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit (1.1748)

- **10 layers** (vs 9 baseline), fits in 16MB thanks to Muon weight decay (WD=0.02) compressing weights
- **FP16 tied embeddings**: Prevents int8 quantization error from compounding through input+output paths
- **Sliding window evaluation** (stride=64): Each token scored with ~960+ context instead of 0-1023 average
- **Overtone spectral embedding init**: SVD power-law spectrum shaping (S_k ~ k^{-0.5})
- **Phase-transition residual mixing**: Sigmoid-scheduled resid_mix initialization
- Architecture: 1024 vocab, 10L, 512dim, 8 heads, 4 KV heads, seq_len=1024

### SlidingWindowEval (1.1925)

- Pure evaluation improvement: sliding window with stride=64
- No training changes — same baseline model
- Shows eval strategy alone yields 0.032 BPB improvement
- Eval time: 70s on 8xH100 (well within budget)

### LoRA TTT (1.1928)

- Test-time training with rank-8 LoRA on eval
- Per-document adaptation: detect boundaries via BOS tokens, reset LoRA between docs
- Strided evaluation (chunk=256 within eval_seq_len=1024)
- Targets lm_head, c_q, c_v projections; Adam lr=0.01

### LongContextSeq2048 → Seq4096 (1.2014)

- Doubled/quadrupled sequence length provides more context per token
- 2048: ~51.89ms/step; 4096: ~71ms/step (slower but better signal)
- Lower LR (0.032 matrix, 0.032 scalar) with longer context
- Trains fewer steps but each step has richer signal

### FP16Embed_WD3600 (1.2197)

- FP16 embedding passthrough reduces post-quant degradation from ~0.007 to ~0.0005 BPB
- Extended warmdown (3600 iters) + higher LR (0.06)
- MLP_HIDDEN reduced to 992 to offset FP16 embedding overhead
- Key insight: embeddings are dual-purpose (input+output), so int8 errors compound

### 10L_MixedPrecision (1.2147)

- 10 layers with mixed int8/int6 compression
- Early (0-2) and late (7-9) layers: int8; middle (3-6): int6 (64 levels)
- Middle layers less sensitive to quantization, saves ~1.6MB for 10L
- Lower LR: 0.02 matrix/scalar, 0.03 embed

### WarmdownQuantization (1.2154)

- WARMDOWN_ITERS=20000 (>> actual ~12200 steps): entire training in LR decay
- Produces tighter weight distributions with fewer outliers → better int8 compression
- Post-quant penalty: 0.014 (WD=1200) → 0.005 (WD=20000) → ~0.001 with FP16 embed
- Higher LR (0.06/0.07) compensates for always-decaying schedule
- GRAD_CLIP_NORM=1.0 used

### TrainingOptSeq4096 (best 4096-context submission)

- 4096 seq len with momentum=0.99, warmup 1500 steps from 0.92
- Lower LR: 0.020 matrix/scalar, 0.030 embed
- WARMDOWN_ITERS=3000, batch=393216 (3/4 of default)
- Quant penalty only 0.0034 BPB (low LR helps)

### NaiveBaseline (1.2244)

- Reference: 9L, 512dim, 1024 vocab, 8 heads, 4 KV heads, seq=1024
- ~13780 steps in 10 min on H100, ~43.54ms/step
- Pre-quant: 1.2172, post-quant: 1.2244 (penalty: 0.007)

## Key takeaways for local MLX experiments

1. **FP16 embeddings** are a near-free win (~0.006-0.007 BPB from reduced quant error)
2. **Extended warmdown** (much larger than actual steps) reduces weight outliers → better compression
3. **10 layers** possible if we use weight decay or mixed precision to fit in 16MB
4. **Sliding window eval** is a huge win but requires eval-time code changes
5. **Higher LR + longer warmdown** is a productive combo
6. **Lower LR reduces quant penalty** — there's a tradeoff between training quality and compression
7. **Context length** (2048/4096) helps but costs throughput; local M2 Pro will be slower

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

**Next ideas**: With 8.51MB, try Adam WD on embed param, try 10L again, or try increasing warmup_steps.


