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



