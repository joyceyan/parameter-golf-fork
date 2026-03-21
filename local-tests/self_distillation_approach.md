# Task: Add EMA self-distillation to train_gpt_submit.py

## Overview

Add a "Mean Teacher" EMA self-distillation loss to the existing training script. The EMA model (already implemented for export-time weight averaging) will also serve as a teacher during training, providing soft label targets via KL divergence. This requires one extra forward pass per step through the EMA weights (no backward pass).

## Architecture: Option C — separate compiled EMA model

Maintain a second `GPT` model instance that holds the EMA weights and stays in eval mode. After each optimizer step + EMA weight update, copy the fresh EMA state into this model. Use its `forward_logits` method for the teacher signal.

## New hyperparameters

Add these to the `Hyperparameters` class:

```python
ema_distill_alpha = float(os.environ.get("EMA_DISTILL_ALPHA", 0.0))  # 0.0 = disabled
ema_distill_start_step = int(os.environ.get("EMA_DISTILL_START_STEP", 500))
```

- `ema_distill_alpha`: Weight of the KL distillation loss. When 0.0, no distillation happens (pure EMA averaging for export only, same as current behavior). When > 0, the total loss becomes `(1 - alpha) * CE_loss + alpha * KL_loss`.
- `ema_distill_start_step`: Don't apply distillation before this step. In early training both model and EMA are near-random, so the EMA signal is noise.

## Implementation steps

### 1. Create the EMA teacher model (after warmup, before training loop)

Right after the existing `ema_state` initialization block and right before `training_time_ms = 0.0`, add:

```python
ema_teacher: nn.Module | None = None
if args.ema_enabled and args.ema_distill_alpha > 0.0:
    ema_teacher = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0,       # teacher doesn't need MTP heads
        mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
    ).to(device).bfloat16()
    for m in ema_teacher.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(ema_teacher)
    # Load initial EMA state (which at this point equals the model's init state)
    ema_teacher_sd = {k: v for k, v in ema_state.items() if "mtp_heads" not in k}
    ema_teacher.load_state_dict(
        {k: v.to(dtype=ema_teacher.state_dict()[k].dtype) for k, v in ema_teacher_sd.items()},
        strict=True,
    )
    ema_teacher.eval()
    ema_teacher = torch.compile(ema_teacher, dynamic=False, fullgraph=True)
    log0(f"ema_distill:enabled alpha={args.ema_distill_alpha} start_step={args.ema_distill_start_step} decay={args.ema_decay}")
```

### 2. Modify GPT.forward to accept optional EMA logits

Do NOT run the EMA forward inside `GPT.forward`. Instead, compute EMA logits in the training loop and pass them in. This keeps the model class clean and avoids issues with `torch.compile`.

Add an optional parameter to `GPT.forward`:

```python
def forward(self, input_ids: Tensor, target_ids: Tensor,
            ema_logits: Tensor | None = None, ema_distill_alpha: float = 0.0) -> Tensor:
```

After computing `main_loss` (the CE loss) and before the MTP block, add:

```python
        if ema_logits is not None and ema_distill_alpha > 0.0:
            # KL divergence: student tries to match EMA's output distribution
            # Compute in float32 for numerical stability
            student_log_probs = F.log_softmax(logits.float(), dim=-1)
            ema_probs = F.softmax(ema_logits.reshape(-1, ema_logits.size(-1)).float().detach(), dim=-1)
            kl_loss = F.kl_div(student_log_probs, ema_probs, reduction="batchmean")
            main_loss = (1.0 - ema_distill_alpha) * main_loss + ema_distill_alpha * kl_loss
```

**Important**: The `ema_logits` must already have `logit_softcap` applied (which `forward_logits` does). The `logits` variable at this point in the code also has softcap applied. So both sides of the KL are in the same space.

### 3. Modify the training loop to compute EMA logits and pass them in

In the inner micro-step loop, change:

```python
# BEFORE:
with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
    loss = model(x, y)

# AFTER:
ema_logits_for_step = None
if ema_teacher is not None and step >= args.ema_distill_start_step:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        ema_logits_for_step = ema_teacher.forward_logits(x)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
    loss = model(x, y, ema_logits=ema_logits_for_step, ema_distill_alpha=args.ema_distill_alpha)
```

**Note on DDP**: Since `model` might be a DDP-wrapped compiled model, the extra `ema_logits` and `ema_distill_alpha` kwargs need to pass through. DDP forwards kwargs to the underlying module, so this should work. But if `torch.compile` with `fullgraph=True` complains about the optional tensor being sometimes None and sometimes a Tensor, you may need to:
- Always pass a tensor (use a dummy zeros tensor when distillation is off), OR
- Conditionally call different compiled versions, OR
- Set `fullgraph=False` (least preferred, may hurt perf)

The safest approach if compile issues arise: always compute `ema_logits` as a zeros tensor when not distilling, so the compiled graph shape is consistent. The `alpha=0.0` path will make the KL term vanish mathematically.

### 4. Sync EMA weights to teacher after each EMA update

Right after the existing EMA update block:

```python
if ema_state is not None:
    d = args.ema_decay
    with torch.no_grad():
        for name, t in base_model.state_dict().items():
            ema_state[name].mul_(d).add_(t.detach().float(), alpha=1.0 - d)
```

Add:

```python
    # Sync EMA weights to the teacher model for next step's distillation
    if ema_teacher is not None:
        with torch.no_grad():
            teacher_sd = ema_teacher.state_dict() if not hasattr(ema_teacher, '_orig_mod') else ema_teacher._orig_mod.state_dict()
            for name, t in ema_state.items():
                if name in teacher_sd:
                    teacher_sd[name].copy_(t.to(dtype=teacher_sd[name].dtype))
```

**Note on torch.compile**: `ema_teacher` is compiled, so accessing its state dict requires going through `_orig_mod`. The `hasattr` check handles both compiled and non-compiled cases. An alternative is to keep a reference to the uncompiled model: `ema_teacher_raw = ema_teacher` before compiling, then `ema_teacher = torch.compile(ema_teacher_raw, ...)`, and use `ema_teacher_raw.state_dict()` for weight syncing.

### 5. Logging

Add a log line after the EMA teacher setup:

```python
log0(f"ema_teacher_params:{sum(p.numel() for p in ema_teacher.parameters())}")
```

In the training log, optionally log the KL loss magnitude separately for debugging. You can do this by returning a tuple from forward or by computing KL outside and logging it.

## What NOT to change

- Do NOT apply distillation loss to MTP heads. The `mtp_loss_weight` block stays as-is.
- Do NOT modify `forward_logits` — it already returns softcapped logits, which is what we want.
- Do NOT change the EMA export logic at the end of training — the EMA weights are still what gets serialized.
- Do NOT change the eval functions.
- Do NOT change the quantization pipeline.

## Environment variables for Run 2

```bash
EMA_ENABLED=1
EMA_DECAY=0.995
EMA_DISTILL_ALPHA=0.1
EMA_DISTILL_START_STEP=500
SWA_ENABLED=0
QAT_ENABLED=0
```

All other hyperparameters remain at their current defaults in the script.

## Testing checklist

Before running on H100:
1. Verify that with `EMA_DISTILL_ALPHA=0.0`, behavior is identical to the unmodified script (the code paths should be no-ops).
2. Verify that with `EMA_DISTILL_ALPHA=0.1`, the training loop runs without errors for at least 50 steps.
3. Verify the EMA teacher weight sync works by checking that after step N, the teacher's weights match `ema_state` (within dtype casting tolerance).
4. Verify that the KL loss term is a reasonable magnitude (should be roughly comparable to the CE loss, not orders of magnitude larger or smaller). Log it for the first 10 steps after `ema_distill_start_step`.
5. Confirm the extra forward pass doesn't cause CUDA OOM — the teacher forward should only allocate activations, no gradients.
