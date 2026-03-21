# Autoresearch: Autonomous ML Experimentation

This repo runs an autonomous experiment loop for the OpenAI Parameter Golf challenge.
Read `program.md` for full setup, tokenizer experiments, research methodology, and record analysis.

## Critical constraints (never forget these)

- **Metric**: `final_int8_zlib_roundtrip_exact val_bpb` (lower is better). This is the ONLY metric that matters.
- **Artifact limit**: compressed model + code must be < **16,000,000 bytes** (16MB). Discard if exceeded.
- **One change at a time**: exactly one modification per experiment. Never combine changes.
- **Venv**: always use `source .venv/bin/activate &&` before python commands.
- **Smoke tests only**: run on 1-shard smoke data with default 10-min wallclock cap. No full dataset runs locally — directional signal only.
- **No new packages**: only use what's already installed (`mlx`, `numpy`, `sentencepiece`, `tqdm`, `huggingface-hub`, `datasets`, stdlib).

## Experiment loop checklist

Every iteration, follow these steps in order:

1. **Read** `notes.md` and `results.tsv`. Every ~5 experiments, `git fetch upstream && git merge upstream/main --no-edit` for new records.
2. **Design** one experiment. Edit `train_gpt_mlx.py` OR run a tokenizer experiment. Not both.
3. **Commit**: `git commit -am "description"`
4. **Record start time**: `TZ=America/Los_Angeles date "+%Y-%m-%d %H:%M"`
5. **Run** (1-shard smoke data, default 10-min wallclock cap):
   ```
   source .venv/bin/activate && DATA_PATH=./data/datasets/fineweb10B_sp1024_smoke python train_gpt_mlx.py > smoke.log 2>&1
   ```
6. **Extract results**:
   ```
   grep "final_int8_zlib_roundtrip_exact" smoke.log
   grep "serialized_model_int8_zlib" smoke.log
   ```
7. **Read training dynamics**: `grep "train_loss" smoke.log` — note trajectory in `notes.md`.
8. **Check crash**: if step 6 is empty, `tail -n 50 smoke.log`, fix or skip.
9. **Check artifact size**: must be < 16,000,000 bytes.
10. **Log to `results.tsv`** (tab-separated, 8 columns):
    ```
    started_pt	commit	roundtrip_val_bpb	artifact_mb	duration_min	vocab	status	description
    ```
11. **Update `notes.md`**: hypothesis, result, insights, training dynamics, future ideas.
12. **Keep/discard**: improved val_bpb AND < 16MB → keep. Otherwise → `git reset --hard HEAD~1`.
13. **Go to step 1. NEVER STOP.**

## Key principles

- **Isolate variables**: one change per experiment, always.
- **Ablate wins**: when something works, strip parts away to find the essential ingredient.
- **Read loss curves**: don't just check the final number — was loss still decreasing? spiking? plateauing?
- **Reproduce before innovating**: if an upstream record beats you, reproduce it first.
- **Variance awareness**: improvements < 0.005 might be noise. Re-run to confirm borderline results.
- **Diminishing returns**: if last 5 experiments were minor HP tweaks, change strategy.
- **Local ≠ production**: M2 Pro (16GB, MLX) is for directional signal only. Final eval is 8xH100 (PyTorch).
