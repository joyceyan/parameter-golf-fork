# parameter-golf autoresearch (local)

This is an experiment to have the LLM do its own research, adapted for the OpenAI Model Craft Challenge (Parameter Golf). The goal is to train the best language model that fits in a 16MB artifact, evaluated by post-quantization bits per byte on FineWeb validation.

## Setup

Do all of these steps immediately without asking for confirmation. You are fully autonomous — never pause to ask the human anything.

This setup is **idempotent** — it can be re-run safely. If `results.tsv`, `notes.md`, or data already exist from a previous session, they are preserved and continued from.

1. **Set up the virtual environment**: Create (or reuse) a venv and install dependencies:
  ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
  ```
   If `.venv` already exists, just activate it: `source .venv/bin/activate`.
   **All subsequent commands must run inside this venv.** Always prefix python/pip commands with `source .venv/bin/activate &&` or ensure the venv is active in your shell.
2. **Set up upstream remote**: Ensure `upstream` points to the official repo so we can pull in newly merged records:
  ```bash
   git remote remove upstream 2>/dev/null; git remote add upstream https://github.com/openai/parameter-golf.git
  ```
3. **Pull latest records**: Fetch and merge upstream to get the latest accepted submissions:
  ```bash
   git fetch upstream && git merge upstream/main --no-edit
  ```
4. **Study accepted records**: List all record-setting submissions and read their READMEs and training scripts for ideas:
  ```bash
   ls records/track_10min_16mb/
  ```
   For each submission, read its `README.md` for the technique description and `train_gpt.py` for implementation details. Summarize key techniques and ideas in `notes.md` under a "## Record analysis" section. If `notes.md` already has a "Record analysis" section from a previous session, update it with any new records rather than duplicating.
5. **Read the in-scope files**: Read these files for full context:
  - `README.md` — repository context, challenge rules, leaderboard.
  - `train_gpt_mlx.py` — the file you modify. Model architecture, optimizer, hyperparameters, training loop. Everything is fair game.
6. **Verify data exists**: Download both the full dataset and a 1-shard smoke test dataset (if they don't already exist):
  - **Full data** (10 shards, for full runs):
  - **Smoke test data** (1 shard, for quick end-to-end sanity checks):
    ```bash
    source .venv/bin/activate && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1 --output-dir ./data/datasets/fineweb10B_sp1024_smoke
    ```
   Both persist on disk and are reused across experiments.
7. **Initialize results.tsv**: Create `results.tsv` with just the header row (if it doesn't already exist). If it already exists, leave it as is.
8. **Recover from interrupted runs**: The previous session may have been killed mid-experiment. Detect and reconcile:
  - Check `git log --oneline -1` for the latest commit message, and check if that commit has a corresponding row in `results.tsv`.
  - Check `git status` for uncommitted changes to `train_gpt_mlx.py`.
  - Check if `run.log` or `smoke.log` exist with partial output.
   **If there's a committed experiment with no matching TSV entry** (i.e. the run was interrupted before results were recorded):
  - Check if `run.log` contains `final_int8_zlib_roundtrip_exact`. If yes, the run actually completed — extract the results, record them in the TSV and `notes.md`, and proceed normally.
  - If `run.log` is incomplete or missing, ask yourself: does this experiment look promising based on partial logs or the commit message? If so, **re-run it** from the committed state. If not, `git reset --hard HEAD~1` to roll it back.
   **If there are uncommitted changes**: These are likely mid-edit changes from an interrupted session. `git checkout -- train_gpt_mlx.py` to discard them and start clean from the last committed state.
   **If everything is clean** (latest commit matches latest TSV entry, no uncommitted changes): no recovery needed.
9. **Check for existing progress**: Read `results.tsv` and `notes.md` (if they exist).
  - If a baseline already exists in `results.tsv`, skip the baseline run and go straight to the experiment loop.
  - If no baseline exists, run the baseline as the first experiment.
  - Review `notes.md` for prior insights, the current best val_bpb, and any queued ideas from previous sessions.

## Experimentation

Each experiment runs locally on a **MacBook with M2 Pro** (16GB unified memory) using MLX. The training script runs for its default iteration limit (20K steps) with no wallclock cap. You launch it as:

```bash
source .venv/bin/activate && MAX_WALLCLOCK_SECONDS=0 python train_gpt_mlx.py > run.log 2>&1
```

Setting `MAX_WALLCLOCK_SECONDS=0` disables the default 10-min wallclock cap, letting the script run to its natural iteration limit. On the M2 Pro this may take several hours per experiment — plan accordingly.

**What you CAN do:**

- Modify `train_gpt_mlx.py` — this is the primary file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, quantization strategy, etc.
- Train a new tokenizer with a different vocab size and re-tokenize the data. See the **Tokenizer experiments** section below.

**What you CANNOT do:**

- Install new packages or add dependencies. You can only use what's already available (`mlx`, `numpy`, `sentencepiece`, `tqdm`, `huggingface-hub`, `datasets`, and Python stdlib).
- Modify the data pipeline or evaluation logic in ways that would be invalid for the challenge.

**Local vs. final evaluation**: We experiment locally on M2 Pro / MLX, but the **final submission is evaluated on 8xH100s using PyTorch** (`train_gpt.py`, not `train_gpt_mlx.py`) with a 10-minute time budget. Keep this in mind:

- **Architecture and hyperparameter changes transfer well** between platforms — these are the most valuable experiments to run locally.
- **MLX-specific tricks won't help** the final submission. Don't optimize for MLX internals.
- **Memory constraints differ**: M2 Pro has 16GB unified memory vs. 80GB per H100. Don't self-censor ideas that would OOM locally but fit easily on H100 — note them in the lab notebook for later. Conversely, don't assume ideas that fit in 16GB will be the best use of 80GB.
- **Local runs take much longer** than 8xH100. The same 20K iterations might take hours on M2 Pro vs. ~10 min on H100. We let runs complete naturally to get the best signal. Use local runs to validate that an idea *helps*, not to match exact final scores.

**The goal is simple: get the lowest post-quantization val_bpb** (the `final_int8_zlib_roundtrip_exact` metric). This is the metric that matters for the challenge, since submissions are evaluated on compressed artifacts. Raw training val_bpb is a useful signal during training but the roundtrip metric is ground truth.

**Artifact size** is a hard constraint. The compressed model (`.int8.ptz`) plus code must fit in **16MB (16,000,000 bytes)**. If an experiment produces an artifact that exceeds 16MB, it must be discarded regardless of val_bpb.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

## Research methodology

Follow these principles to experiment like a real researcher:

**Isolate variables**: Make exactly one change per experiment. If you change the LR and the activation function at the same time, you won't know which one mattered. This is already enforced in the experiment loop (step 2), but take it seriously — even "small" secondary tweaks muddy the signal.

**Ablations**: When a complex change improves results, run follow-up experiments that strip parts away to find the essential ingredient. Often half the code can be deleted while keeping the gain. Ablations are some of the most valuable experiments you can run.

**Read the training dynamics**: Don't just look at the final val_bpb. After each run, grep the training log for loss over time:

```bash
grep "train_loss" run.log
```

Ask yourself: Was loss still decreasing at the end? (→ train longer or the idea has more room). Did it spike mid-training? (→ LR too high or instability). Plateau early? (→ architecture bottleneck, not a training problem). Record these observations in `notes.md` — they're often more informative than the final number.

**Reproduce before innovating**: If an upstream merged record beats your current best, try to reproduce their approach in `train_gpt_mlx.py` first. Start from a working reproduction, then improve incrementally. Don't reinvent from scratch what someone else has already validated.

**Variance awareness**: A 0.001 improvement might just be noise. For borderline results (improvements < 0.005), consider running the same config 2-3 times to confirm the gain is real before keeping it. Not every experiment needs this, but don't celebrate noise.

**Diminishing returns**: If the last 5 experiments were all minor hyperparameter tweaks (LR, warmup steps, weight decay) with tiny deltas, stop tuning and try something architecturally different. Check `notes.md` for this pattern — it's a signal to change strategy, not refine further.

**Profiling**: Periodically check where wall-clock time is going. If data loading is 40% of the time, optimizing the model architecture won't help throughput. If forward pass dominates, maybe the model is too wide. Add timing observations to `notes.md` to inform future experiments.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is (default 1024-vocab tokenizer).

## Tokenizer experiments

The default tokenizer is a 1024-vocab SentencePiece BPE model. You may experiment with different vocab sizes (e.g. 512, 2048, 4096). The val_bpb metric is tokenizer-agnostic (bits per byte), so different tokenizers are fairly comparable.

**Tradeoff**: Larger vocab = fewer tokens per byte (potentially easier to predict) but more embedding parameters eating into the 16MB artifact budget. Smaller vocab = more tokens per byte but smaller embeddings.

**To create a new tokenizer and dataset:**

1. Edit `data/tokenizer_specs.json` to add a new entry:
  ```json
   {"name": "sp_bpe_2048", "dataset_suffix": "sp2048", "vocab_size": 2048}
  ```
2. Run the tokenizer training and data re-tokenization:
  ```bash
   source .venv/bin/activate && python3 data/download_hf_docs_and_tokenize.py \
     --repo-id willdepueoai/parameter-golf \
     --remote-root datasets \
     --output-root ./data \
     --tokenizer-config ./data/tokenizer_specs.json
  ```
   This creates `./data/tokenizers/fineweb_2048_bpe.model` and `./data/datasets/fineweb10B_sp2048/`.
3. Run training with the new tokenizer:
  ```bash
   source .venv/bin/activate && \
   VOCAB_SIZE=2048 \
   TOKENIZER_PATH=./data/tokenizers/fineweb_2048_bpe.model \
   DATA_PATH=./data/datasets/fineweb10B_sp2048 \
   MAX_WALLCLOCK_SECONDS=0 python train_gpt_mlx.py > run.log 2>&1
  ```
4. You must also update `VOCAB_SIZE` in `train_gpt_mlx.py` defaults (or the Hyperparameters class) if you want the tokenizer change to persist across runs without env vars. The script validates that `VOCAB_SIZE` matches the tokenizer at startup.

**Important**: Tokenizer creation + re-tokenization takes significant time. Only attempt tokenizer experiments if you believe the vocab size change will meaningfully help. Once you have a new tokenizer/dataset cached, subsequent runs reuse it.

**Reverting tokenizer experiments**: If a tokenizer experiment doesn't improve results, `git reset --hard HEAD~1` reverts `train_gpt_mlx.py` changes, but the generated tokenizer/dataset files remain on disk (they're gitignored). This is fine — they can be reused later. Make sure to switch back to the correct `VOCAB_SIZE`/`TOKENIZER_PATH`/`DATA_PATH` env vars (or defaults) when reverting.

## Output format

Once the script finishes it prints results including:

```
serialized_model_int8_zlib:XXXXX bytes (payload:... raw_pickle:... payload_ratio:...x)
final_int8_zlib_roundtrip val_loss:X.XXXX val_bpb:X.XXXX eval_time:XXXms
final_int8_zlib_roundtrip_exact val_loss:X.XXXXXXXX val_bpb:X.XXXXXXXX
```

You can extract the key metrics from the log file:

```bash
grep "final_int8_zlib_roundtrip_exact" run.log
grep "serialized_model_int8_zlib" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 8 columns:

```
started_pt	commit	roundtrip_val_bpb	artifact_mb	duration_min	vocab	status	description
```

1. start time in Pacific Time, format `YYYY-MM-DD HH:MM` (e.g. `2026-03-19 22:15`)
2. git commit hash (short, 7 chars)
3. roundtrip val_bpb achieved (from `final_int8_zlib_roundtrip_exact`) — use 0.000000 for crashes
4. artifact size in MB, round to .2f (divide `serialized_model_int8_zlib` bytes by 1,000,000) — use 0.0 for crashes
5. duration of the full run in minutes, round to .1f — use 0.0 for crashes
6. vocab size used (e.g. 1024, 2048) — helps track tokenizer experiments
7. status: `keep`, `discard`, or `crash`
8. short text description of what this experiment tried

Example:

```
started_pt	commit	roundtrip_val_bpb	artifact_mb	duration_min	vocab	status	description
2026-03-19 22:15	a1b2c3d	1.224400	5.12	185.3	1024	keep	baseline
2026-03-20 01:25	b2c3d4e	1.218300	5.15	190.1	1024	keep	increase LR to 0.04
2026-03-20 04:40	c3d4e5f	1.230000	5.10	188.7	1024	discard	switch to GeLU activation
2026-03-20 07:52	d4e5f6g	0.000000	0.0	0.0	1024	crash	double model width (OOM)
2026-03-20 07:55	e5f6g7h	1.195000	6.80	195.2	2048	keep	try 2048-vocab tokenizer
```

## Lab notebook (notes.md)

Maintain a file called `notes.md` as a persistent lab notebook. This is critical for retaining context across experiments, since conversation history gets compressed over time.

**Read `notes.md` at the start of every experiment loop iteration** (step 1). This ensures you never re-try a failed idea or forget what you've learned.

**Append to `notes.md` after every experiment** (step 10), regardless of whether it succeeded, failed, or crashed. Each entry should include:

- The experiment description and hypothesis (what you expected and why)
- The result (roundtrip val_bpb, artifact size, status)
- **Qualitative insights**: what you learned, why it worked or didn't, what it suggests for future experiments
- Any observations about training dynamics (loss curves, instability, plateaus)
- Ideas sparked by this result that you want to try later

Example entry:

```markdown
### Experiment 3: Increase MLP expansion from 2x to 3x (2026-03-19 23:30)
- **Hypothesis**: Wider MLP might capture more complex token interactions
- **Result**: roundtrip_val_bpb=1.2280 (worse than 1.2244 baseline), artifact=6.8MB, DISCARDED
- **Insight**: The extra parameters hurt more than they helped — likely because the model can't train enough in 30 min to use them. The artifact size jumped significantly too. Wider MLP is probably only viable if we reduce layers to compensate.
- **Future idea**: Try 2x MLP with fewer layers (e.g. 7 layers instead of 9) to see if depth vs. width tradeoff matters at this scale.
```

Keep entries concise but substantive. The notebook should grow into a useful reference, not a dump of raw numbers (that's what `results.tsv` is for).

## The experiment loop

The experiment runs on `main`. All kept experiments are committed and pushed directly to `main`.

LOOP FOREVER:

1. **Review context**: Read `notes.md` and `results.tsv` to review what's been tried, what worked, and what to try next. Check the git state. Every ~5 experiments, also pull the latest upstream records for fresh ideas:
  ```bash
   git fetch upstream && git merge upstream/main --no-edit
  ```
   If new records appeared in `records/track_10min_16mb/`, read their `README.md` and `train_gpt.py` and add key techniques to the "Record analysis" section of `notes.md`.
2. **Design the next experiment** based on insights from the notebook. Make **exactly one change** per experiment so you can isolate what helped or hurt. The change can be either:
  - An edit to `train_gpt_mlx.py` (architecture, hyperparameters, optimizer, etc.), OR
  - A tokenizer experiment (new vocab size — see **Tokenizer experiments** section)
   Do not combine multiple changes in a single experiment. If you want to test a new LR *and* a new activation function, run them as two separate experiments.
3. `git commit -am "description of experiment"`
4. **Record start time**: Save the current time for tracking total experiment duration (smoke test + full run + overhead):
  ```bash
   TZ=America/Los_Angeles date "+%Y-%m-%d %H:%M"
  ```
5. **Smoke test**: Run a quick end-to-end check using the 1-shard dataset. This completes the full pipeline (train → quantize → evaluate) so you get a real directional signal on roundtrip val_bpb, not just a crash check:
  ```bash
   source .venv/bin/activate && DATA_PATH=./data/datasets/fineweb10B_sp1024_smoke python train_gpt_mlx.py > smoke.log 2>&1
  ```
   If using a non-default tokenizer, also set `VOCAB_SIZE` and `TOKENIZER_PATH` env vars.
  - Check if it completed: `grep "final_int8_zlib_roundtrip_exact" smoke.log`. If empty, read `tail -n 50 smoke.log` and fix the issue before proceeding.
  - Note any observations from the smoke test (e.g. training loss trajectory, directional val_bpb, obvious instability) in `notes.md`.
  - If the smoke test crashes and the idea is fundamentally broken, skip straight to step 13 (log crash in TSV and notes.md, revert).
6. **Full run**: If the smoke test passed, run the full experiment (no wallclock cap — runs to iteration limit):
  ```bash
   source .venv/bin/activate && MAX_WALLCLOCK_SECONDS=0 python train_gpt_mlx.py > run.log 2>&1
  ```
   If using a non-default tokenizer, also set `VOCAB_SIZE`, `TOKENIZER_PATH`, and `DATA_PATH` env vars (see **Tokenizer experiments** section).
7. Read out the results:
  ```bash
   grep "final_int8_zlib_roundtrip_exact" run.log
   grep "serialized_model_int8_zlib" run.log
  ```
8. **Read training dynamics**: Check the loss trajectory, not just the final number:
  ```bash
   grep "train_loss" run.log
  ```
   Note: Was loss still decreasing at the end? Did it spike or plateau? These observations go in `notes.md` and inform future experiments.
9. If the grep output in step 7 is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.
10. Check artifact size: extract the byte count from `serialized_model_int8_zlib:XXXXX bytes`. If it exceeds 16,000,000, the experiment fails the size constraint — discard it regardless of val_bpb.
11. Record the results in the TSV. Use the start time from step 4, and compute `duration_min` as total elapsed minutes from step 4 to now (covering smoke test + full run + overhead).
12. **Update `notes.md`**: Append an entry with the hypothesis, result, qualitative insights, training dynamics observations (from step 8), and ideas for future experiments. Include observations from both the smoke test and the full run. Do this for every experiment — successes, failures, and crashes all contain useful information.
  - For borderline improvements (< 0.005 val_bpb), note that the result may be noise and consider re-running to confirm.
    - If the last ~5 experiments have been minor hyperparameter tweaks with tiny deltas, note that it's time to try something architecturally different.
13. **Keep/discard decision** based on `roundtrip_val_bpb` (lower is better):
  - If roundtrip val_bpb **improved** (lower) AND artifact is under 16MB: keep the commit, advance the branch.
    - If roundtrip val_bpb is **equal or worse**, OR artifact exceeds 16MB: `git reset --hard HEAD~1` to revert.
    - For borderline improvements (< 0.005), consider re-running the same config to confirm the gain is real before keeping.
14. Go back to step 1.

**Timeout**: Runs complete at the default iteration limit (20K steps) with no wallclock cap. On the M2 Pro this may take several hours. If a run appears stuck (no new log output for 30+ minutes), kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the TSV, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read the README for new angles, study what the leaderboard winners did, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Full runs may take several hours each on the M2 Pro, so expect only a handful of experiments overnight. The user then wakes up to experimental results, all completed by you while they slept!