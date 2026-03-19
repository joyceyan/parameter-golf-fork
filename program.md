# autoresearch (parameter-golf edition)

This is an experiment to have the LLM do its own research, adapted for the OpenAI Model Craft Challenge (Parameter Golf). The goal is to train the best language model that fits in a 16MB artifact, evaluated by post-quantization bits per byte on FineWeb validation.

## Usage

```bash
claude -p "$(cat program.md)" --dangerously-skip-permissions
```

This runs Claude Code in non-interactive mode with all permissions auto-approved, allowing fully autonomous operation. Claude will create a branch, establish a baseline, and loop experiments indefinitely until you kill the process (`Ctrl+C`).

## Setup

Do all of these steps immediately without asking for confirmation. You are fully autonomous — never pause to ask the human anything.

1. **Set up the virtual environment**: Create (or reuse) a venv and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
   ```
   If `.venv` already exists, just activate it: `source .venv/bin/activate`.
   **All subsequent commands must run inside this venv.** Always prefix python/pip commands with `source .venv/bin/activate &&` or ensure the venv is active in your shell.
2. **Pick a run tag**: use today's date (e.g. `mar19`). If `autoresearch/<tag>` already exists, append a suffix (e.g. `mar19b`).
3. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
4. **Read the in-scope files**: Read these files for full context:
   - `README.md` — repository context, challenge rules, leaderboard.
   - `train_gpt_mlx.py` — the file you modify. Model architecture, optimizer, hyperparameters, training loop. Everything is fair game.
5. **Verify data exists**: Check that `./data/datasets/fineweb10B_sp1024/` contains training shards and validation data. If not, download it:
   ```bash
   source .venv/bin/activate && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
   ```
6. **Initialize results.tsv**: Create `results.tsv` with just the header row.
7. **Immediately start the baseline run** — do not wait for confirmation.

## Experimentation

Each experiment runs locally on Apple Silicon (M2 Pro) using MLX. The training script runs for a **configurable time budget** controlled by `MAX_WALLCLOCK_SECONDS` (default 600s). For overnight runs, we use 1800s (30 minutes) to allow deeper training. You launch it as:

```bash
source .venv/bin/activate && MAX_WALLCLOCK_SECONDS=1800 python train_gpt_mlx.py > run.log 2>&1
```

**What you CAN do:**
- Modify `train_gpt_mlx.py` — this is the primary file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, quantization strategy, etc.
- Train a new tokenizer with a different vocab size and re-tokenize the data. See the **Tokenizer experiments** section below.

**What you CANNOT do:**
- Install new packages or add dependencies. You can only use what's already available (`mlx`, `numpy`, `sentencepiece`, `tqdm`, `huggingface-hub`, `datasets`, and Python stdlib).
- Modify the data pipeline or evaluation logic in ways that would be invalid for the challenge.
- Make the training script longer than 1500 lines (hard repo rule).

**The goal is simple: get the lowest post-quantization val_bpb** (the `final_int8_zlib_roundtrip_exact` metric). This is the metric that matters for the challenge, since submissions are evaluated on compressed artifacts. Raw training val_bpb is a useful signal during training but the roundtrip metric is ground truth.

**Artifact size** is a hard constraint. The compressed model (`.int8.ptz`) plus code must fit in **16MB (16,000,000 bytes)**. If an experiment produces an artifact that exceeds 16MB, it must be discarded regardless of val_bpb.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

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
   MAX_WALLCLOCK_SECONDS=1800 python train_gpt_mlx.py > run.log 2>&1
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

The TSV has a header row and 6 columns:

```
commit	roundtrip_val_bpb	artifact_mb	vocab	status	description
```

1. git commit hash (short, 7 chars)
2. roundtrip val_bpb achieved (from `final_int8_zlib_roundtrip_exact`) — use 0.000000 for crashes
3. artifact size in MB, round to .2f (divide `serialized_model_int8_zlib` bytes by 1,000,000) — use 0.0 for crashes
4. vocab size used (e.g. 1024, 2048) — helps track tokenizer experiments
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	roundtrip_val_bpb	artifact_mb	vocab	status	description
a1b2c3d	1.224400	5.12	1024	keep	baseline (30min M2 Pro)
b2c3d4e	1.218300	5.15	1024	keep	increase LR to 0.04
c3d4e5f	1.230000	5.10	1024	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	1024	crash	double model width (OOM)
e5f6g7h	1.195000	6.80	2048	keep	try 2048-vocab tokenizer
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar19`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `train_gpt_mlx.py` with an experimental idea by directly hacking the code.
3. `git commit -am "description of experiment"`
4. Run the experiment (redirect everything — do NOT use tee or let output flood your context):
   ```bash
   source .venv/bin/activate && MAX_WALLCLOCK_SECONDS=1800 python train_gpt_mlx.py > run.log 2>&1
   ```
   If using a non-default tokenizer, also set `VOCAB_SIZE`, `TOKENIZER_PATH`, and `DATA_PATH` env vars (see **Tokenizer experiments** section).
5. Read out the results:
   ```bash
   grep "final_int8_zlib_roundtrip_exact" run.log
   grep "serialized_model_int8_zlib" run.log
   ```
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.
7. Check artifact size: extract the byte count from `serialized_model_int8_zlib:XXXXX bytes`. If it exceeds 16,000,000, the experiment fails the size constraint — discard it regardless of val_bpb.
8. Record the results in the TSV.
9. **Keep/discard decision** based on `roundtrip_val_bpb` (lower is better):
   - If roundtrip val_bpb **improved** (lower) AND artifact is under 16MB: keep the commit, advance the branch.
   - If roundtrip val_bpb is **equal or worse**, OR artifact exceeds 16MB: `git reset --hard HEAD~1` to revert.
10. Go back to step 1.

**Timeout**: Each experiment should take roughly `MAX_WALLCLOCK_SECONDS` plus a few minutes for startup, compilation, quantization, and evaluation. If a run exceeds 60 minutes total, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the TSV, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read the README for new angles, study what the leaderboard winners did, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes ~35 minutes (30 min training + overhead), you can run about 2/hour, for a total of about 16 over a full night. The user then wakes up to experimental results, all completed by you while they slept!
