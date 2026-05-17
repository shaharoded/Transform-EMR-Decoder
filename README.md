# autoresearch — EMR Event Prediction

Autonomous hyperparameter and architecture search for my thesis's model, adapted from Karpathy's [autoresearcher](https://github.com/karpathy/autoresearch) repo.

## What this is

An AI agent autonomously experiments on an EMR (Electronic Medical Record) complication-prediction model overnight. Each experiment modifies files under `emr_model/transform_emr/`, trains for a fixed epoch budget, checks if the held-out evaluation metric improved, keeps or discards, and repeats — logging every result in `results.tsv`.

The architecture is a three-phase pipeline (`emr_model/`):

- **Phase 1 — EMREmbedding** — learns compact, time-aware representations of clinical events using hierarchical token embeddings, Time2Vec, and static patient context.
- **Phase 2 — GPT Transformer** — a causal decoder over Phase-1 embeddings; predicts the next clinical event and learns outcome timing via a curriculum of auxiliary losses.
- **Phase 3 — Outcome Fine-tuning** — backbone frozen; outcome head fine-tuned on natural-distribution data to sharpen complication risk scores.

The training data is derived from MIMIC-III and contains diabetes patients' longitudinal event sequences including lab results, vitals, diagnoses, medications, meals, and outcome events (complications, death, release).

## Files that matter

| File | Role |
|------|------|
| `api.py` | **Fixed.** Data loading, training orchestration, final evaluation. Do NOT modify. |
| `evaluation.py` | **Fixed.** Post-training evaluation metrics (AUROC/AUPRC/MAE). Do NOT modify. |
| `emr_model/transform_emr/config/model_config.py` | **Agent edits this.** `MODEL_CONFIG` (architecture dims) and `TRAINING_SETTINGS` (hyperparameters, schedulers). |
| `emr_model/transform_emr/` | Model source — agent may modify for architecture changes. |
| `program.md` | **Human edits this.** Instructions and research context for the autonomous agent. |
| `emr_model/data/source/` | Training data (fixed). |
| `results.tsv` | Experiment log (gitignored, untracked by git). |

## Metrics

All metrics are computed by `evaluation.py::evaluate_on_test_set` on the held-out validation set via autoregressive generation — never modified by the agent.

**Primary — `outcome_auroc` (higher is better)**
Mean per-complication AUROC from pooled episode-level AUC. For each generated trajectory, time is divided into 24-hour non-overlapping windows; a window is labelled positive if any ground-truth episode of that complication falls within ±24h of the window edges. AUROC is computed from (window, score) pairs pooled across all patients, then averaged across complications with at least 3 positive windows. Random = 0.5, perfect = 1.0.

**Secondary — `outcome_auprc` (higher is better)**
Mean per-complication average precision (AUPRC) from the same pooled window evaluation. Reflects precision across recall thresholds — more sensitive to false alarms than AUROC.

**Tertiary — `onset_mae_hrs` (lower is better)**
Mean absolute error between the predicted onset time (generated step with peak `P_outcome`) and the ground-truth first occurrence of that complication, in hours. Averaged across patients where the complication occurred.

## Design choices

- **Immutable contract.** `api.py` and `evaluation.py` are the fixed ground truth. The agent edits `model_config.py` (hyperparameters) and files under `emr_model/transform_emr/` (architecture).
- **Embedder caching.** Phase 1 is skipped automatically when `(embed_dim, time2vec_dim, ctx_dim)` are unchanged — saving ~30 min per experiment.
- **Fresh Phase 2 and Phase 3 per experiment.** Those checkpoints are cleared before each run so experiments are independent.
- **Phase 3 for outcome alignment.** The backbone is frozen and only the outcome head is fine-tuned on natural-distribution data — prevents oversampling bias from contaminating risk scores.
- **Generation-based evaluation.** The final metrics are computed from autoregressive generation (not teacher-forced logits), matching real clinical deployment: the model generates a trajectory from 2 days of seed data and its outcome-head risk scores are evaluated against ground-truth future episodes.
- **Epoch budget with early stopping.** Training terminates when validation stops improving (patience configurable), bounded by a maximum epoch count.
- **Data sampling.** `TRAINING_SETTINGS["sample"]` controls how many patients to use. Set to `None` for full training runs; set to a small integer (e.g. `50`) for quick smoke-tests.

## Project structure

```
api.py                  fixed: training orchestration (do not modify)
evaluation.py           fixed: evaluation metrics (do not modify)
program.md              agent instructions (human modifies)
analysis.ipynb          experiment analysis / visualisation
pyproject.toml          dependencies
results.tsv             experiment log (gitignored, untracked)
run.log                 last experiment output (gitignored)
emr_model/
  transform_emr/
    config/
      model_config.py   MODEL_CONFIG + TRAINING_SETTINGS (agent modifies)
      dataset_config.py data paths and special tokens (fixed)
    embedder.py         Phase-1 EMREmbedding
    transformer.py      Phase-2/3 GPT + finetune_transformer
    dataset.py          tokenizer and dataloader
    loss.py             loss functions
    schedulers.py       auxiliary loss curriculum scheduling
    inference.py        autoregressive generation (used by evaluation.py)
    utils.py            masking, targets, penalties
    diagnose.py         model health checks
  data/
    source/
      temporal_data.csv      patient event sequences
      context_data.csv       patient context features
  checkpoints/          saved model weights (gitignored)
```

---

## End-to-end workflow

The agent runs autonomously on a RunPod GPU pod (A40 / A5000, 24–48 GB VRAM). The pattern that works reliably:

- The pod runs Claude Code inside `tmux` so it survives SSH drops.
- A non-root `agent` user runs Claude (the `--dangerously-skip-permissions` flag refuses to run as root).
- The agent commits experiments locally only. **Root pushes to GitHub** periodically; you `git pull origin main` on your own machine to read the latest `status.md` / `results.tsv` / code.
- The agent maintains a running `status.md` at the repo root (overwritten every meaningful step), which is the canonical source of progress.

### One-time local setup

Generate a key for the pod (run once on your local Windows machine):

```powershell
ssh-keygen -t ed25519 -C "runpod" -f "$env:USERPROFILE\.ssh\id_ed25519"
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub" | clip
```

Paste it into **RunPod → Settings → SSH Public Keys**, and also into **GitHub → Settings → SSH and GPG keys → New SSH key** (so the pod can both be SSH-targeted and push to GitHub from inside).

---

### First-time pod bring-up (per pod)

RunPod gives you the SSH command on the pod's Connect page — looks like `ssh root@<HOST> -p <PORT>`. Use it from local PowerShell:

```powershell
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519
```

On the pod, install tooling and clone the repo (~5 min total):

```bash
# tmux for persistent sessions
apt-get update && apt-get install -y tmux

# Node 20 (Claude Code requires Node 18+; the default Node 12 has a conflict)
apt-get remove --purge -y nodejs libnode-dev libnode72 2>/dev/null
apt-get autoremove -y
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
npm install -g @anthropic-ai/claude-code

# Clone the repo (works because root's SSH key is registered with GitHub)
cd /workspace
git clone git@github.com:shaharoded/AutoResearcher-TransformEMR.git autoresearch
cd autoresearch
pip install -e .
pip install scikit-learn tqdm openpyxl joblib matplotlib pandas pyarrow

# Drop the data CSVs into place — SCP them from local (see "Copying data" below)
mkdir -p emr_model/data/source
# scp emr_model/data/source/temporal_data.csv and context_data.csv from local

# Create a non-root user for the agent (the --dangerously-skip-permissions flag won't run as root)
useradd -m -s /bin/bash agent
mkdir -p /home/agent/.ssh
cp /root/.ssh/id_ed25519 /root/.ssh/id_ed25519.pub /home/agent/.ssh/
chmod 700 /home/agent/.ssh
chmod 600 /home/agent/.ssh/id_ed25519
chmod 644 /home/agent/.ssh/id_ed25519.pub
chmod -R a+rwX /workspace/autoresearch     # network volume blocks chown; chmod works
```

#### Copying data to the pod

The training CSVs are gitignored. SCP them from your local repo (run in local PowerShell):

```powershell
scp -P <PORT> -i ~/.ssh/id_ed25519 emr_model\data\source\temporal_data.csv root@<HOST>:/workspace/autoresearch/emr_model/data/source/
scp -P <PORT> -i ~/.ssh/id_ed25519 emr_model\data\source\context_data.csv  root@<HOST>:/workspace/autoresearch/emr_model/data/source/
```

---

### Daily workflow

#### Starting the agent

Always SSH in as root, switch to `agent`, then attach a tmux session:

```bash
# Local PowerShell:
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519

# On the pod:
su - agent
cd /workspace/autoresearch
git config --global --add safe.directory /workspace/autoresearch   # needed because of network-volume ownership

tmux new -s claude
claude --dangerously-skip-permissions
```

On first launch Claude will prompt for a browser login — paste the URL into your local browser, log in, paste the token back. Then in the Claude prompt:

> Read `program.md` and all files in its Setup section. Check `results.tsv` and `git log --oneline -10` for state. Begin the experiment loop now. Update `status.md` after every meaningful step and commit it locally; do NOT push — root will push.

Detach the tmux session with `Ctrl+B` then `D`. The agent keeps running.

#### Reconnecting after an SSH drop

The tmux session survives SSH disconnects. Just re-SSH and reattach:

```bash
# Local:
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519

# On the pod:
su - agent
cd /workspace/autoresearch
tmux ls                            # confirm 'claude' is listed
tmux attach -t claude              # or `-d` to detach any other viewer
```

If `tmux ls` says "no server running", the session is gone (pod was restarted). Start fresh with `tmux new -s claude && claude --dangerously-skip-permissions`.

#### Watching progress externally

The agent commits locally only — `status.md` is the canonical progress file. As **root** in `/workspace/autoresearch`, push to GitHub whenever you want a sync point:

```bash
# As root on the pod (root has working GitHub SSH auth):
cd /workspace/autoresearch
git push origin main
```

Then locally, fetch and read:

```powershell
git fetch origin
git pull --ff-only origin main
cat status.md
```

If `git pull` complains about local `status.md` changes, move it aside first:
```powershell
move status.md $env:TEMP\status_local_old.md
git pull --ff-only origin main
```

#### Running an experiment yourself

You don't usually need to; the agent does it. If you want to verify the pipeline manually:

```bash
# As agent in /workspace/autoresearch:
python api.py > run.log 2>&1
grep "^outcome_auroc:\|^outcome_auprc:\|^onset_mae_hrs:\|^peak_vram_mb:" run.log
```

Smoke test first by setting `sample=50, phase{1,2,3}_n_epochs=1` in `emr_model/transform_emr/config/model_config.py`.

---

### Pausing / stopping a session

The agent pauses on its own when `program.md`'s stop criterion is met. To stop manually:
- Reattach the tmux session (`tmux attach -t claude`).
- Send `/exit` to Claude, or interrupt with `Esc` then exit.
- Detach (`Ctrl+B D`).
- Kill the tmux session when done: `tmux kill-session -t claude`.

### Stopping the pod (pause billing)

The pod's **container disk** is ephemeral — wiped on stop unless you used a Network Volume. Before stopping, decide what to preserve:

| What | Where it lives | Survives pod stop? |
|---|---|---|
| Code, `program.md`, `status.md` | Git (push to origin) | Yes — via GitHub |
| `results.tsv` | Container disk (gitignored) | **No** — SCP it off if you want the history |
| Phase-1/2/3 checkpoints | Container disk | **No** — SCP if you want to resume the trained model |
| Tokenizer cache | Container disk | **No** — slow to rebuild (~minutes) |

To save the artifacts before stopping (run in local PowerShell):

```powershell
$DST = "$env:USERPROFILE\runpod_backup\$(Get-Date -Format yyyy-MM-dd)"
New-Item -ItemType Directory -Force -Path $DST

# Experiment log
scp -P <PORT> -i ~/.ssh/id_ed25519 root@<HOST>:/workspace/autoresearch/results.tsv $DST\

# Best-model checkpoints (skip if you'll retrain on different data anyway)
scp -P <PORT> -i ~/.ssh/id_ed25519 root@<HOST>:/workspace/autoresearch/emr_model/checkpoints/phase3/ckpt_best.pt $DST\phase3_ckpt_best.pt
scp -P <PORT> -i ~/.ssh/id_ed25519 root@<HOST>:/workspace/autoresearch/emr_model/checkpoints/phase2/ckpt_best.pt $DST\phase2_ckpt_best.pt
scp -P <PORT> -i ~/.ssh/id_ed25519 root@<HOST>:/workspace/autoresearch/emr_model/checkpoints/phase1/ckpt_best.pt $DST\phase1_ckpt_best.pt
scp -P <PORT> -i ~/.ssh/id_ed25519 root@<HOST>:/workspace/autoresearch/emr_model/checkpoints/tokenizer.pt $DST\tokenizer.pt
```

Then **RunPod → My Pods → Stop** (not Terminate — Stop pauses billing but keeps the pod definition).

If you used a **Network Volume** mounted at `/workspace`, everything persists across stops and SCP is unnecessary.

---

### Common gotchas

- **`chown` fails on `/workspace`** — RunPod's network volume disallows ownership changes. Use `chmod -R a+rwX /workspace/autoresearch` instead. Run as root after pulling new files.
- **`fatal: detected dubious ownership`** when `agent` runs git in `/workspace` — fix once: `git config --global --add safe.directory /workspace/autoresearch`.
- **Agent can't push to GitHub** — agent's SSH key may not have correct perms on the network volume. Workaround: agent commits locally only; root pushes. See the "Watching progress externally" section.
- **`--dangerously-skip-permissions cannot be used with root/sudo privileges`** — switch to the non-root `agent` user before launching Claude.
- **`git pull --ff-only` rejects** — usually the agent did a `git reset --hard` for a DISCARD that re-shaped local history, and origin moved forward via root push. Compare with `git log --oneline main..origin/main`; if you're sure remote is canonical, `git reset --hard origin/main`.
- **GitHub authenticity prompt** when `agent` runs git — pre-trust the host key:
  ```bash
  ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts
  ```
- **Old node 12 conflict** when installing Node 20 — first `apt-get remove --purge -y nodejs libnode-dev libnode72`, then install the nodesource Node 20 package.

---

## Running the agent (quick reference)

In the `claude` prompt, after `program.md` exists and the data is in place:

```
Read program.md and all files listed in its Setup section.
Check results.tsv and git log --oneline -10 for state.
Begin the experiment loop now. Update status.md after every meaningful step and commit locally; do NOT push — I will push from root.
```

The agent will iterate autonomously: smoke test → commit → full run → log row to `results.tsv` → KEEP/DISCARD per `program.md` rules → update `status.md` → repeat.
