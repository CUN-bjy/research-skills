---
name: run-experiment
description: Guide through launching an AI/ML training experiment. Use when user asks to "run an experiment", "launch training", "start training", or needs help setting up ML training with optional wandb integration.
disable-model-invocation: true
---

# /run-experiment — AI/ML Training Experiment Launcher

You are guiding the user through setting up and launching an ML training experiment. Follow the 5 phases below **in order**. Do not skip phases. Use the reference files in `references/` for detailed patterns.

---

## Phase 1 — Gather Experiment Configuration

Collect all required information interactively using `AskUserQuestion`. After each answer, proceed to the next question.

### Step 1.1: Repository Path

Ask the user for the repository/project path where the training code lives.

```
AskUserQuestion:
  question: "What is the path to the training repository/project?"
  header: "Repo path"
  options:
    - label: "Current directory"
      description: "Use the current working directory as the project root"
    - label: "Specify path"
      description: "Enter a custom path to the project"
```

After receiving the path:
- Verify the directory exists with `ls`
- Identify key files: look for `*.py` training scripts, `requirements.txt`, `pyproject.toml`, `setup.py`, `environment.yml`
- Read the main training script to understand its structure (imports, argparse, main function, training loop)

### Step 1.2: Training Command

Ask for the training command/script to run.

```
AskUserQuestion:
  question: "What is the training command to run? (e.g., 'python train.py --epochs 10')"
  header: "Command"
  options:
    - label: "Auto-detect"
      description: "Let me find the main training script and suggest a command"
    - label: "Specify command"
      description: "Enter the exact training command"
```

If auto-detect: scan for common entry points (`train.py`, `main.py`, `run.py`, `run_training.py`) and present findings.

### Step 1.3: wandb Integration

```
AskUserQuestion:
  question: "Do you want to integrate Weights & Biases (wandb) for experiment tracking?"
  header: "wandb"
  options:
    - label: "Yes"
      description: "Add wandb logging to the training script (will modify code with backup)"
    - label: "No"
      description: "Skip wandb integration"
    - label: "Already integrated"
      description: "The code already has wandb — just make sure it's configured"
```

If Yes: also ask for the wandb project name.
```
AskUserQuestion:
  question: "What wandb project name should be used?"
  header: "Project"
  options:
    - label: "Use repo name"
      description: "Use the repository directory name as the wandb project"
    - label: "Specify name"
      description: "Enter a custom wandb project name"
```

### Step 1.4: Conda Environment

```
AskUserQuestion:
  question: "How should the Python environment be managed?"
  header: "Environment"
  options:
    - label: "Create new conda env"
      description: "Create a fresh conda environment for this experiment"
    - label: "Use existing conda env"
      description: "Use an already-created conda environment"
    - label: "Skip (use system Python)"
      description: "Run with the current Python environment as-is"
```

If creating new: ask for environment name and Python version.
If using existing: run `conda env list` and present available environments.

### Step 1.5: GPU Selection

Run `nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader` to detect GPUs.

Present available GPUs with their current utilization:

```
AskUserQuestion:
  question: "Which GPU(s) should be used for training?"
  header: "GPU"
  options:
    - (dynamically generated from nvidia-smi output, e.g.)
    - label: "GPU 0"
      description: "NVIDIA A100 — 2.1/80.0 GB used, 0% util"
    - label: "GPU 0,1"
      description: "Use multiple GPUs (data parallel)"
    - label: "All GPUs"
      description: "Use all available GPUs"
```

### Step 1.6: Confirmation

Display a summary table and ask for confirmation:

```markdown
| Setting        | Value                                |
|----------------|--------------------------------------|
| Repository     | /path/to/repo                        |
| Command        | python train.py --epochs 100         |
| wandb          | Yes — project: "my-project"          |
| Environment    | conda: my-exp-env (Python 3.10)      |
| GPU            | 0                                    |
```

```
AskUserQuestion:
  question: "Does this experiment configuration look correct?"
  header: "Confirm"
  options:
    - label: "Yes, proceed"
      description: "Start setting up the environment and launch the experiment"
    - label: "Edit settings"
      description: "Go back and change some settings"
```

---

## Phase 2 — Environment Setup

Reference: `references/conda-environment-guide.md`

### Step 2.1: Conda Environment

If creating a new environment:
```bash
conda create -n <env_name> python=<version> -y
```

Verify:
```bash
conda run -n <env_name> python --version
```

### Step 2.2: Install Dependencies

Auto-detect and install dependencies in priority order:
1. `requirements.txt` → `conda run -n <env> pip install -r requirements.txt`
2. `pyproject.toml` → `conda run -n <env> pip install -e .`
3. `setup.py` → `conda run -n <env> pip install -e .`
4. `environment.yml` → `conda env update -n <env> -f environment.yml`

If wandb is selected and not already in dependencies:
```bash
conda run -n <env> pip install wandb
```

### Step 2.3: Verify CUDA/PyTorch

```bash
conda run -n <env> python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

If CUDA is not available, consult `references/conda-environment-guide.md` for troubleshooting.

### Step 2.4: Verify wandb Login (if selected)

```bash
conda run -n <env> python -c "import wandb; wandb.login()"
```

If not logged in, instruct user to run `wandb login` or set `WANDB_API_KEY`.

---

## Phase 3 — wandb Integration (if selected)

**Skip this phase entirely if wandb was not selected or is already integrated.**

Reference: `references/wandb-injection-guide.md`

### Step 3.1: Backup Original Script

```bash
cp <training_script> <training_script>.backup
```

Tell the user: "Backed up `<training_script>` to `<training_script>.backup`"

### Step 3.2: Analyze Training Script Structure

Read the training script and identify:
1. **Import section** — where to add `import wandb`
2. **Config/args section** — where `wandb.init(config=...)` goes
3. **Training loop** — where to add `wandb.log({"train/loss": loss, ...})`
4. **Evaluation section** — where to add `wandb.log({"eval/...": metric, ...})`
5. **End of training** — where to add `wandb.finish()`

### Step 3.3: Check for Framework Shortcuts

Before manually injecting wandb code, check if the script uses:
- **HuggingFace Trainer**: Just add `report_to="wandb"` in TrainingArguments + set `WANDB_PROJECT` env var
- **PyTorch Lightning**: Add `WandbLogger` to the Trainer
- **Keras/TF**: Add `WandbCallback`

If a framework shortcut applies, use it instead of manual injection.

### Step 3.4: Inject wandb Code

For vanilla PyTorch / custom training loops, inject code at the 4 locations identified in Step 3.2.

Use `Edit` tool for each insertion point. Show the diff to the user after all edits.

**Important patterns from the reference guide:**
- Use `wandb.init(project=<project>, config=<args_or_config_dict>)` early, after config is parsed
- In the train loop: `wandb.log({"train/loss": loss.item(), "train/lr": lr, "step": step})`
- In eval: `wandb.log({"eval/loss": eval_loss, "eval/accuracy": acc, "epoch": epoch})`
- At the end: `wandb.finish()`
- For distributed training: guard all wandb calls with `if rank == 0:` or `if is_main_process:`

### Step 3.5: User Approval

Show the complete diff of changes and ask:

```
AskUserQuestion:
  question: "These wandb integration changes will be applied. Look correct?"
  header: "Approve"
  options:
    - label: "Approve changes"
      description: "Apply the wandb integration code"
    - label: "Revert"
      description: "Restore the original file from backup"
    - label: "Edit manually"
      description: "Let me make manual adjustments first"
```

---

## Phase 4 — Launch Training

### Step 4.1: Construct Launch Command

Build the full command:
```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> nohup conda run --no-capture-output -n <env> <training_command> > <repo_path>/experiment.log 2>&1 &
```

If wandb is enabled, prepend:
```bash
WANDB_PROJECT=<project_name> CUDA_VISIBLE_DEVICES=<gpu_ids> nohup conda run --no-capture-output -n <env> <training_command> > <repo_path>/experiment.log 2>&1 &
```

### Step 4.2: Execute

Run the command via `Bash` tool with `run_in_background: false`. Capture the PID:
```bash
# Run in a single command to capture PID
CUDA_VISIBLE_DEVICES=<gpu> nohup conda run --no-capture-output -n <env> <command> > <repo>/experiment.log 2>&1 &
echo "PID: $!"
```

### Step 4.3: Verify Launch

Wait 5-10 seconds, then:

1. Check process is alive:
```bash
ps -p <PID> -o pid,stat,etime,cmd
```

2. Check log for early errors:
```bash
tail -50 <repo>/experiment.log
```

3. Report to user:
```markdown
**Experiment launched!**
- PID: <pid>
- Log: <repo>/experiment.log
- Monitor: `tail -f <repo>/experiment.log`
```

If the process died immediately, read the log, diagnose the error using `references/monitoring-and-debugging.md`, and suggest fixes.

---

## Phase 5 — Monitoring & Follow-up

Reference: `references/monitoring-and-debugging.md`

### Step 5.1: wandb Dashboard (if enabled)

Extract the wandb run URL from the log:
```bash
grep -m1 "wandb:" <repo>/experiment.log | grep "https://"
```

Share the URL with the user.

### Step 5.2: Live Metrics (if wandb enabled)

Query latest metrics via wandb API:
```bash
conda run -n <env> python -c "
import wandb
api = wandb.Api()
runs = api.runs('<wandb_entity>/<project>', order='-created_at', per_page=1)
if runs:
    run = runs[0]
    print(f'Run: {run.name} ({run.state})')
    print(f'URL: {run.url}')
    for k, v in run.summary.items():
        if not k.startswith('_'):
            print(f'  {k}: {v}')
"
```

### Step 5.3: System Monitoring

```bash
# GPU utilization
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader

# Process status
ps -p <PID> -o pid,stat,%cpu,%mem,etime
```

### Step 5.4: Error Diagnosis

If the user reports issues or the process has died:

1. Read the last 100 lines of the log
2. Check for common failure patterns (see `references/monitoring-and-debugging.md`):
   - OOM → suggest reducing batch size or enabling gradient checkpointing
   - NaN loss → check learning rate, data preprocessing
   - CUDA errors → check GPU availability, driver compatibility
   - Import errors → check dependencies
3. Propose specific fixes and offer to re-launch

### Step 5.5: Cleanup Info

Provide the user with useful commands:
```markdown
**Useful commands:**
- View live log: `tail -f <repo>/experiment.log`
- Check GPU: `nvidia-smi`
- Stop training: `kill <PID>`
- Resume wandb: The run ID is logged — use `wandb.init(resume="must", id="<run_id>")` to resume
```
