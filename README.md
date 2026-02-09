# research-skills

A collection of Claude Code skills that automate AI research experiment workflows.

## Skills

### `/run-experiment`

An interactive skill that guides you through the full lifecycle of launching an ML training experiment — from environment setup to monitoring.

| Phase | What it does |
|-------|-------------|
| 1. Configuration | Collects repo path, training command, wandb preference, conda env, and GPU selection |
| 2. Environment Setup | Creates conda env, installs dependencies, verifies CUDA/PyTorch |
| 3. wandb Integration | Analyzes your training script and injects wandb tracking code (with backup) |
| 4. Launch | Runs training in background with `nohup`, captures PID and log path |
| 5. Monitoring | Provides wandb dashboard links, GPU monitoring, and error diagnosis |

**Key features:**
- Automatic dependency detection (`requirements.txt`, `pyproject.toml`, `setup.py`, `environment.yml`)
- Framework-aware wandb injection (HuggingFace Trainer, PyTorch Lightning, Keras shortcuts)
- Distributed training support (rank-0 guarding for wandb calls)
- Built-in error diagnosis for OOM, NaN loss, CUDA errors, and training stalls

## Installation

### Option 1: User-level (available in all projects)

```bash
git clone https://github.com/CUN-bjy/research-skills.git
cp -r research-skills/.claude/skills/run-experiment ~/.claude/skills/
```

### Option 2: Project-level (scoped to a single repo)

```bash
git clone https://github.com/CUN-bjy/research-skills.git
cp -r research-skills/.claude/skills/run-experiment your-project/.claude/skills/
```

Then open Claude Code and type `/run-experiment` to get started.

## Directory Structure

```
.claude/skills/run-experiment/
├── SKILL.md                              # Main skill definition (5-phase workflow)
└── references/
    ├── conda-environment-guide.md        # conda patterns for non-interactive shells
    ├── wandb-injection-guide.md          # How to inject wandb into training scripts
    └── monitoring-and-debugging.md       # Experiment monitoring & error diagnosis
```

## License

MIT
