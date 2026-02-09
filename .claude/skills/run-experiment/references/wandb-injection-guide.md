# wandb Injection Guide

How to add Weights & Biases tracking to an arbitrary training script.

---

## 1. Locating Injection Points

Read the training script and identify these 5 locations:

### Import Section
- Find the last `import` or `from ... import` line in the top-level imports
- Insert `import wandb` after it

### Config/Args Section
- Look for `argparse.ArgumentParser`, `OmegaConf.load`, `@dataclass` config classes, or plain dict configs
- `wandb.init()` goes **after** the config is fully parsed/constructed

### Training Loop
- Look for patterns: `for epoch in`, `for step in`, `for batch in`, `model.train()`, `optimizer.step()`
- Insert `wandb.log()` after the loss computation, typically right after `loss.backward()` or `optimizer.step()`

### Evaluation Section
- Look for patterns: `model.eval()`, `with torch.no_grad()`, `evaluate(`, `val_loader`
- Insert `wandb.log()` after evaluation metrics are computed

### End of Training
- Look for: the end of the training loop, `save_model`, `torch.save`, or the last line of `main()`
- Insert `wandb.finish()` as the very last call

---

## 2. wandb.init() Patterns by Config Type

### argparse
```python
args = parser.parse_args()
wandb.init(project="<PROJECT>", config=vars(args))
```

### OmegaConf / Hydra
```python
cfg = OmegaConf.load("config.yaml")
wandb.init(project="<PROJECT>", config=OmegaConf.to_container(cfg, resolve=True))
```

### Dataclass Config
```python
config = TrainingConfig()
wandb.init(project="<PROJECT>", config=config.__dict__)
```

### Plain Dict
```python
config = {"lr": 1e-4, "batch_size": 32, ...}
wandb.init(project="<PROJECT>", config=config)
```

### No Config Object
```python
wandb.init(project="<PROJECT>", config={
    "learning_rate": lr,
    "batch_size": batch_size,
    "epochs": num_epochs,
    # add other hyperparameters manually
})
```

---

## 3. wandb.log() Patterns

### Training Loop
```python
# After loss.backward() / optimizer.step()
wandb.log({
    "train/loss": loss.item(),
    "train/lr": optimizer.param_groups[0]["lr"],
    "step": global_step,
})
```

Log every N steps to avoid overhead:
```python
if global_step % log_every == 0:
    wandb.log({"train/loss": loss.item(), "step": global_step})
```

### Evaluation
```python
wandb.log({
    "eval/loss": eval_loss,
    "eval/accuracy": accuracy,
    "eval/perplexity": perplexity,
    "epoch": epoch,
})
```

### Additional Useful Logs
```python
# Learning rate schedule
wandb.log({"train/lr": scheduler.get_last_lr()[0], "step": step})

# Gradient norm
wandb.log({"train/grad_norm": total_norm, "step": step})

# Custom metrics
wandb.log({"train/throughput_samples_per_sec": throughput, "step": step})
```

---

## 4. Distributed Training Guard

When using `torch.distributed`, `deepspeed`, or `accelerate`, **only log from rank 0**:

```python
import os

# Common rank detection patterns
local_rank = int(os.environ.get("LOCAL_RANK", 0))
# or
local_rank = args.local_rank
# or (with accelerate)
is_main = accelerator.is_main_process

# Guard wandb calls
if local_rank == 0:
    wandb.init(project="<PROJECT>", config=vars(args))

# In training loop
if local_rank == 0:
    wandb.log({"train/loss": loss.item()})

# At end
if local_rank == 0:
    wandb.finish()
```

For **Accelerate** specifically:
```python
from accelerate import Accelerator
accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(project_name="<PROJECT>", config=vars(args))
# ... training ...
accelerator.log({"train/loss": loss.item()}, step=step)
accelerator.end_training()
```

---

## 5. Framework-Specific Shortcuts

### HuggingFace Transformers Trainer

**Do NOT manually inject wandb code.** Instead:

1. Set environment variable:
```bash
export WANDB_PROJECT="<PROJECT>"
```

2. Add to TrainingArguments:
```python
training_args = TrainingArguments(
    ...,
    report_to="wandb",
    run_name="<optional_run_name>",
)
```

That's it. The Trainer handles everything automatically.

### PyTorch Lightning

```python
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(project="<PROJECT>", log_model=False)
trainer = Trainer(logger=wandb_logger, ...)
```

Lightning handles `wandb.init()`, `wandb.log()`, and `wandb.finish()` automatically.

### Keras / TensorFlow

```python
from wandb.integration.keras import WandbCallback

model.fit(
    ...,
    callbacks=[WandbCallback(project="<PROJECT>")],
)
```

---

## 6. Common Pitfalls

### Multiple wandb.init() calls
- If the script already has `wandb.init()`, don't add another one
- Check with `grep -n "wandb.init\|wandb.log\|import wandb"` first

### Logging tensors directly
```python
# BAD — logs a tensor object
wandb.log({"loss": loss})

# GOOD — logs a Python float
wandb.log({"loss": loss.item()})
```

### Logging too frequently
- Don't log every single step in a tight loop (causes overhead)
- Log every 10-100 steps depending on step speed

### Missing wandb.finish()
- Always call `wandb.finish()` at the end
- Without it, the run may not sync properly and appear as "crashed"

### Config not captured
- Always pass config to `wandb.init(config=...)` — this is essential for experiment comparison
- If config changes during training (e.g., LR schedule), log those changes too
