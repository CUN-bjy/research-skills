# Monitoring and Debugging Guide

How to monitor running experiments and diagnose common failures.

---

## 1. Log Monitoring

### Live Tail
```bash
tail -f <repo>/experiment.log
```

### Last N Lines
```bash
tail -100 <repo>/experiment.log
```

### Search for Errors
```bash
grep -i "error\|exception\|traceback\|failed\|killed" <repo>/experiment.log
```

### Search for Metrics
```bash
grep -i "loss\|accuracy\|epoch\|step" <repo>/experiment.log | tail -20
```

---

## 2. GPU Monitoring

### Current Status
```bash
nvidia-smi
```

### Compact View (for scripts)
```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
```

### Continuous Monitoring (every 2 seconds)
```bash
watch -n 2 nvidia-smi
```

### Check Which Process Uses Which GPU
```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
```

---

## 3. Process Monitoring

### Check if Training is Running
```bash
ps -p <PID> -o pid,stat,%cpu,%mem,etime,cmd
```

### Find Training Process (if PID is lost)
```bash
ps aux | grep "python.*train"
```

### Monitor Resource Usage
```bash
top -p <PID> -b -n 1
```

---

## 4. wandb Metrics via API

### Query Latest Run Metrics
```python
import wandb
api = wandb.Api()

# Get the most recent run
runs = api.runs("<entity>/<project>", order="-created_at", per_page=1)
if runs:
    run = runs[0]
    print(f"Run: {run.name} (state: {run.state})")
    print(f"URL: {run.url}")
    print(f"\nSummary metrics:")
    for key, value in sorted(run.summary.items()):
        if not key.startswith("_"):
            print(f"  {key}: {value}")
    print(f"\nConfig:")
    for key, value in sorted(run.config.items()):
        if not key.startswith("_"):
            print(f"  {key}: {value}")
```

### Query Training History
```python
import wandb
api = wandb.Api()
run = api.runs("<entity>/<project>", order="-created_at", per_page=1)[0]

# Get last 10 logged steps
history = run.scan_history(keys=["train/loss", "step"], page_size=10)
for row in list(history)[-10:]:
    print(f"  step {row.get('step', '?')}: loss={row.get('train/loss', '?')}")
```

### Check Run State
Possible states: `running`, `finished`, `crashed`, `failed`, `killed`

```python
run = api.runs("<entity>/<project>", order="-created_at", per_page=1)[0]
print(f"State: {run.state}")  # running, finished, crashed, etc.
```

---

## 5. Common Failure Patterns

### Instant Death (Process exits immediately)

**Check**: `tail -50 <repo>/experiment.log`

Common causes:
- **ImportError**: Missing package → `conda run -n <env> pip install <package>`
- **FileNotFoundError**: Missing data/config file → check paths
- **SyntaxError**: Bad code edit (e.g., from wandb injection) → check the modified file, restore from `.backup`
- **CUDA out of memory on init**: GPU already in use → check `nvidia-smi`, use a different GPU
- **Permission denied**: File not executable or path issue

### OOM (Out of Memory)

**Symptoms in log**:
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
torch.cuda.OutOfMemoryError
```

**Fixes** (try in order):
1. Reduce batch size (e.g., halve it)
2. Enable gradient checkpointing:
   ```python
   model.gradient_checkpointing_enable()
   ```
3. Use mixed precision (fp16/bf16):
   ```python
   # Add --fp16 or --bf16 flag, or:
   from torch.cuda.amp import autocast, GradScaler
   ```
4. Free GPU memory from other processes
5. Use a GPU with more memory

### NaN Loss

**Symptoms in log**:
```
loss: nan
NaN detected in loss
```

**Diagnosis**:
- Check learning rate (too high → NaN)
- Check data preprocessing (missing values, incorrect normalization)
- Check for division by zero
- Check gradient clipping settings

**Fixes**:
1. Reduce learning rate (try 10x smaller)
2. Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
3. Use mixed precision with loss scaling
4. Check data pipeline for corrupted samples

### Training Stall (No Progress)

**Symptoms**: Process is running but log shows no new output.

**Diagnosis**:
```bash
# Check if process is alive
ps -p <PID> -o stat
# D = disk sleep (I/O wait), S = sleeping, R = running

# Check GPU utilization
nvidia-smi
# 0% utilization = likely data loading bottleneck

# Check CPU usage
top -p <PID> -b -n 1
```

**Common causes**:
- **Data loading bottleneck**: Increase `num_workers` in DataLoader
- **Distributed training deadlock**: One rank crashed, others hang → kill all and restart
- **Disk I/O**: Slow storage for large datasets → consider caching or faster storage
- **Waiting for input**: Script prompting for something → check if it needs stdin

### CUDA/cuDNN Errors

**Symptoms**:
```
CUDA error: device-side assert triggered
cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
NCCL error: unhandled system error
```

**Fixes**:
1. Set `CUDA_LAUNCH_BLOCKING=1` for better error messages:
   ```bash
   CUDA_LAUNCH_BLOCKING=1 conda run -n <env> python train.py
   ```
2. Check for tensor shape mismatches
3. Check for out-of-bounds indices (common with embedding layers)
4. Verify CUDA/cuDNN/driver version compatibility

---

## 6. Stopping and Cleanup

### Graceful Stop
```bash
kill <PID>
# This sends SIGTERM — most training scripts handle this gracefully
```

### Force Stop
```bash
kill -9 <PID>
```

### Stop All Training Processes
```bash
# Be careful with this!
pkill -f "python.*train"
```

### Clean Up GPU Memory (after force kill)
```bash
# Check for zombie GPU processes
nvidia-smi
# If memory is still held by a dead process:
# The process should release memory automatically. If not:
sudo fuser -v /dev/nvidia*  # See what's using GPUs
```

### Clean Up wandb

If a run was killed and appears as "crashed" in wandb:
```python
import wandb
wandb.init(project="<project>", resume="must", id="<run_id>")
wandb.finish()  # Properly close it
```

Or from CLI:
```bash
wandb sync <repo>/wandb/latest-run/  # Sync any unsynced data
```

---

## 7. Quick Diagnostic Checklist

When something goes wrong, run through this checklist:

1. **Is the process running?** `ps -p <PID>`
2. **What's in the log?** `tail -50 <repo>/experiment.log`
3. **Any errors?** `grep -i "error\|exception" <repo>/experiment.log | tail -10`
4. **GPU status?** `nvidia-smi`
5. **Disk space?** `df -h .`
6. **Memory?** `free -h`
7. **wandb state?** Check dashboard or API for run state
