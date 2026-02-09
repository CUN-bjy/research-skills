# Conda Environment Guide

Patterns for managing conda environments in non-interactive (Claude Code) contexts.

---

## 1. Using `conda run` (Non-Interactive Shell)

Claude Code runs commands in non-interactive bash. `conda activate` does not work reliably here. **Always use `conda run` instead.**

### Basic Pattern
```bash
# Instead of: conda activate myenv && python train.py
conda run -n myenv python train.py
```

### With Output Streaming
```bash
# --no-capture-output lets stdout/stderr stream through in real time
conda run --no-capture-output -n myenv python train.py
```

### With Environment Variables
```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n myenv python train.py
```

### With nohup (Background)
```bash
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n myenv python train.py > experiment.log 2>&1 &
echo "PID: $!"
```

---

## 2. Creating Environments

### Basic Creation
```bash
conda create -n <env_name> python=<version> -y
```

Common Python versions for ML:
- Python 3.10: Safe default, wide compatibility
- Python 3.11: Good for newer projects
- Python 3.12: Newest, some packages may not support yet

### Verify Creation
```bash
conda run -n <env_name> python --version
```

---

## 3. Installing Dependencies

### Priority Order for Dependency Detection

Check for these files in the repository and install in this priority:

1. **`requirements.txt`** (most common)
```bash
conda run -n <env> pip install -r requirements.txt
```

2. **`pyproject.toml`** (modern Python projects)
```bash
conda run -n <env> pip install -e .
# or if extras are needed:
conda run -n <env> pip install -e ".[dev,train]"
```

3. **`setup.py`** (older projects)
```bash
conda run -n <env> pip install -e .
```

4. **`environment.yml`** (conda-native)
```bash
conda env update -n <env> -f environment.yml
```

### Additional Common Installs
```bash
# wandb (if not in dependencies)
conda run -n <env> pip install wandb

# Common ML extras that might be missing
conda run -n <env> pip install tensorboard scipy matplotlib
```

---

## 4. CUDA / PyTorch Verification

### PyTorch + CUDA Check
```bash
conda run -n <env> python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

### JAX + CUDA Check
```bash
conda run -n <env> python -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'Devices: {jax.devices()}')
print(f'GPU count: {jax.device_count(\"gpu\")}')
"
```

### TensorFlow + CUDA Check
```bash
conda run -n <env> python -c "
import tensorflow as tf
print(f'TF version: {tf.__version__}')
print(f'GPUs: {tf.config.list_physical_devices(\"GPU\")}')
"
```

---

## 5. Common Errors and Fixes

### CUDA Version Mismatch
**Symptom**: `RuntimeError: CUDA error: no kernel image is available for execution on the device`
or PyTorch says CUDA is available but operations fail.

**Diagnosis**:
```bash
nvidia-smi  # Check driver CUDA version
conda run -n <env> python -c "import torch; print(torch.version.cuda)"  # Check PyTorch CUDA version
```

**Fix**: Reinstall PyTorch with the correct CUDA version:
```bash
# Check what CUDA version your driver supports (nvidia-smi top right)
# Then install matching PyTorch
conda run -n <env> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# or cu118, cu124 depending on your driver
```

### Package Conflicts
**Symptom**: `pip install` fails with version conflicts.

**Fix options**:
1. Try installing without version constraints:
```bash
conda run -n <env> pip install -r requirements.txt --no-deps
conda run -n <env> pip install -r requirements.txt  # then retry with deps
```

2. Create a fresh environment with a different Python version

3. Install conflicting packages one by one to identify the conflict

### "conda: command not found"
**Fix**:
```bash
# Find conda installation
which conda || find /opt /home -name "conda" -type f 2>/dev/null | head -5

# Common locations
export PATH="/opt/conda/bin:$PATH"
# or
export PATH="$HOME/miniconda3/bin:$PATH"
# or
export PATH="$HOME/anaconda3/bin:$PATH"
```

### Pip Install Hangs or Is Extremely Slow
**Possible causes**:
- Large package download (PyTorch is ~2GB)
- Network issues

**Mitigation**: Set a timeout and use verbose mode:
```bash
conda run -n <env> pip install -r requirements.txt -v --timeout 120
```

---

## 6. Environment Management Tips

### List Existing Environments
```bash
conda env list
```

### Remove an Environment
```bash
conda env remove -n <env_name>
```

### Clone an Environment (fast way to try modifications)
```bash
conda create -n <new_env> --clone <existing_env>
```

### Export for Reproducibility
```bash
conda run -n <env> pip freeze > requirements-frozen.txt
```
