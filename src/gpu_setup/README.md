# GPU Setup Utilities

This package contains helper tools to verify GPU visibility and troubleshoot CUDA runtime wiring for TensorFlow/XGBoost/LightGBM.

## Quick Start

```bash
uv sync
uv run src/gpu_setup/tools/verify_setup.py
```

## Commands

```bash
# Verify Python deps + GPU visibility
uv run src/gpu_setup/tools/verify_setup.py

# Show CUDA/driver diagnostic details
uv run src/gpu_setup/tools/diagnose_cuda.py

# Attempt cleanup/recovery checks
uv run src/gpu_setup/tools/gpu_recovery.py --cleanup

# Cleanup stale symlinks (system path)
bash src/gpu_setup/scripts/cleanup_broken_links.sh

# Rebuild CUDA bridge symlinks (system path)
bash src/gpu_setup/scripts/fix_cuda_paths.sh
```

## Notes

- `nvidia-smi` can be healthy while TensorFlow still fails if CUDA user-space libs are missing.
- This repository configures NVIDIA runtime paths from `.venv` automatically in runtime scripts.
- If GPU is unavailable, workflow scripts still run on CPU.

