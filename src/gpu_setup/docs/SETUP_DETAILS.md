# GPU Setup Details

This project uses NVIDIA CUDA runtime wheels from `pyproject.toml` and configures runtime paths automatically in scripts.

## Install

```bash
uv sync
```

## Verify

```bash
uv run src/gpu_setup/tools/verify_setup.py
```

## Notes

- Driver health is checked with `nvidia-smi`.
- TensorFlow build metadata can be checked with:

```bash
uv run python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"
```

