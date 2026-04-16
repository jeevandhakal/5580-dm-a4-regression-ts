# GPU Troubleshooting

## 1) Baseline checks

```bash
uv sync
nvidia-smi
uv run src/gpu_setup/tools/verify_setup.py
```

If `nvidia-smi` works but TensorFlow shows no GPU, it is usually a CUDA user-space library/toolchain issue.

## 2) Deeper diagnostics

```bash
uv run src/gpu_setup/tools/diagnose_cuda.py
```

## 3) Recovery path

```bash
uv run src/gpu_setup/tools/gpu_recovery.py --cleanup
bash src/gpu_setup/scripts/cleanup_broken_links.sh
bash src/gpu_setup/scripts/fix_cuda_paths.sh
```

## 4) Validate end-to-end

```bash
uv run src/workflow/scripts/2_final_retrain_and_evaluate.py
```

## Common symptoms

- `Could not find cuda drivers` + `GPU []`: runtime/toolchain not visible to TensorFlow.
- `No PTX compilation provider`: missing `ptxas`/`nvlink` toolchain.
- `Can't find libdevice`: missing `cuda_nvcc` libdevice path.

## Current repo behavior

Workflow scripts auto-wire NVIDIA library paths from `.venv` before importing TensorFlow.

