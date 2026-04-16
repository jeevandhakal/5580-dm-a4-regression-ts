# GPU Quick Start

```bash
uv sync
uv run src/gpu_setup/tools/verify_setup.py
uv run src/gpu_setup/tools/diagnose_cuda.py
```

If GPU is detected, run:

```bash
uv run src/workflow/scripts/2_final_retrain_and_evaluate.py
```

If not detected, follow `src/gpu_setup/docs/TROUBLESHOOTING.md`.

