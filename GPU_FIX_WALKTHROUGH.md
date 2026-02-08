# GPU Fix Walkthrough - TensorFlow 2.20.0 (RTX 4070)

This document summarizes the fix applied to enable TensorFlow GPU acceleration on Zorin OS (Ubuntu-based) with an NVIDIA RTX 4070.

## Problem
TensorFlow 2.20.0 expected **CUDA 12.5.1** and **cuDNN 9**, but the system only had CUDA 12.0 and cuDNN 8.9.2 installed. This version mismatch prevented TensorFlow from detecting the GPU and caused XLA compilation failures.

## Solution: The Hybrid Bridge
We created a symbolic link bridge in `/usr/local/cuda/lib64/` that aliases existing libraries to the names and versions TensorFlow expects.

### 1. Hybrid Library Sourcing
The remediation script `fix_cuda_paths.sh` was updated to source libraries from two locations:
-   **System (`/usr/lib/x86_64-linux-gnu/`):** For CUDA core libraries (libcudart, libcublas, libcusolver, etc.).
-   **Virtual Environment (`.venv`):** For **cuDNN 9** libraries (`nvidia-cudnn-cu12`). This was required because the system's cuDNN 8.9.2 was incompatible with TensorFlow's XLA compiler.

### 2. Remediation Script Actions
The script performed the following:
-   created `/usr/local/cuda/lib64` and `/usr/local/cuda/bin`.
-   Linked system CUDA 12.0 libraries to `so.12` and `so.11` (aliasing).
-   **Crucially:** Linked the `.venv`'s `libcudnn.so.9` and its sub-libraries (ops, cnn, adv) to `/usr/local/cuda/lib64`, resolving the XLA "cudnnCreate" symbol error.
-   Linked `ptxas` to `/usr/local/cuda/bin/ptxas`.
-   Refreshed `ldconfig`.

### 3. Environment Configuration (Fish Shell)
To use the bridge, the following environment variables verified to work:

```fish
# Prioritize the bridged libraries
set -gx LD_LIBRARY_PATH /usr/local/cuda/lib64 $LD_LIBRARY_PATH

# Point XLA to the correct libdevice bitcode directory
set -gx XLA_FLAGS "--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit/libdevice"

# Optional: Disable oneDNN floating point warnings
set -gx TF_ENABLE_ONEDNN_OPTS 0
```

## Health Check Results
A comprehensive health check suite (`gpu_health_check.py`) was run to verify stability and performance:

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Matrix Multiplication** | **Success** | 5000x5000 random matrix op on GPU:0 |
| **XLA JIT Support** | **Enabled** | XLA compiler successfully uses cuDNN 9 bridge |
| **Tensor Core Utilization** | **Active** | Mixed precision policy (`mixed_float16`) active |
| **Available VRAM** | **~7.06 GB** | Memory growth enabled |

## Verification Scripts
You can re-verify the status at any time using the provided scripts:

1.  **Quick Check:** `python code/verify_gpu.py`
2.  **Full Health Check:** `bash code/run_health_check.sh`
