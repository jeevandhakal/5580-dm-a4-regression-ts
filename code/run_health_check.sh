#!/bin/bash

# Ensure environment is set for the script
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit/libdevice"
export TF_CPP_MIN_LOG_LEVEL=2 # Reduce noise

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Starting Comprehensive GPU Health Check..."
# Use the absolute paths derived from script location
"$ROOT_DIR/.venv/bin/python3" "$SCRIPT_DIR/gpu_health_check.py"
