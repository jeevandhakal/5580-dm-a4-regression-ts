#!/usr/bin/env python3
"""CUDA/GPU diagnostic tool."""
import subprocess
import os
from pathlib import Path

def run_cmd(cmd):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

print("=" * 70)
print("CUDA/GPU DIAGNOSTIC REPORT")
print("=" * 70)

# 1. Check system CUDA packages
print("\n1. System CUDA packages (dpkg):")
dpkg_out = run_cmd("dpkg -l | grep -i 'cuda\\|nvidia' | head -5")
if dpkg_out:
    print(dpkg_out[:200])
else:
    print("  [WARN] No CUDA packages found via dpkg")

# 2. Check for CUDA installations
print("\n2. CUDA Toolkit locations:")
for path in ["/usr/local/cuda", "/opt/cuda", "/usr/lib/nvidia-cuda-toolkit"]:
    if os.path.exists(path):
        print(f"  ✓ Found: {path}")
    else:
        print(f"  ✗ Missing: {path}")

# 3. Check for key libraries
print("\n3. System CUDA libraries (/usr/lib/x86_64-linux-gnu):")
system_libs = ["libcudart", "libcublas", "libcusolver", "libcuda"]
for lib in system_libs[:2]:
    out = run_cmd(f"find /usr/lib/x86_64-linux-gnu -name '{lib}*.so*' 2>/dev/null | head -1")
    if out:
        print(f"  ✓ {lib}: Found")
    else:
        print(f"  ✗ {lib}: NOT FOUND")

# 4. Check Python TensorFlow
print("\n4. Python environment:")
try:
    import tensorflow as tf
    print(f"  ✓ TensorFlow version: {tf.__version__}")
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    print(f"  {'✓' if gpu_available else '✗'} GPU available: {gpu_available}")
    if gpu_available:
        gpus = tf.config.list_physical_devices('GPU')
        print(f"    Detected GPUs: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"      GPU {i}: {gpu}")
except ImportError:
    print("  ✗ TensorFlow not installed")
except Exception as e:
    print(f"  ⚠ TensorFlow error: {e}")

# 5. Check NVIDIA driver
print("\n5. NVIDIA Driver:")
driver_out = run_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null")
if driver_out and "Error" not in driver_out:
    print(f"  ✓ Driver version: {driver_out}")
else:
    print("  ✗ nvidia-smi not available or driver not installed")

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)
print("""
If GPU is NOT available (GPU reported as False):
  1. Your code will work on CPU (fully functional)
  2. (Optional) Install CUDA for GPU acceleration
  3. See: src/gpu_setup/docs/TROUBLESHOOTING.md

If GPU IS available:
  1. Your code will use GPU automatically
  2. 5-10x faster training than CPU
  3. Everything else stays the same
""")

