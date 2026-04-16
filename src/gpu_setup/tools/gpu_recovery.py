#!/usr/bin/env python3
"""GPU recovery and cleanup utility."""
import subprocess
import sys
from pathlib import Path

def cleanup_broken_links():
    """Remove broken symlinks from /usr/local/cuda/lib64."""
    cuda_lib_dir = Path("/usr/local/cuda/lib64")
    if not cuda_lib_dir.exists():
        print("✓ /usr/local/cuda/lib64 does not exist yet")
        return

    print("Cleaning up broken links in /usr/local/cuda/lib64...")
    broken_count = 0
    for link in cuda_lib_dir.glob("*.so*"):
        if link.is_symlink():
            try:
                link.resolve(strict=True)
            except (FileNotFoundError, RuntimeError):
                try:
                    link.unlink()
                    broken_count += 1
                except PermissionError:
                    subprocess.run(f"sudo rm -f {link}", shell=True, capture_output=True)
                    broken_count += 1

    if broken_count > 0:
        print(f"  Removed {broken_count} broken links")
    else:
        print("  No broken links found")

def detect_gpu():
    """Check if GPU is available."""
    print("\n" + "="*70)
    print("GPU DETECTION")
    print("="*70)

    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ TensorFlow sees {len(gpus)} GPU device(s)")
            return True
        else:
            print("✗ TensorFlow not detecting GPU")
            return False
    except Exception as e:
        print(f"⚠ TensorFlow GPU check error: {e}")
        return False

def suggest_device_settings(gpu_available):
    """Print recommended device settings."""
    print("\n" + "="*70)
    print("RECOMMENDED DEVICE SETTINGS")
    print("="*70)

    if gpu_available:
        print("\n✓ GPU is available. Use these settings in code:")
        print("""
  XGBoost:
    device='cuda'
    tree_method='gpu_hist'
  
  LightGBM:
    device='gpu'
  
  Your project's src/final_retrain.py already does this automatically!
""")
    else:
        print("\n✗ GPU not available. CPU will be used:")
        print("""
  Your project automatically uses CPU.
  No action needed - everything works!
  
  (Optional) To enable GPU later, install CUDA/cuDNN
  See: src/gpu_setup/docs/TROUBLESHOOTING.md
""")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU recovery and configuration")
    parser.add_argument("--cleanup", action="store_true", help="Clean up broken links")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("GPU SETUP RECOVERY TOOL")
    print("="*70)

    if args.cleanup:
        cleanup_broken_links()

    gpu_available = detect_gpu()
    suggest_device_settings(gpu_available)

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    if gpu_available:
        print("""
Your GPU is ready! Your code will use it automatically.
Training will be 5-10x faster than CPU.
""")
    else:
        print("""
Your project works on CPU.
Training is slower but fully functional.

(Optional) To enable GPU:
  1. Install CUDA Toolkit 12.0+
  2. Install cuDNN 9+
  3. Run this tool again
  
See: src/gpu_setup/docs/TROUBLESHOOTING.md for detailed steps
""")

    return 0 if gpu_available else 1

if __name__ == "__main__":
    sys.exit(main())

