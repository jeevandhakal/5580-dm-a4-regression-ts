import tensorflow as tf
import os

print("--- TensorFlow GPU Verification ---")
print(f"TensorFlow Version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
print(f"Num GPUs Available: {len(gpus)}")

for i, gpu in enumerate(gpus):
    print(f"GPU {i}: {gpu.name} (Type: {gpu.device_type})")

try:
    from tensorflow.python.client import device_lib
    print("\n--- Detailed Device Info ---")
    print(device_lib.list_local_devices())
except Exception as e:
    print(f"\nCould not get detailed info: {e}")

if not gpus:
    print("\n[!] NO GPU DETECTED. Check LD_LIBRARY_PATH and XLA_FLAGS.")
else:
    print("\n[+] SUCCESS: GPU detected and available.")
