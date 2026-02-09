import tensorflow as tf
import numpy as np
import os
import subprocess
import time

def check_memory_growth():
    print("--- 1. Memory Growth Verification ---")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[+] Memory growth set to True for all GPUs.")
            return "Enabled"
        except Exception as e:
            print(f"[!] Warning: Could not set memory growth (likely already initialized): {e}")
            return "Active/Initialized"
    return "N/A"

def functional_math_test():
    print("\n--- 2. Functional Math Test (MatMul) ---")
    try:
        size = 5000
        with tf.device('/GPU:0'):
            a = tf.random.normal([size, size])
            b = tf.random.normal([size, size])
            
            # Warm up
            _ = tf.matmul(a, b)
            
            start = time.time()
            c = tf.matmul(a, b)
            end = time.time()
            
            val = tf.reduce_sum(c).numpy()
            print(f"[+] MatMul (5000x5000) successful. Result sum: {val:.2f}")
            print(f"[+] Execution time: {end - start:.4f} seconds.")
            return "Success" if val != 0 else "Fail (Zero Result)"
    except Exception as e:
        print(f"[!] MatMul Failed: {e}")
        return f"Fail ({type(e).__name__})"

def xla_compilation_check():
    print("\n--- 3. XLA Compilation Check ---")
    try:
        @tf.function(jit_compile=True)
        def xla_op(x):
            return tf.reduce_sum(tf.square(x))

        x = tf.random.normal([1000, 1000])
        res = xla_op(x)
        print(f"[+] XLA JIT compilation successful. Result: {res.numpy():.2f}")
        return "Enabled"
    except Exception as e:
        print(f"[!] XLA Compilation Failed: {e}")
        return "Disabled"

def mixed_precision_check():
    print("\n--- 4. Mixed Precision Policy ---")
    try:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"[+] Global policy set to: {mixed_precision.global_policy().name}")
        
        # Check if GPU supports it well (Compute Capability >= 7.0)
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"]).decode().strip()
        print(f"[+] GPU Compute Capability: {gpu_info}")
        if float(gpu_info) >= 7.0:
            print("[+] Tensor Cores should be active for FP16 ops.")
            return "Active"
        else:
            print("[!] Warning: Compute capability < 7.0. Tensor Cores may not be utilized.")
            return "Hardware Restricted"
    except Exception as e:
        print(f"[!] Mixed Precision Setup Failed: {e}")
        return "Inactive"

def driver_consistency_audit():
    print("\n--- 5. Driver/Toolkit Consistency ---")
    try:
        build_info = tf.sysconfig.get_build_info()
        print(f"TF Build Info: {build_info}")
        
        nv_smi = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).decode().strip()
        print(f"NVIDIA Driver: {nv_smi}")
        
        # Simple heuristic check
        return nv_smi, build_info.get('cuda_version', 'Unknown')
    except Exception as e:
        print(f"[!] Consistency Audit Failed: {e}")
        return "Error", "Error"

def get_vram_info():
    try:
        vram_free = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).decode().strip()
        return f"{int(vram_free)/1024:.2f} GB"
    except:
        return "Unknown"

if __name__ == "__main__":
    mem = check_memory_growth()
    math = functional_math_test()
    xla = xla_compilation_check()
    mp = mixed_precision_check()
    vram = get_vram_info()
    
    print("\n" + "="*30)
    print("      HEALTH CHECK SUMMARY")
    print("="*30)
    print(f"Matrix Multiplication:   {math}")
    print(f"XLA JIT Support:         {xla}")
    print(f"Tensor Core Utilization: {mp}")
    print(f"Available VRAM:          {vram}")
    print("="*30)
