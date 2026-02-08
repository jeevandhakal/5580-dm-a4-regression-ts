#!/bin/bash

# Ensure directories exist
sudo mkdir -p /usr/local/cuda/lib64
sudo mkdir -p /usr/local/cuda/bin

SRC_DIR="/usr/lib/x86_64-linux-gnu"
# Look for venv libraries (preferring these for cuDNN)
VENV_LIB_DIR=$(ls -d .venv/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ' ')
DST_DIR="/usr/local/cuda/lib64"

# Specific mappings to ensure we hit the right files
map_lib() {
    local base=$1
    local target_vers=$2
    
    # Search for the library. 
    # Try VENV_LIB_DIR first for cuDNN/Cublas if they are there
    local real_file=""
    for dir in $VENV_LIB_DIR $SRC_DIR; do
        real_file=$(find "$dir" -maxdepth 1 -name "$base.so.[0-9]*.[0-9]*.[0-9]*" 2>/dev/null | head -n 1)
        if [ -z "$real_file" ]; then
            real_file=$(find "$dir" -maxdepth 1 -name "$base.so.[0-9]*.[0-9]*" 2>/dev/null | head -n 1)
        fi
        if [ -z "$real_file" ]; then
            real_file=$(find "$dir" -maxdepth 1 -name "$base.so.[0-9]*" 2>/dev/null | grep -v "\.so$" | head -n 1)
        fi
        [ -n "$real_file" ] && break
    done

    if [ -n "$real_file" ]; then
        # Resolve to absolute real path
        local real_path=$(readlink -f "$real_file")
        echo "Found $base: $real_path"
        
        for ver in $target_vers; do
            echo "  Linking to $DST_DIR/$base.so.$ver"
            sudo ln -sf "$real_path" "$DST_DIR/$base.so.$ver"
        done
        # Also base link
        sudo ln -sf "$real_path" "$DST_DIR/$base.so"
    else
        echo "Warning: Could not find $base in $SRC_DIR"
    fi
}

# 1. CUDA Core Libraries (Target 11 and 12)
map_lib "libcudart" "11 12"
map_lib "libcublas" "11 12"
map_lib "libcublasLt" "11 12"
map_lib "libcusolver" "11 12"
map_lib "libcusparse" "11 12"
map_lib "libcurand" "11 12"
map_lib "libcufft" "11 12"
map_lib "libnvrtc" "11 12"
map_lib "libnvjitlink" "12"
map_lib "libcupti" "11 12"
map_lib "libcuda" "1 11 12"

# 2. NPP Libraries (Common XLA requirements)
NPP_LIBS=("libnppc" "libnppial" "libnppicc" "libnppidei" "libnppif" "libnppig" "libnppim" "libnppist" "libnppisu" "libnppitc" "libnpps")
for lib in "${NPP_LIBS[@]}"; do
    map_lib "$lib" "11 12"
done

# 3. cuDNN & Sub-layers (Target 8 and 9)
# If venv has cuDNN 9, we prioritize it and ensure NO version 8 links interfere for the same base name
if [[ "$VENV_LIB_DIR" == *".venv"* ]]; then
    PROJECT_ROOT=$(pwd)
    CUDNN_VENV_DIR_REL=$(find .venv -name "libcudnn_ops.so.9" -exec dirname {} \; | head -n 1)
    if [ -n "$CUDNN_VENV_DIR_REL" ]; then
        CUDNN_VENV_DIR="$PROJECT_ROOT/$CUDNN_VENV_DIR_REL"
        echo "Detected cuDNN 9 in venv at $CUDNN_VENV_DIR. Enforcing version 9 consistency..."
        
        # Helper to link venv cuDNN
        link_venv_cudnn() {
            local src_file=$1
            local dst_base=$2
            sudo ln -sf "$CUDNN_VENV_DIR/$src_file" "$DST_DIR/$dst_base.so.9"
            sudo ln -sf "$CUDNN_VENV_DIR/$src_file" "$DST_DIR/$dst_base.so.8" # Lie to TF if it asks for 8
            sudo ln -sf "$CUDNN_VENV_DIR/$src_file" "$DST_DIR/$dst_base.so"
        }

        link_venv_cudnn "libcudnn.so.9" "libcudnn"
        link_venv_cudnn "libcudnn_ops.so.9" "libcudnn_ops_infer"
        link_venv_cudnn "libcudnn_ops.so.9" "libcudnn_ops_train"
        link_venv_cudnn "libcudnn_cnn.so.9" "libcudnn_cnn_infer"
        link_venv_cudnn "libcudnn_cnn.so.9" "libcudnn_cnn_train"
        link_venv_cudnn "libcudnn_adv.so.9" "libcudnn_adv_infer"
        link_venv_cudnn "libcudnn_adv.so.9" "libcudnn_adv_train"
        # Also the consolidated ones
        link_venv_cudnn "libcudnn_ops.so.9" "libcudnn_ops"
        link_venv_cudnn "libcudnn_cnn.so.9" "libcudnn_cnn"
        link_venv_cudnn "libcudnn_adv.so.9" "libcudnn_adv"
        link_venv_cudnn "libcudnn_graph.so.9" "libcudnn_graph"
        link_venv_cudnn "libcudnn_heuristic.so.9" "libcudnn_heuristic"
        link_venv_cudnn "libcudnn_engines_runtime_compiled.so.9" "libcudnn_engines_runtime_compiled"
        link_venv_cudnn "libcudnn_engines_precompiled.so.9" "libcudnn_engines_precompiled"
    else
        # Fallback to system version 8
        map_lib "libcudnn" "8 9"
        map_lib "libcudnn_ops_infer" "8"
        map_lib "libcudnn_ops_train" "8"
        map_lib "libcudnn_cnn_infer" "8"
        map_lib "libcudnn_cnn_train" "8"
        map_lib "libcudnn_adv_infer" "8"
        map_lib "libcudnn_adv_train" "8"
    fi
else
    map_lib "libcudnn" "8 9"
    map_lib "libcudnn_ops_infer" "8 9"
    map_lib "libcudnn_ops_train" "8 9"
    map_lib "libcudnn_cnn_infer" "8 9"
    map_lib "libcudnn_cnn_train" "8 9"
    map_lib "libcudnn_adv_infer" "8 9"
    map_lib "libcudnn_adv_train" "8 9"
fi

# 4. Link ptxas
if [ -f "/usr/bin/ptxas" ]; then
    echo "Linking ptxas"
    sudo ln -sf "/usr/bin/ptxas" "/usr/local/cuda/bin/ptxas"
fi

echo "Refreshing ldconfig..."
sudo ldconfig

echo -e "\n--- INTERNAL VERIFICATION ---"
BROKEN=0
for f in $DST_DIR/*.so*; do
    if [ ! -e "$f" ]; then
        echo "FAILED: Broken link $f"
        BROKEN=1
    fi
done

if [ $BROKEN -eq 0 ]; then
    echo "SUCCESS: All links verified."
else
    echo "ERROR: Some links are broken. Check library presence in $SRC_DIR"
fi
echo "Done!"

