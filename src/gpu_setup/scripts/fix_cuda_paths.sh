#!/bin/bash

# CUDA Path Fixup Script - Improved version
# This script creates symbolic links to CUDA/cuDNN libraries for compatibility
# Usage: bash fix_cuda_paths.sh [--cleanup]

CLEANUP_MODE=false
if [ "$1" = "--cleanup" ]; then
    CLEANUP_MODE=true
fi

# Ensure directories exist
sudo mkdir -p /usr/local/cuda/lib64
sudo mkdir -p /usr/local/cuda/bin

# If cleanup requested, remove all existing broken links first
if [ "$CLEANUP_MODE" = true ]; then
    echo "Cleaning up broken/existing links in /usr/local/cuda/lib64..."
    sudo find /usr/local/cuda/lib64 -type l -delete 2>/dev/null
    echo "Done."
fi

SRC_DIR="/usr/lib/x86_64-linux-gnu"
VENV_LIB_DIR=$(ls -d .venv/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ' ')
CUDA_TOOLKIT_DIR="/usr/local/cuda"
DST_DIR="/usr/local/cuda/lib64"

map_lib() {
    local base=$1
    local target_vers=$2

    local real_file=""
    for dir in $VENV_LIB_DIR $SRC_DIR; do
        [ -z "$dir" ] && continue
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
        local real_path=$(readlink -f "$real_file")
        echo "Found $base: $real_path"

        for ver in $target_vers; do
            echo "  Linking to $DST_DIR/$base.so.$ver"
            sudo ln -sf "$real_path" "$DST_DIR/$base.so.$ver"
        done
        sudo ln -sf "$real_path" "$DST_DIR/$base.so"
    else
        echo "Warning: Could not find $base (skipping)"
    fi
}

# 1. CUDA Core Libraries
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

# 2. NPP Libraries
NPP_LIBS=(
    "libnppc"
    "libnppial"
    "libnppicc"
    "libnppidei"
    "libnppif"
    "libnppig"
    "libnppim"
    "libnppist"
    "libnppisu"
    "libnppitc"
    "libnpps"
)
for lib in "${NPP_LIBS[@]}"; do
    map_lib "$lib" "11 12"
done

# 3. cuDNN Libraries
if [[ "$VENV_LIB_DIR" == *".venv"* ]]; then
    PROJECT_ROOT=$(pwd)
    CUDNN_VENV_DIR_REL=$(find .venv -name "libcudnn_ops.so.9" -exec dirname {} \; | head -n 1)
    if [ -n "$CUDNN_VENV_DIR_REL" ]; then
        CUDNN_VENV_DIR="$PROJECT_ROOT/$CUDNN_VENV_DIR_REL"
        echo "Detected cuDNN 9 in venv at $CUDNN_VENV_DIR. Enforcing version 9 consistency..."

        link_venv_cudnn() {
            local src_file=$1
            local dst_base=$2
            sudo ln -sf "$CUDNN_VENV_DIR/$src_file" "$DST_DIR/$dst_base.so.9"
            sudo ln -sf "$CUDNN_VENV_DIR/$src_file" "$DST_DIR/$dst_base.so.8"
            sudo ln -sf "$CUDNN_VENV_DIR/$src_file" "$DST_DIR/$dst_base.so"
        }

        link_venv_cudnn "libcudnn.so.9" "libcudnn"
        link_venv_cudnn "libcudnn_ops.so.9" "libcudnn_ops_infer"
        link_venv_cudnn "libcudnn_ops.so.9" "libcudnn_ops_train"
        link_venv_cudnn "libcudnn_cnn.so.9" "libcudnn_cnn_infer"
        link_venv_cudnn "libcudnn_cnn.so.9" "libcudnn_cnn_train"
        link_venv_cudnn "libcudnn_adv.so.9" "libcudnn_adv_infer"
        link_venv_cudnn "libcudnn_adv.so.9" "libcudnn_adv_train"
        link_venv_cudnn "libcudnn_ops.so.9" "libcudnn_ops"
        link_venv_cudnn "libcudnn_cnn.so.9" "libcudnn_cnn"
        link_venv_cudnn "libcudnn_adv.so.9" "libcudnn_adv"
        link_venv_cudnn "libcudnn_graph.so.9" "libcudnn_graph"
        link_venv_cudnn "libcudnn_heuristic.so.9" "libcudnn_heuristic"
        link_venv_cudnn "libcudnn_engines_runtime_compiled.so.9" "libcudnn_engines_runtime_compiled"
        link_venv_cudnn "libcudnn_engines_precompiled.so.9" "libcudnn_engines_precompiled"
    else
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

echo ""
echo "--- VERIFICATION REPORT ---"
TOTAL=0
VALID=0
BROKEN=0
for f in $DST_DIR/*.so*; do
    if [ -e "$f" ] || [ -L "$f" ]; then
        TOTAL=$((TOTAL + 1))
        if [ -e "$f" ]; then
            VALID=$((VALID + 1))
        else
            BROKEN=$((BROKEN + 1))
            if [ $BROKEN -le 3 ]; then
                echo "⚠ Broken link: $(basename "$f")"
            fi
        fi
    fi
done

echo ""
echo "Summary:"
echo "  Total links: $TOTAL"
echo "  Valid: $VALID"
echo "  Broken: $BROKEN"

if [ $BROKEN -eq 0 ]; then
    echo ""
    echo "✓ SUCCESS: All library links are valid!"
else
    echo ""
    echo "⚠ WARNING: $BROKEN broken links remain."
    echo "  This typically means those libraries are not installed on this system."
    echo "  The script found and linked what WAS available."
    echo ""
    echo "If you need GPU acceleration, see src/gpu_setup/docs/TROUBLESHOOTING.md"
fi
echo ""
echo "Done!"

