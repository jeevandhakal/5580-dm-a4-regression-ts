#!/bin/bash
# Cleanup broken CUDA symlinks and prepare for fresh setup

echo "======================================================================"
echo "CUDA Symlink Cleanup Utility"
echo "======================================================================"
echo ""

if [ ! -d "/usr/local/cuda/lib64" ]; then
    echo "✓ /usr/local/cuda/lib64 does not exist (no cleanup needed)"
    exit 0
fi

echo "Current status of /usr/local/cuda/lib64:"
echo "  Total entries: $(ls -1 /usr/local/cuda/lib64 2>/dev/null | wc -l)"

BROKEN=0
VALID=0
for f in /usr/local/cuda/lib64/*.so*; do
    [ -e "$f" ] && [ -L "$f" ] && [ ! -e "$(readlink "$f")" ] && BROKEN=$((BROKEN+1))
    [ -e "$f" ] && VALID=$((VALID+1))
done

echo "  Valid links: $VALID"
echo "  Broken links: $BROKEN"
echo ""

if [ $BROKEN -eq 0 ]; then
    echo "✓ No broken links found"
    echo ""
    echo "Note: If you want to completely reset and re-run fix_cuda_paths.sh:"
    echo "  bash src/gpu_setup/scripts/fix_cuda_paths.sh"
else
    echo "⚠ Found $BROKEN broken links. Removing..."
    sudo find /usr/local/cuda/lib64 -type l ! -exec test -e {} \; -delete
    echo "✓ Broken links removed"
    echo ""
    echo "Next steps:"
    echo "  1. If CUDA/cuDNN is installed: bash src/gpu_setup/scripts/fix_cuda_paths.sh"
    echo "  2. If not: python3 src/gpu_setup/tools/gpu_recovery.py  (for diagnostics)"
fi

echo ""
echo "Done!"

