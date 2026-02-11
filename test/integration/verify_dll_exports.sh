#!/bin/bash
# Verify shared library exports on Linux using nm

if [ -z "$1" ]; then
    echo "Usage: $0 <path-to-so>"
    exit 1
fi

SO_PATH="$1"

echo "Verifying shared library exports: $SO_PATH"
echo

# Check if nm is available
if ! command -v nm &> /dev/null; then
    echo "Error: nm not found. Please install binutils."
    exit 1
fi

echo "=== Shared Library Exports ==="
nm -D "$SO_PATH" | grep " T "

echo
echo "=== Checking Required Symbols ==="

# Check for required symbols
if nm -D "$SO_PATH" | grep " T " | grep -q "inference_init"; then
    echo "[OK] inference_init found"
else
    echo "[FAIL] inference_init NOT found"
fi

if nm -D "$SO_PATH" | grep " T " | grep -q "inference_compute"; then
    echo "[OK] inference_compute found"
else
    echo "[FAIL] inference_compute NOT found"
fi

if nm -D "$SO_PATH" | grep " T " | grep -q "inference_cleanup"; then
    echo "[OK] inference_cleanup found"
else
    echo "[FAIL] inference_cleanup NOT found"
fi

echo
echo "Verification complete."
