#!/bin/bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

# Verify all buffer offsets are 4096-byte aligned from LLVM IR

LLVM_IR="$1"
ALIGNMENT=4096

if [ -z "$LLVM_IR" ]; then
  echo "Usage: $0 <llvm_ir_file>"
  exit 1
fi

if [ ! -f "$LLVM_IR" ]; then
  echo "ERROR: File not found: $LLVM_IR"
  exit 1
fi

# Extract offsets from hipdnn.buffer_offsets metadata
# Format: hipdnn.buffer_offsets = !{...} or array<i64: 0, 3211264, ...>
offsets=$(grep -oP 'hipdnn\.buffer_offsets.*?array<i64:\s*\K[^>]+' "$LLVM_IR" | head -1)

if [ -z "$offsets" ]; then
  echo "ERROR: No hipdnn.buffer_offsets found in $LLVM_IR"
  exit 1
fi

echo "Found offsets: $offsets"

# Parse comma-separated offsets
IFS=',' read -ra OFFSET_ARRAY <<< "$offsets"
all_aligned=true

for offset in "${OFFSET_ARRAY[@]}"; do
  # Trim whitespace
  offset=$(echo "$offset" | tr -d ' ')

  # Check if offset is a number
  if ! [[ "$offset" =~ ^[0-9]+$ ]]; then
    echo "WARNING: Skipping non-numeric offset: $offset"
    continue
  fi

  remainder=$((offset % ALIGNMENT))
  if [ $remainder -ne 0 ]; then
    echo "ERROR: Offset $offset not aligned to $ALIGNMENT (remainder: $remainder)"
    all_aligned=false
  else
    echo "✓ Offset $offset is aligned to $ALIGNMENT"
  fi
done

if [ "$all_aligned" = true ]; then
  echo "SUCCESS: All offsets are 4096-byte aligned"
  exit 0
else
  echo "FAILURE: Some offsets are not aligned"
  exit 1
fi
