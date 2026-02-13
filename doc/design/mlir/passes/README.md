<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# MLIR Passes Documentation

Documentation for individual MLIR transformation passes in the compilation pipeline.

## Pass Documentation

### Major Lowering Passes
- [OnnxToHip.md](OnnxToHip.md) - First major lowering (ONNX → HIP dialect)
- [HipToLLVM.md](HipToLLVM.md) - Wrapper generation for GPU calls (HIP → LLVM)
- [GenerateInterfacePass.md](GenerateInterfacePass.md) - C interface generation (init/compute/cleanup)

### Supporting Documentation
- [WHY-HIP-WRAPPERS.md](WHY-HIP-WRAPPERS.md) - Justification for wrapper approach in HipToLLVM

## Pass Pipeline

The complete lowering pipeline:
```
ONNX Dialect → HIP Dialect → LLVM Dialect → LLVM IR
     |              |              |
  OnnxToHip    HipToLLVM   GenerateInterface
```

See [../LOWERING-PIPELINE.md](../LOWERING-PIPELINE.md) for detailed IR examples at each stage.
