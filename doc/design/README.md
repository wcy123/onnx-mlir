<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Design Documents

Architecture and design decisions for the ONNX HIP/DNN Execution Provider.

All documents in this directory (including subdirectories) have **Document Type: Design**.

## Document Status

| Document | Status | Date | Description |
|----------|--------|------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Self-Reviewed | 2026-02-12 | Entry point with 7 major design decisions |
| [MLIR-COMPILATION-OVERVIEW.md](MLIR-COMPILATION-OVERVIEW.md) | Self-Reviewed | 2026-02-12 | Compilation pipeline overview |
| [RUNTIME-ARCHITECTURE.md](RUNTIME-ARCHITECTURE.md) | Self-Reviewed | 2026-02-12 | Runtime state and context design |
| [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md) | Self-Reviewed | 2026-02-13 | Model constants (weights, biases) |
| [MEMORY-MANAGEMENT.md](MEMORY-MANAGEMENT.md) | Draft | 2026-02-09 | GPU memory allocation strategy |
| [DYNAMIC-SHAPE-DESIGN.md](DYNAMIC-SHAPE-DESIGN.md) | Draft | 2026-02-10 | Runtime-determined tensor dimensions |
| [EPCONTEXT-MEMORY-OPTIMIZATION.md](EPCONTEXT-MEMORY-OPTIMIZATION.md) | Draft | 2026-02-13 | EP context memory optimization |
| [alternatives/NATIVE-VS-IR-COMPARISON.md](alternatives/NATIVE-VS-IR-COMPARISON.md) | Draft | 2026-02-11 | Native DLL vs LLVM IR storage |
| [mlir/MODULE-STRUCTURE.md](mlir/MODULE-STRUCTURE.md) | Draft | - | MLIR module organization |
| [mlir/LOWERING-PIPELINE.md](mlir/LOWERING-PIPELINE.md) | Self-Reviewed | 2026-02-12 | Transformation stages |
| [mlir/INTERFACE-DESIGN.md](mlir/INTERFACE-DESIGN.md) | Draft | - | C interface and prerequisites |
| [mlir/HIP-DIALECT-DESIGN.md](mlir/HIP-DIALECT-DESIGN.md) | Draft | - | HIP dialect and wrappers |
| [mlir/passes/OnnxToHip.md](mlir/passes/OnnxToHip.md) | Self-Reviewed | 2026-02-13 | ONNX → HIP dialect lowering |
| [mlir/passes/HipToLLVM.md](mlir/passes/HipToLLVM.md) | Self-Reviewed | 2026-02-13 | HIP → LLVM lowering and wrappers |
| [mlir/passes/GenerateInterfacePass.md](mlir/passes/GenerateInterfacePass.md) | Self-Reviewed | 2026-02-12 | C interface generation pass |
| [mlir/passes/WHY-HIP-WRAPPERS.md](mlir/passes/WHY-HIP-WRAPPERS.md) | Draft | - | Justification for wrapper approach |
