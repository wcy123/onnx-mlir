<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# MLIR Compilation Design Documents

This directory contains detailed design documents for the MLIR-based compilation pipeline.

---

## Document Navigation

### Start Here
- **[../MLIR-COMPILATION-OVERVIEW.md](../MLIR-COMPILATION-OVERVIEW.md)** - High-level overview of the compilation pipeline

### Core Design Documents

| Document | Description | Key Topics |
|----------|-------------|------------|
| [MODULE-STRUCTURE.md](MODULE-STRUCTURE.md) | MLIR module organization | - Module structure after all passes<br>- Internal vs exported functions<br>- Function call graph |
| [LOWERING-PIPELINE.md](LOWERING-PIPELINE.md) | Transformation stages | - ONNX → HIP → LLVM stages<br>- Pass-by-pass IR examples<br>- Dynamic shape handling at each stage |
| [INTERFACE-DESIGN.md](INTERFACE-DESIGN.md) | C interface and prerequisites | - Two-layer architecture<br>- GenerateInterfacePass prerequisites<br>- Error handling strategy |
| [HIP-DIALECT-DESIGN.md](HIP-DIALECT-DESIGN.md) | HIP dialect and wrappers | - !hip.context type design<br>- Wrapper function generation<br>- Dynamic shapes in wrappers |
| [CONSTANT-MANAGEMENT.md](CONSTANT-MANAGEMENT.md) | Constant handling | - Compile-time extraction to globals<br>- Runtime GPU upload<br>- Retrieval and cleanup |

### Related Documents (in parent directory)

| Document | Description |
|----------|-------------|
| [../STATE-AND-CONTEXT.md](../STATE-AND-CONTEXT.md) | Runtime state/context structure |
| [../DYNAMIC-SHAPE-DESIGN.md](../DYNAMIC-SHAPE-DESIGN.md) | Comprehensive dynamic shape support |
| [../DEMO.md](../DEMO.md) | End-to-end demo walkthrough |
| [../ARCHITECTURE.md](../ARCHITECTURE.md) | Overall system architecture |

---

## Quick Reference

### For Implementing Passes

**OnnxToHip Pass:**
- See: [LOWERING-PIPELINE.md](LOWERING-PIPELINE.md) - Stage 2
- Generates: @main, constant helpers (initialize_constants, release_constants, get_constant_count)
- Key: Extract constants to llvm.mlir.global

**HipToLLVM Pass:**
- See: [LOWERING-PIPELINE.md](LOWERING-PIPELINE.md) - Stage 3
- See: [HIP-DIALECT-DESIGN.md](HIP-DIALECT-DESIGN.md) - Wrapper generation
- Generates: Wrapper functions for MIOpen/hipBLAS calls
- Key: Support dynamic shapes by extracting dimensions from memref structs

**GenerateInterfacePass:**
- See: [INTERFACE-DESIGN.md](INTERFACE-DESIGN.md) - Prerequisites section
- Generates: inference_init, inference_compute, inference_cleanup
- Key: Parse span_t, load runtime dimensions, build memref structs

### For Understanding Data Flow

**Constant lifecycle:**
1. ONNX proto → llvm.mlir.global (compile-time)
2. Global → GPU memory via initialize_constants (runtime init)
3. GPU pointer retrieval via hip_get_constant (runtime compute)
4. GPU memory freed via release_constants (runtime cleanup)

See [CONSTANT-MANAGEMENT.md](CONSTANT-MANAGEMENT.md) for details.

**Dynamic shape flow:**
1. User: tensor_t.shape = [batch, channels, height, width]
2. inference_compute: Load dimensions from tensor_t.shape (runtime!)
3. Build memref struct with runtime sizes/strides
4. @main: Pass memref structs to wrappers
5. Wrappers: Extract dimensions, pass to MIOpen

See [../DYNAMIC-SHAPE-DESIGN.md](../DYNAMIC-SHAPE-DESIGN.md) and [HIP-DIALECT-DESIGN.md](HIP-DIALECT-DESIGN.md).

---

## Document Organization

This directory was created to organize the previously monolithic MLIR-COMPILATION-DESIGN.md (1870 lines) into focused, maintainable documents.

**Design principles:**
- Each document covers one major topic
- Cross-references between documents
- Code examples inline
- Clear prerequisites and dependencies
