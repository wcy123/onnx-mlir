# MLIR Compilation Overview

**Date:** 2026-02-10
**Status:** Design Document
**Related:** [ARCHITECTURE.md](ARCHITECTURE.md), [mlir/](mlir/)

---

## Purpose

This document provides a high-level overview of how ONNX models are compiled through MLIR to native DLL code for AMD ROCm GPUs. For detailed designs, see the documents in `doc/mlir/`.

---

## Compilation Pipeline Overview

```
ONNX Model (.onnx)
  ↓
┌──────────────────────────────────────────────────┐
│ MorphiZen Framework (existing)                   │
│ - Parse ONNX to MLIR (func, arith, ONNX dialect) │
└──────────────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────────────┐
│ Pass 1: OnnxToHip                                │
│ - Convert ONNX operations → HIP operations       │
│ - Extract constants to globals                   │
│ - Generate @main and constant helpers            │
└──────────────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────────────┐
│ Pass 2: HipToLLVM                                │
│ - Lower HIP operations → MIOpen/hipBLAS calls    │
│ - Generate wrapper functions                     │
│ - Convert types to LLVM                          │
└──────────────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────────────┐
│ Pass 3: GenerateInterfacePass                    │
│ - Generate C interface wrappers                  │
│ - inference_init/compute/cleanup                 │
└──────────────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────────────┐
│ LLVM Backend                                     │
│ - Translate to LLVM IR                           │
│ - Compile to native object file                  │
│ - Link to DLL                                    │
└──────────────────────────────────────────────────┘
  ↓
Native DLL (inference.dll / inference.so)
  ↓
Embedded in ONNX EPContext
```

For detailed transformation through each stage, see [mlir/LOWERING-PIPELINE.md](mlir/LOWERING-PIPELINE.md).

---

## Output: 3-Function Interface

The compiled DLL exports exactly 3 functions:

### 1. inference_init
```c
int inference_init(void** out_state);
```
- Allocates runtime state (context)
- Creates GPU handles (stream, MIOpen, hipBLAS)
- Uploads constants (weights) to GPU
- Returns state pointer via out parameter

### 2. inference_compute
```c
int inference_compute(void* state, span_t* inputs, span_t* outputs);
```
- Parses input/output tensors from span_t
- Executes GPU operations (all inline)
- Returns status code (0 = success)

### 3. inference_cleanup
```c
int inference_cleanup(void* state);
```
- Frees GPU constant memory
- Destroys GPU handles
- Frees state structure

---

### Parameter Details

**About the `state` parameter:** An opaque pointer (`void*`) representing the execution context. Internally contains GPU handles (stream, MIOpen, hipBLAS) and pre-uploaded constant pointers. Allocated once in `init`, used throughout execution, freed in `cleanup`. See [STATE-AND-CONTEXT.md](STATE-AND-CONTEXT.md) for details.

**About `span_t` and tensor interface:**

The C interface uses these structs to pass tensors between CustomOp and compiled DLL:

```c
// Single tensor descriptor
typedef struct {
    void* data;        // Pointer to tensor data (CPU or GPU memory)
    int64_t* shape;    // Runtime dimensions (e.g., [2, 3, 256, 256])
    int rank;          // Number of dimensions (e.g., 4)
    int data_type;     // Element type (float32, int64, etc.)
} tensor_t;

// Array of tensors
typedef struct {
    tensor_t* data;    // Array of tensor descriptors
    size_t count;      // Number of tensors (N inputs or M outputs)
} span_t;
```

**Usage in inference_compute:**
- User provides `inputs` (span_t with N tensors) and `outputs` (span_t with M tensors)
- `inference_compute` loads runtime dimensions from `tensor_t.shape`
- Builds MLIR memref structs with these runtime values
- Calls `@main` which executes GPU computation

See [mlir/INTERFACE-DESIGN.md](mlir/INTERFACE-DESIGN.md) for complete interface specification.

---

## Two-Layer Architecture

**Layer 1: C Interface (Public API)**
- `inference_init`, `inference_compute`, `inference_cleanup`
- Exported from DLL for CustomOp
- Handle span_t ↔ memref impedance mismatch

**Layer 2: Internal MLIR Functions (Private)**
- `@main(context, inputs, outputs) -> i32` - Actual computation
- `initialize_constants(context) -> i32` - Upload constants
- `release_constants(context) -> i32` - Free GPU memory
- `get_constant_count() -> i64` - Metadata helper

**Why two layers?**
- C interface uses `span_t` (dynamic, opaque)
- MLIR uses `memref` (typed, structured)
- Wrappers bridge the gap

See [mlir/INTERFACE-DESIGN.md](mlir/INTERFACE-DESIGN.md) for details.

---

## Dynamic Shape Support

**Critical requirement:** All components must support dynamic shapes from Day 1.

- **Rank**: Compile-time known (e.g., 4D tensor)
- **Dimensions**: Runtime values (loaded from tensor_t.shape)
- **Strides**: Calculated at runtime from dimensions
- **Interface**: No changes needed for dynamic vs static shapes

See [DYNAMIC-SHAPE-DESIGN.md](DYNAMIC-SHAPE-DESIGN.md) for comprehensive details.

---

## Compilation to DLL

**TODO**: Document LLVM IR generation, native compilation, linking, and EPContext embedding details.

---

## Detailed Design Documents

| Document | Description |
|----------|-------------|
| [mlir/MODULE-STRUCTURE.md](mlir/MODULE-STRUCTURE.md) | MLIR module structure and organization |
| [mlir/LOWERING-PIPELINE.md](mlir/LOWERING-PIPELINE.md) | Pass pipeline and transformation stages |
| [mlir/INTERFACE-DESIGN.md](mlir/INTERFACE-DESIGN.md) | C interface and GenerateInterfacePass prerequisites |
| [mlir/HIP-DIALECT-DESIGN.md](mlir/HIP-DIALECT-DESIGN.md) | HIP context, types, and wrapper functions |
| [mlir/CONSTANT-MANAGEMENT.md](mlir/CONSTANT-MANAGEMENT.md) | Constant handling (globals, upload, retrieval) |
| [STATE-AND-CONTEXT.md](STATE-AND-CONTEXT.md) | Runtime state/context structure |
| [DYNAMIC-SHAPE-DESIGN.md](DYNAMIC-SHAPE-DESIGN.md) | Dynamic shape support |

---

## Related Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system architecture
- [STATE-AND-CONTEXT.md](STATE-AND-CONTEXT.md) - Runtime state structure and lifecycle
- [DEMO.md](DEMO.md) - End-to-end demo walkthrough
