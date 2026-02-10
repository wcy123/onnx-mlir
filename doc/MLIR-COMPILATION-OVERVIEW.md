# MLIR Compilation Overview

**Date:** 2026-02-10
**Status:** Design Document
**Related:** [ARCHITECTURE.md](ARCHITECTURE.md), [mlir/](mlir/)

---

## Purpose

This document provides a high-level overview of how ONNX models are compiled through MLIR to native DLL code for AMD ROCm GPUs. For detailed designs, see the documents in `doc/mlir/`.

---

## Key Design Decisions

1. **Inline lowering**: Each ONNX operation is lowered inline (no function-per-node)
2. **Explicit state passing**: State is passed as function parameter (no thread-local globals)
3. **Constants in DLL**: Weights embedded in `.data` section, uploaded to GPU in `init`
4. **Direct C interface mapping**: MLIR function signatures match C interface exactly
5. **Dynamic shapes from Day 1**: Rank is compile-time, dimension values are runtime

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
int inference_compute(void* state, span_t inputs, span_t outputs);
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

## Runtime State Structure

```c
struct HipExecutionContext {
    hipStream_t stream;              // GPU stream for async operations
    miopenHandle_t miopenHandle;     // MIOpen library handle
    hipblasLtHandle_t hipblasHandle; // hipBLAS library handle
    void** gpu_constants;            // Dynamically allocated array of GPU pointers
};
```

- Allocated in `inference_init`
- Passed to all operations
- Freed in `inference_cleanup`

See [STATE-AND-CONTEXT.md](STATE-AND-CONTEXT.md) for details.

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

### LLVM IR Generation

```cpp
mlir::registerLLVMDialectTranslation(*context);
auto llvmModule = mlir::translateModuleToLLVMIR(mlirModule, llvmContext);
```

### Native Code Compilation

```cpp
// Set target (x86_64-pc-windows-msvc or x86_64-unknown-linux-gnu)
llvmModule->setTargetTriple(TargetTriple.normalize());
llvmModule->setDataLayout(targetMachine->createDataLayout());

// Emit object file
llvm::legacy::PassManager pass;
targetMachine->addPassesToEmitFile(pass, dest, nullptr, llvm::CGFT_ObjectFile);
pass.run(*llvmModule);
```

### Linking

```bash
# Windows (MSVC)
link.exe /DLL /OUT:inference.dll inference.obj hip.lib miopen.lib hipblaslt.lib

# Linux (GCC)
gcc -shared -o inference.so inference.o -lhip -lmiopen -lhipblaslt
```

### EPContext Embedding

- DLL bytes embedded in ONNX model EPContext node
- CustomOp loads DLL from memory (MemoryModule library)
- No disk I/O at runtime
- Fast startup (~1-10ms)

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

## Key Design Principles

1. **Simplicity first**: Start with LLVM dialect directly, refine later
2. **Explicit state**: Pass as parameter, no globals
3. **Inline operations**: All computation in @main
4. **C ABI compatibility**: Clean integration with CustomOp
5. **Performance**: AOT compilation, no JIT overhead
6. **Dynamic shapes**: Runtime dimensions from Day 1

---

## Related Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system architecture
- [DEMO.md](DEMO.md) - End-to-end demo walkthrough
