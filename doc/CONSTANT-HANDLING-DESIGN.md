<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Constant Handling Design
## ONNX Initializers in MLIR-based Compilation Pipeline

**Note:** This is the authoritative source for constant handling design and implementation.

**Status**: Design in Progress
**Date**: 2026-02-10
**Related**: ARCHITECTURE.md, MLIR-COMPILATION-DESIGN.md, STATE-AND-CONTEXT.md

---

## Executive Summary

This document defines the architecture for handling ONNX model constants (weights, biases, embeddings) in the MLIR-based compilation pipeline for the HipDNN Execution Provider.

**Key innovation**: Constants are embedded in the compiled DLL as LLVM globals, uploaded to GPU once during initialization, and accessed through a state structure—achieving clean function signatures while maintaining optimal performance.

---

## Problem Statement

Deep learning models (e.g., ResNet50) contain hundreds of constant tensors:
- 100+ convolutional layers, each with weights and biases
- Batch normalization layers with scale/bias/mean/variance tensors
- Embedding layers with large lookup tables

In ONNX-MLIR, these appear as `onnx.Constant` operations within function bodies.

**Requirements**:
- Clean function signatures that scale regardless of model size
- Efficient constant management (load once, use many times)
- Type-safe constant access
- Single-artifact deployment

---

## Core Design Decisions

### Decision 1: State-Based Constant Management

**Choice**: Store constants in execution state with pre-uploaded GPU pointers.

**Rationale**:
- Constants have different lifecycle than inputs/outputs (loaded once vs per-inference)
- GPU memory allocation/upload is expensive—do it once, reuse many times
- Matches industry standard execution provider patterns (TensorRT, QNN, VitisAI)
- Clean separation of initialization vs execution concerns

**Architecture** (see [STATE-AND-CONTEXT.md](STATE-AND-CONTEXT.md) for full design):
```c
// Internal state structure (opaque to C interface)
struct HipExecutionState {
    hipStream_t stream;
    miopenHandle_t miopenHandle;
    hipblasLtHandle_t hipblasHandle;
    void** gpu_constants;  // Array of pre-uploaded constant pointers
};
```

**In MLIR**: Functions receive `%ctx: !hip.context` parameter to access pre-uploaded constants.

### Decision 2: Embed Constants in Compiled DLL

**Choice**: Generate `llvm.mlir.global` operations with embedded `dense<...>` constant data.

**Rationale**:
- DLL already contains compiled code—colocating data avoids external dependencies
- LLVM globals provide type-safe, addressable storage in CPU memory
- No runtime parsing of ONNX model needed
- Enables compiler optimizations on constant data
- Simplifies deployment (single artifact)

**Trade-offs**:
- ✅ Self-contained DLL (no external files)
- ✅ Type safety at compile time
- ⚠️ Larger DLL size (~100KB-1MB for typical models)
- ⚠️ GPU architecture-specific (must match runtime hardware)

### Decision 3: Module-Level MLIR Pass

**Choice**: `ConvertOnnxToHipPass` operates on `ModuleOp`, not `func::FuncOp`.

**Rationale**:
- Need to discover constants across **all** functions (including subgraphs from ONNX If/Loop/Scan)
- Need to create module-level `llvm.mlir.global` operations
- Need to generate module-level initialization functions
- Need shared constant registry with consistent global indexing

Module-level pass provides full visibility and control over all functions.

### Decision 4: Sequential Global Indexing

**Choice**: Assign each constant a unique index (0, 1, 2, ..., N-1) in discovery order.

**Rationale**:
- Simple, deterministic, reproducible
- Direct array indexing in `state->gpu_constants[]`
- No hash collisions or lookup overhead
- Easy to debug (indices match discovery order)

Trade-off: No automatic deduplication (future optimization).

### Decision 5: HIP Dialect Operations for Constants

**Choice**: Define three HIP operations: `hip.get_constant`, `hip.upload_constant`, `hip.release_constant`.

**Rationale**:
- **Clean abstraction layers**: ONNX→HIP stays in HIP dialect, HIP→LLVM handles lowering to runtime calls
- **Semantic clarity**: Each operation has clear, unambiguous meaning
- **Optimization flexibility**: HIP→LLVM can choose naive (individual hipMalloc) or optimized (batched) lowering
- **Future extensibility**: Can add attributes for optimization hints (pinned memory, async upload, etc.)

**Operations**:
```tablegen
// Get reference to pre-uploaded constant (used in @main)
hip.get_constant(%ctx, index) -> memref

// Upload constant to GPU (used in @initialize_constants)
hip.upload_constant(%ctx, index, cpu_data, size)

// Free GPU memory (used in @release_constants)
hip.release_constant(%ctx, index)
```

### Decision 6: Generated Initialization Functions

**Choice**: ConvertOnnxToHipPass generates three metadata functions in the compiled DLL.

**Functions**:
- `get_constant_count() -> i64`: Returns total number of constants
- `initialize_constants(state*) -> i32`: Uploads all constants to GPU
- `release_constants(state*) -> i32`: Frees all GPU memory

**Rationale**:
- Runtime knows constant count without parsing ONNX model
- Initialization code is compiled (fast, no JIT overhead)
- Standard C ABI enables simple dlsym resolution
- Separation of concerns: compiler generates, runtime invokes

### Decision 7: ONNX Function Identification (in ONNX→HIP Pass)

**Choice**: The ONNX→HIP pass processes only functions with tensor types + ONNX dialect operations.

**Rationale**:
- ONNX→HIP pass can coexist with other MLIR passes in pipeline
- Order-independent in pass manager
- Won't accidentally transform non-ONNX helper functions (e.g., LLVM utility functions)
- Robust to future additions of utility functions at other dialect levels
- **Idempotent**: If the pass runs twice by accident, already-transformed functions are skipped (they have `memref` types and HIP operations, not `tensor` types and ONNX operations)

**Identification criteria in `ConvertOnnxToHipPass`**:
1. Function signature contains `TensorType`
2. Function body contains ONNX dialect operations

This ensures the pass only transforms ONNX functions, leaving other functions untouched. After transformation, functions have `memref` types and HIP dialect operations, so they won't match the criteria on subsequent runs.

---

## Architecture Overview

### High-Level Flow

```
ONNX Model
    ↓ (ONNX-MLIR import)
MLIR with onnx.Constant operations
    ↓ (ConvertOnnxToHipPass - MODULE-level)
    │
    ├─→ Discovery: Find all onnx.Constant, assign indices 0..N-1
    ├─→ Generate: llvm.mlir.global for each constant (embed data)
    ├─→ Generate: @get_constant_count(), @initialize_constants(), @release_constants()
    └─→ Transform: Replace onnx.Constant with hip.get_constant(%ctx, index)
    ↓
HIP dialect with state-based constant access
    ↓ (ConvertHipToLLVMPass)
LLVM dialect with HIP runtime calls
    ↓ (LLVM compilation)
Compiled DLL with:
  - Embedded constant data (LLVM globals)
  - Initialization functions
  - Inference function using gpu_constants[]
```

### Runtime Initialization

The generated constant management functions are **internal** to the compiled DLL, called from within `inference_init`:

```c
// Inside compiled DLL (generated by ONNX→HIP pass):

// Helper function: returns constant count (generated by pass)
int64_t get_constant_count() {
  return 200;  // Compile-time known
}

// Helper function: uploads all constants (generated by pass)
int initialize_constants(void* state) {
  // Phase 1 naive implementation (individual allocations):
  // For each constant:
  //   - Get CPU pointer: llvm.mlir.addressof @constant_N
  //   - Allocate GPU: hipMalloc (individual allocation)
  //   - Upload: hipMemcpy(GPU, CPU, size, H2D)
  //   - Store in state->gpu_constants[N]
  //
  // Future optimizations (see Appendix C):
  //   - Batched upload: single hipMemcpy for all constants
  //   - Pinned memory: hipMallocHost for faster transfers
  //   - Memory pooling: single allocation with offset management
  return 0;
}

// Helper function: releases all constants (generated by pass)
int release_constants(void* state) {
  // For each constant:
  //   - hipFree(state->gpu_constants[N])
  return 0;
}

// Public entry point (called by CustomOp via dlsym)
int inference_init(void** out_state) {
  // 1. Allocate state structure
  HipExecutionState* state = new HipExecutionState();

  // 2. Create GPU handles
  hipStreamCreate(&state->stream);
  miopenCreate(&state->miopenHandle);
  // ...

  // 3. Allocate constant pointer array
  state->gpu_constants = new void*[get_constant_count()];  // Internal call

  // 4. Upload all constants to GPU
  initialize_constants(state);  // Internal call

  // 5. Return state
  *out_state = state;
  return 0;
}
```

**Key point**: `get_constant_count`, `initialize_constants`, and `release_constants` are **internal helper functions** within the compiled DLL, not external API. Only `inference_init/compute/cleanup` are exposed via dlsym.

### Inference Execution

```mlir
// BEFORE (ONNX dialect):
func.func @main(%input: tensor<1x3x224x224xf32>) -> tensor<...> {
  %w = "onnx.Constant"() {value = dense<...>}
  %conv = "onnx.Conv"(%input, %w) {...}
  return %conv
}

// AFTER (HIP dialect):
func.func @main(%ctx: !hip.context,
                %input: memref<1x3x224x224xf32, 1>,
                %output: memref<...>) -> i32 {
  // Get pre-uploaded constant from state
  %w = hip.get_constant(%ctx, 0) : (!hip.context, i64) -> memref<...>

  // Use constant in computation
  hip.conv(%ctx, %input, %w, %output) {...}

  %success = llvm.mlir.constant(0 : i32) : i32
  return %success : i32
}
```

**Key insight**: No constant arguments—function signature stays clean regardless of model size.

---

## Design Principles

1. **Separation of Concerns**: Initialization (once) vs execution (many times)
2. **Single Source of Truth**: Constant data embedded in DLL, no external files
3. **Clean Abstractions**: Each MLIR dialect level maintains semantic clarity
4. **Performance**: Pre-upload eliminates repeated allocation/transfer overhead
5. **Scalability**: Design handles 10 constants or 10,000 constants equally well
6. **Industry Alignment**: Follows patterns from TensorRT EP, QNN EP, VitisAI EP

---

## Open Design Questions

### Question 3: Subgraph Constant Handling

**Context**: ONNX control flow operators (If, Loop, Scan) create subgraphs as separate `func.func`. In original ONNX, constants may be passed as arguments to subgraphs.

**Example (BEFORE conversion)**:
```mlir
func.func @main_graph(%input: tensor<...>) -> tensor<...> {
  %w0 = "onnx.Constant"() {value = dense<...>}  // Constant defined in parent
  %result = func.call @subgraph_if_then(%input, %w0)  // Passed to subgraph
  ...
}

func.func @subgraph_if_then(%arg0: tensor<...>, %arg1: tensor<...>) -> tensor<...> {
  %w1 = "onnx.Constant"() {value = dense<...>}  // Local constant
  %conv = "onnx.Conv"(%arg0, %arg1, %w1)  // Uses both passed (%arg1) and local (%w1)
  ...
}
```

**Design decision**: Subgraphs receive only `%ctx` and load constants from state (consistent with design principle of eliminating constant arguments)

**After conversion**:
```mlir
func.func @main(%ctx: !hip.context, %input: memref<...>, %output: memref<...>) -> i32 {
  %temp = hip.alloc(%ctx) : memref<...>
  func.call @subgraph_if_then(%ctx, %input, %temp)  // Only %ctx, no constants
  ...
}

func.func @subgraph_if_then(%ctx: !hip.context, %arg0: memref<...>, %output: memref<...>) -> i32 {
  // Load constants from state by global index
  %w0 = hip.get_constant(%ctx, 0)  // Was passed as argument in ONNX
  %w1 = hip.get_constant(%ctx, 1)  // Was local constant in ONNX

  hip.conv(%ctx, %arg0, %w0, %w1, %output) {...}
  return %c0_i32 : i32
}
```

**Implementation question** (not yet resolved):
- How does the pass know that `%arg1` in the original ONNX subgraph corresponds to global constant index 0?
- Need to track constant provenance: map ONNX constant SSA values to global indices across function boundaries

**TODO**: Implement constant provenance tracking in Phase 6 (Subgraph Handling).

---

## Appendix A: Implementation Details

### A.1: ONNX Function Identification

```cpp
bool isOnnxFunction(func::FuncOp funcOp) {
  auto funcType = funcOp.getFunctionType();

  // Quick filter: ONNX functions use tensor types
  bool hasTensorTypes = llvm::any_of(funcType.getInputs(), [](Type t) {
    return isa<TensorType>(t);
  }) || llvm::any_of(funcType.getResults(), [](Type t) {
    return isa<TensorType>(t);
  });

  if (!hasTensorTypes)
    return false;

  // Confirm: must have ONNX dialect operations
  bool hasOnnxOps = false;
  funcOp.walk([&](Operation *op) {
    if (auto *dialect = op->getDialect()) {
      if (isa<ONNXDialect>(dialect)) {
        hasOnnxOps = true;
        return WalkResult::interrupt();
      }
    }
  });

  return hasOnnxOps;
}
```

### A.2: Constant Discovery

```cpp
struct ConstantInfo {
  size_t globalIndex;      // Index in state->gpu_constants[]
  ElementsAttr value;      // The dense<...> constant data
  Type type;               // tensor<64x3x3x3xf32>
  size_t sizeInBytes;      // For allocation/transfer
  StringRef name;          // Generated name (e.g., "constant_0")
};

DenseMap<Value, ConstantInfo> constantRegistry;
size_t nextGlobalIndex = 0;

// Walk all ONNX functions
for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
  if (!isOnnxFunction(funcOp))
    continue;

  // Discover constants in this function
  funcOp.walk([&](ONNXConstantOp constOp) {
    ConstantInfo info;
    info.globalIndex = nextGlobalIndex++;
    info.value = constOp.getValueAttr().cast<ElementsAttr>();
    info.type = constOp.getType();
    info.sizeInBytes = calculateSizeInBytes(info.type);
    info.name = "constant_" + std::to_string(info.globalIndex);

    constantRegistry[constOp.getResult()] = info;
  });
}
```

### A.3: LLVM Global Generation

```cpp
OpBuilder builder(moduleOp.getBodyRegion());

for (auto& [value, info] : constantRegistry) {
  // Create global constant with embedded data
  auto globalOp = builder.create<LLVM::GlobalOp>(
    moduleOp.getLoc(),
    convertTypeToLLVM(info.type),
    /*isConstant=*/true,
    LLVM::Linkage::Internal,
    info.name,
    info.value  // Embed dense<...> data
  );
}
```

**Generated MLIR**:
```mlir
llvm.mlir.global internal constant @constant_0(dense<[1.0, 2.0, ...]> : tensor<64x3x3x3xf32>)
  : !llvm.array<1728 x f32>

llvm.mlir.global internal constant @constant_1(dense<[0.5, ...]> : tensor<64xf32>)
  : !llvm.array<64 x f32>
```

### A.4: Initialization Function Generation

```mlir
// 1. Query constant count
llvm.func @get_constant_count() -> i64 {
  %count = llvm.mlir.constant(200 : i64) : i64
  llvm.return %count : i64
}

// 2. Upload all constants to GPU
func.func @initialize_constants(%ctx: !hip.context) -> i32 {
  // For each constant:
  %data_0 = llvm.mlir.addressof @constant_0 : !llvm.ptr
  %size_0 = llvm.mlir.constant(6912 : i64) : i64
  %index_0 = llvm.mlir.constant(0 : i64) : i64
  hip.upload_constant(%ctx, %index_0, %data_0, %size_0)

  // ... repeat for all constants ...

  %success = llvm.mlir.constant(0 : i32) : i32
  return %success : i32
}

// 3. Release all constants
func.func @release_constants(%ctx: !hip.context) -> i32 {
  %index_0 = llvm.mlir.constant(0 : i64) : i64
  hip.release_constant(%ctx, %index_0)

  // ... repeat for all constants ...

  %success = llvm.mlir.constant(0 : i32) : i32
  return %success : i32
}
```

### A.5: Runtime Interface

For state structure design and lifecycle, see [STATE-AND-CONTEXT.md](STATE-AND-CONTEXT.md).

```c
// Internal state structure (opaque to C interface)
struct HipExecutionState {
    hipStream_t stream;
    miopenHandle_t miopenHandle;
    hipblasLtHandle_t hipblasHandle;
    void** gpu_constants;  // Array of GPU pointers
};

// Generated functions (called by runtime)
extern "C" int64_t get_constant_count();
extern "C" int initialize_constants(void* state);
extern "C" int release_constants(void* state);

// Runtime implementation
int inference_init(void** out_state) {
    // Allocate state on heap
    HipExecutionState* state = new HipExecutionState();

    // Allocate constant pointer array
    state->gpu_constants = new void*[get_constant_count()];

    // Upload constants to GPU
    initialize_constants(state);

    // Return opaque pointer
    *out_state = state;
    return 0;
}

int inference_cleanup(void* state) {
    // Free GPU constant memory
    release_constants(state);

    // Free state structure
    delete static_cast<HipExecutionState*>(state);
    return 0;
}
```

---

## Appendix B: Implementation Phases

### Phase 1: Module-Level Pass Infrastructure (Week 1)
- Convert `ConvertOnnxToHipPass` to module-level
- Implement `isOnnxFunction()` helper
- Test with multi-function modules

### Phase 2: Constant Discovery (Week 1-2)
- Implement constant registry
- Walk all ONNX functions, discover constants
- Assign sequential indices
- Test registry correctness

### Phase 3: LLVM Global Generation (Week 2)
- Generate `llvm.mlir.global` for each constant
- Embed `dense<...>` data
- Verify in LLVM IR output

### Phase 4: Initialization Functions (Week 2-3)
- Generate `@get_constant_count()`
- Generate `@initialize_constants()` with hip.upload_constant
- Generate `@release_constants()` with hip.release_constant
- Test compilation and linking

### Phase 5: Constant Access in @main (Week 3)
- Replace `onnx.Constant` with `hip.get_constant`
- Test convolution with pre-uploaded weights
- Verify correctness

### Phase 6: Subgraph Handling (Week 4)
- Decide ctx-only vs ctx+constants approach
- Implement subgraph transformation
- Test with ONNX If/Loop/Scan models

### Phase 7: Integration & Validation (Week 4-5)
- Update runtime to call generated functions
- End-to-end test with ResNet50
- Performance benchmarking
- Accuracy validation

---

## Appendix C: Future Optimizations

1. **Constant Deduplication**: Share GPU memory for identical constants

---

## References

- MLIR Module-Level Passes: https://mlir.llvm.org/docs/PassManagement/
- LLVM GlobalOp: https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlir-global
- HIP Runtime API: https://rocm.docs.amd.com/projects/HIP/
- ONNX Runtime EPContext: https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html

---

**Document Status**: Design in Progress - Question 3 (subgraph handling) remains open
