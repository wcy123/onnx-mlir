# Constant Handling Design
## ONNX Initializers in MLIR-based Compilation Pipeline

**Status**: Design in Progress
**Date**: 2026-02-10
**Related**: ARCHITECTURE.md, MLIR-COMPILATION-DESIGN.md

---

## Executive Summary

This document defines the architecture for handling ONNX model constants (weights, biases, embeddings) in the MLIR-based compilation pipeline for the HipDNN Execution Provider. The design eliminates the naive approach of passing 200+ constant arguments through function signatures by introducing a state-based constant management system with pre-compiled initialization code.

**Key innovation**: Constants are embedded in the compiled DLL as LLVM globals, uploaded to GPU once during initialization, and accessed through a state structure—achieving clean function signatures while maintaining optimal performance.

---

## Problem Statement

Deep learning models (e.g., ResNet50) contain hundreds of constant tensors:
- 100+ convolutional layers, each with weights and biases
- Batch normalization layers with scale/bias/mean/variance tensors
- Embedding layers with large lookup tables

In ONNX-MLIR, these appear as `onnx.Constant` operations within function bodies.

**Challenge**: How do we transform these constants during ONNX→HIP conversion without creating unmaintainable function signatures?

**Naive approach failure**:
```mlir
// Treating constants as function arguments creates:
func.func @main(%ctx: !hip.context, %input: memref<...>,
                %w0: memref<...>, %b0: memref<...>, %w1: memref<...>,
                // ... 196 more constant parameters ...
                %output: memref<...>) -> i32
```

Problems:
- ❌ 200+ function arguments (unmaintainable)
- ❌ Calling convention overhead
- ❌ Doesn't match execution model (constants loaded once, used repeatedly)
- ❌ Doesn't scale to larger models

---

## Core Design Decisions

### Decision 1: State-Based Constant Management

**Choice**: Store constants in a state structure with pre-uploaded GPU pointers.

**Rationale**:
- Constants have different lifecycle than inputs/outputs (loaded once vs per-inference)
- GPU memory allocation/upload is expensive—do it once, reuse many times
- Matches industry standard execution provider patterns (TensorRT, QNN, VitisAI)
- Clean separation of initialization vs execution concerns

**Architecture**:
```c
struct State {
    hipStream_t stream;
    miopenHandle_t miopenHandle;
    void** gpu_weights;  // Array of pre-uploaded constant pointers
};
```

Functions receive `%ctx: !hip.context` instead of individual constants.

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
- Direct array indexing in `state->gpu_weights[]`
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

### Decision 7: ONNX Function Identification

**Choice**: Process only functions with tensor types + ONNX dialect operations.

**Rationale**:
- Pass can coexist with other MLIR passes in pipeline
- Order-independent in pass manager
- Won't accidentally transform non-ONNX helper functions
- Robust to future additions of utility functions

**Identification criteria**:
1. Function signature contains `TensorType`
2. Function body contains ONNX dialect operations

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
  - Inference function using gpu_weights[]
```

### Runtime Initialization

```c
// 1. Runtime loads DLL, resolves symbols
get_constant_count_fn = dlsym(dll, "get_constant_count");
initialize_constants_fn = dlsym(dll, "initialize_constants");

// 2. Create state structure
State* state = new State();
state->gpu_weights = new void*[get_constant_count()];

// 3. Upload all constants to GPU (once)
initialize_constants(state);  // Generated code: hipMalloc + hipMemcpy for each

// 4. State is ready for inference
// @main(%ctx, input, output) uses gpu_weights[] internally
```

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

**Context**: ONNX control flow operators (If, Loop, Scan) create subgraphs as separate `func.func`.

**Example**:
```mlir
func.func @main_graph(%input: tensor<...>) -> tensor<...> {
  %w0 = "onnx.Constant"() {value = dense<...>}
  %result = func.call @subgraph_if_then(%input, %w0)
  ...
}

func.func @subgraph_if_then(%arg0: tensor<...>, %arg1: tensor<...>) -> tensor<...> {
  %w1 = "onnx.Constant"() {value = dense<...>}  // Local constant
  %conv = "onnx.Conv"(%arg0, %arg1, %w1)
  ...
}
```

**Question**: After conversion, should `@subgraph_if_then`:
- **Option A**: Receive only `%ctx`, load all constants (including passed ones) from state?
- **Option B**: Receive `%ctx` + pre-loaded constants as arguments?

**Trade-offs**:
- Option A: Clean signatures, but caller and callee must agree on global indices
- Option B: Reintroduces constant arguments (defeats purpose of design)

**TODO**: Decide based on real-world ONNX model analysis.

---

## Appendix A: Alternatives Considered

### Alternative 1: External Constant File

**Approach**: Store constants in separate binary file, load at runtime.

**Rejected because**:
- ❌ Requires disk I/O at inference time
- ❌ Two artifacts to manage (DLL + data file)
- ❌ Versioning/compatibility issues (DLL vs data mismatch)
- ❌ More complex deployment

### Alternative 2: JIT Compilation with LLVM IR

**Approach**: Store LLVM IR in EPContext, JIT compile at runtime to access constants.

**Rejected because**:
- ❌ 100-500ms JIT overhead unacceptable for inference
- ❌ LLVM runtime dependency (50-200 MB)
- ❌ Defeats purpose of EPContext (eliminate recompilation)

### Alternative 3: Function-Level Pass

**Approach**: Keep `ConvertOnnxToHipPass` as `OperationPass<func::FuncOp>`.

**Rejected because**:
- ❌ Cannot create module-level `llvm.mlir.global` operations
- ❌ Cannot discover constants across multiple functions
- ❌ Cannot generate module-level initialization functions
- ❌ No shared constant registry across functions

### Alternative 4: Constant Arguments with Bundling

**Approach**: Bundle constants into a single struct argument.

**Rejected because**:
- ❌ Still requires passing data through call chain
- ❌ Type-unsafe (void* or unions)
- ❌ Runtime packing/unpacking overhead
- ❌ Doesn't reflect actual execution model

### Alternative 5: hash-Based Constant Indexing

**Approach**: Use hash of constant data as index instead of sequential numbers.

**Rejected because**:
- ❌ Hash collisions require resolution logic
- ❌ Non-deterministic indices complicate debugging
- ❌ Lookup overhead vs direct array indexing
- ✅ Could enable deduplication (future optimization)

---

## Appendix B: Implementation Details

### B.1: ONNX Function Identification

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

### B.2: Constant Discovery

```cpp
struct ConstantInfo {
  size_t globalIndex;      // Index in state->gpu_weights[]
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

### B.3: LLVM Global Generation

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

### B.4: Initialization Function Generation

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

### B.5: Runtime Interface

```c
struct State {
    hipStream_t stream;
    miopenHandle_t miopenHandle;
    hipblasLtHandle_t hipblasHandle;
    void** gpu_weights;  // Array of GPU pointers
};

extern "C" int64_t get_constant_count();
extern "C" int initialize_constants(void* state_ptr);
extern "C" int release_constants(void* state_ptr);

int inference_init(void** state_ptr) {
    State* state = new State();
    state->gpu_weights = new void*[get_constant_count()];
    initialize_constants(state);  // Upload to GPU
    *state_ptr = state;
    return 0;
}

int inference_release(void* state_ptr) {
    release_constants(state_ptr);
    delete state;
    return 0;
}
```

---

## Appendix C: Implementation Phases

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

## Appendix D: Future Optimizations

1. **Constant Deduplication**: Share GPU memory for identical constants (requires hash-based registry)
2. **Lazy Upload**: Only upload constants actually used (requires liveness analysis)
3. **Compression**: Compress data in DLL, decompress during upload (trade CPU for size)
4. **Quantization**: INT8/INT4 constant support with dequantization kernels
5. **Batched Upload**: Single `hipMemcpy` for all constants (requires memory layout planning)
6. **Pinned Memory**: Use `hipHostMalloc` for faster transfers
7. **Async Upload**: Overlap upload with other initialization (requires stream management)

---

## References

- MLIR Module-Level Passes: https://mlir.llvm.org/docs/PassManagement/
- LLVM GlobalOp: https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlir-global
- HIP Runtime API: https://rocm.docs.amd.com/projects/HIP/
- ONNX Runtime EPContext: https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html

---

**Document Status**: Design in Progress - Question 3 (subgraph handling) remains open
