<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Runtime Architecture

**Date**: 2026-02-12
**Status**: Design Document
**Related**: [ARCHITECTURE.md](ARCHITECTURE.md), [MLIR-COMPILATION-DESIGN.md](MLIR-COMPILATION-DESIGN.md), [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md)

---

## Runtime Integration Pipeline

This diagram shows how the Runtime integrates into the compilation flow (complementary to [ARCHITECTURE.md System Architecture](ARCHITECTURE.md#system-architecture)):

```
┌──────────────────────────────────────────────────────────────────┐
│  BUILD TIME (Once - when building EP DLL)                        │
│  Produces: Embedded Runtime bitcode                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  hipdnn_ep_runtime.cpp + hipdnn_ep_runtime.h                     │
│         ↓                                                         │
│  clang -c -emit-llvm -O2 -std=c++17                              │
│         ↓                                                         │
│  runtime.bc (LLVM bitcode)                                       │
│         ↓                                                         │
│  xxd.py --var runtime_bc_data                                    │
│         ↓                                                         │
│  runtime_ir_data.cpp (embedded as unsigned char array)           │
│         ↓                                                         │
│  Compiled into EP DLL (libHipDnnEpRuntime.a linked)              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│  MODEL COMPILATION TIME (Per model - Level-1 Pass)               │
│  Input: ONNX model    Output: EPContext with embedded DLL        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [From ARCHITECTURE.md: ONNX → MLIR transformations]             │
│         ↓                                                         │
│  MLIR (HIP dialect) → LLVM dialect conversion                    │
│         ↓                                                         │
│  ┌────────────────────────────────────────────────────┐          │
│  │ LLVM IR Generation                                 │          │
│  │  - @main(state, inputs, outputs) function          │          │
│  │  - @inference_init/compute/cleanup wrappers        │          │
│  │  - Calls to @hipdnn_ep_get_stream, etc.            │          │
│  │  - llvm.mlir.global for embedded constants         │          │
│  │  - @initialize_constants helper                    │          │
│  └────────────────────────────────────────────────────┘          │
│         ↓                                                         │
│  ┌────────────────────────────────────────────────────┐          │
│  │ Runtime IR Merging (llvm::Linker API)              │          │
│  │  1. Parse embedded runtime_bc_data → Runtime IR    │          │
│  │  2. llvm::Linker::linkInModule(Runtime IR)         │          │
│  │  3. Resolve function declarations:                 │          │
│  │     - @hipdnn_ep_state_init(ptr, i64) -> i32       │          │
│  │     - @hipdnn_ep_get_stream(ptr) -> ptr            │          │
│  │     - @hipdnn_ep_get_constant(ptr, i64) -> ptr     │          │
│  │     - @wrap_miopenConvolutionForward(...)          │          │
│  │     - @wrap_hipblasLtGemm(...)                     │          │
│  │     - @wrap_hipMalloc/Free/MemcpyH2D/D2H/Sync     │          │
│  └────────────────────────────────────────────────────┘          │
│         ↓                                                         │
│  Combined LLVM IR Module                                         │
│    (Generated code + Runtime implementation merged)              │
│         ↓                                                         │
│  ┌────────────────────────────────────────────────────┐          │
│  │ LLVM Optimization (PassBuilder O2)                 │          │
│  │  Key transformations:                              │          │
│  │  - Inline @hipdnn_ep_get_stream() → load instr     │          │
│  │  - Inline @hipdnn_ep_get_constant() → array access │          │
│  │  - Dead code elimination (unused Runtime code)     │          │
│  │  - Cross-module inlining opportunities             │          │
│  └────────────────────────────────────────────────────┘          │
│         ↓                                                         │
│  Optimized LLVM IR (accessor calls eliminated)                   │
│         ↓                                                         │
│  ┌────────────────────────────────────────────────────┐          │
│  │ Native Code Generation                             │          │
│  │  - LLVM IR → Object code (.obj/.o)                 │          │
│  │  - Link with libHipDnnEpRuntime.a (static)          │          │
│  │  - Link ROCm libraries:                            │          │
│  │    * amdhip64.lib (HIP runtime)                    │          │
│  │    * MIOpen.lib (DNN operations)                   │          │
│  │    * hipblaslt.lib (BLAS operations)               │          │
│  │  - Produce: model.dll                              │          │
│  └────────────────────────────────────────────────────┘          │
│         ↓                                                         │
│  model.dll (native x64 DLL)                                      │
│    Exports:                                                       │
│      - inference_init(void** out_state) -> i32                   │
│      - inference_compute(void* state, span_t*, span_t*) -> i32   │
│      - inference_cleanup(void* state) -> i32                     │
│    Contains:                                                      │
│      - Inlined Runtime code (no function call overhead)          │
│      - Embedded constants (weights/biases in .data section)      │
│      - GPU operation calls (MIOpen, hipBLAS)                     │
│         ↓                                                         │
│  ┌────────────────────────────────────────────────────┐          │
│  │ EPContext Packaging                                │          │
│  │  1. Read model.dll into memory buffer              │          │
│  │  2. Create EPContext node in ONNX graph            │          │
│  │  3. Embed DLL as binary attribute                  │          │
│  │  4. Replace original graph with EPContext node     │          │
│  │  5. Save as model_with_context.onnx                │          │
│  └────────────────────────────────────────────────────┘          │
│         ↓                                                         │
│  model_with_context.onnx (single file deployment)                │
│    Contains:                                                      │
│      - EPContext node (com.microsoft:EPContext)                  │
│      - Embedded model.dll (pre-compiled, optimized)              │
│      - No ONNX graph (replaced by compiled artifact)             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│  INFERENCE TIME (Runtime - repeatedly executed)                  │
│  Dependencies: HIP runtime, MIOpen, hipBLASLt                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Load model_with_context.onnx                                    │
│         ↓                                                         │
│  CustomOp recognizes EPContext node                              │
│         ↓                                                         │
│  Extract embedded model.dll from EPContext attribute             │
│         ↓                                                         │
│  Load DLL from memory (MemoryModule or platform API)             │
│         ↓                                                         │
│  Resolve function pointers:                                      │
│    - fn_init = GetProcAddress("inference_init")                  │
│    - fn_compute = GetProcAddress("inference_compute")            │
│    - fn_cleanup = GetProcAddress("inference_cleanup")            │
│         ↓                                                         │
│  Session Initialization:                                         │
│    void* state = nullptr;                                        │
│    fn_init(&state);  // Creates GPU handles, allocates constants │
│         ↓                                                         │
│  Inference Execution (repeated):                                 │
│    fn_compute(state, inputs, outputs);  // Reuses GPU resources  │
│         ↓                                                         │
│  Session Cleanup:                                                │
│    fn_cleanup(state);  // Frees GPU resources                    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Key Integration Points:**
1. **Build Time**: Runtime compiled to bitcode once, embedded in EP DLL
2. **Model Compilation**: Runtime IR merged with each model's generated IR
3. **Optimization**: LLVM inlines Runtime accessor functions (zero-cost abstraction)
4. **Packaging**: Final DLL embedded in EPContext for single-file deployment
5. **Inference**: CustomOp loads DLL from memory, calls exported functions

---

## Overview

### What is the Runtime?

The Runtime is a static library that manages GPU execution state for compiled ONNX models.

**Core Problem**: MLIR-generated code needs to:
- Manage persistent GPU resources (stream, library handles)
- Access pre-uploaded model weights (constants)
- Call GPU operations (convolution, GEMM) without coupling to implementation details

**Solution**: Opaque RuntimeState with accessor functions
- External code sees: `void* state` (opaque pointer)
- Internal code owns: GPU handles, constant mappings, resources
- Clean abstraction: Runtime can evolve without breaking generated code

### Three-Function Lifecycle

**`inference_init(void** out_state)`** - Create GPU resources once
- Allocates RuntimeState
- Creates HIP stream, MIOpen handle, hipBLAS handle
- Returns opaque state pointer

**`inference_compute(void* state, span_t* inputs, span_t* outputs)`** - Execute inference (reuses resources)
- Parses input/output tensors
- Allocates GPU buffers
- Copies data H2D, calls @main, copies results D2H
- Synchronizes and frees buffers

**`inference_cleanup(void* state)`** - Free all resources
- Synchronizes GPU operations
- Destroys library handles in LIFO order
- Frees RuntimeState memory

---

## Architecture

### Opaque Handle Design

The abstraction boundary separates generated code from runtime internals:

```
┌─────────────────────────────────────────────────────────┐
│  Generated Code (LLVM IR)                                │
│  - Sees: void* state (opaque pointer)                   │
│  - Calls: hipdnn_ep_get_stream(state)                   │
│  - Calls: hipdnn_ep_get_constant(state, index)          │
│  - Cannot: Access internal fields directly              │
└─────────────────────────────────────────────────────────┘
                         ↕ (Abstraction Boundary)
┌─────────────────────────────────────────────────────────┐
│  Runtime Library (C++ Implementation)                    │
│  - Owns: struct RuntimeState {                          │
│           hipStream_t stream;                            │
│           miopenHandle_t miopen_handle;                  │
│           hipblasLtHandle_t hipblas_handle;              │
│           void** gpu_constants;                          │
│           size_t num_constants;                          │
│         }                                                │
│  - Provides: Accessor functions                         │
│  - Can: Evolve internal layout freely                   │
└─────────────────────────────────────────────────────────┘
```

### Why Opaque?

Generated code uses accessor functions instead of direct field access (GEP):

**✅ Correct - Opaque access:**
```mlir
%stream = llvm.call @hipdnn_ep_get_stream(%state) : (!llvm.ptr) -> !llvm.ptr
%constant = llvm.call @hipdnn_ep_get_constant(%state, %idx) : (!llvm.ptr, i64) -> !llvm.ptr
```

**❌ Forbidden - Direct access:**
```mlir
%stream_ptr = llvm.getelementptr %state[0, 0] : (!llvm.ptr) -> !llvm.ptr
```

**Benefits:**
- Add GPU libraries (rocFFT, rocRAND) without breaking generated code
- Optimize internal layout independently
- Each compiled model is self-contained

**Implementation:** Real accessor functions are inlined to zero cost via LLVM IR merging (see Compilation Pipeline section).

---

## Compilation Pipeline: LLVM IR Merging

This section details how the Runtime library integrates with compiled models. For the overall system architecture, see [ARCHITECTURE.md Design Decision #7](ARCHITECTURE.md#7-llvm-ir-merging-for-zero-cost-runtime-abstraction).

### Build-Time Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ BUILD TIME (Once - when building EP DLL)                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  hipdnn_ep_runtime.cpp                                  │
│         ↓                                                │
│  clang -c -emit-llvm -O2                                │
│         ↓                                                │
│  runtime.bc (LLVM bitcode)                              │
│         ↓                                                │
│  xxd.py --var runtime_bc_data                           │
│         ↓                                                │
│  runtime_ir_data.cpp (embedded C array)                 │
│         ↓                                                │
│  Compiled into libHipDnnEpRuntime.a                     │
│  (EP DLL contains embedded bitcode)                     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ MODEL COMPILATION (Per model)                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  MLIR → LLVM IR (generated code)                        │
│         ↓                                                │
│  llvm::Linker::linkInModule()                           │
│  (Merge runtime.bc with generated IR)                   │
│         ↓                                                │
│  Combined LLVM IR module                                │
│         ↓                                                │
│  LLVM PassBuilder (O2 optimization)                     │
│  - Inline hipdnn_ep_get_stream() → single load          │
│  - Inline hipdnn_ep_get_constant() → single load        │
│         ↓                                                │
│  Optimized native code (model.dll)                      │
│  (Accessor functions disappeared - inlined away)        │
└─────────────────────────────────────────────────────────┘
```

### Zero-Cost Abstraction

Runtime accessor functions have **zero overhead** in final binary:

**Before optimization (after linking):**
```llvm
%stream = call ptr @hipdnn_ep_get_stream(ptr %state)
```

**After LLVM O2 inlining:**
```llvm
%stream_ptr = getelementptr inbounds %struct.RuntimeState, ptr %state, i32 0, i32 0
%stream = load ptr, ptr %stream_ptr
```

Result: Clean abstraction in source code, efficient code in binary.

### Build Configuration

```cmake
# CMakeLists.txt (lib/Runtime)
if(ENABLE_RUNTIME_IR_MERGING)
    add_custom_command(
        OUTPUT runtime.bc
        COMMAND ${CMAKE_CXX_COMPILER} -c -emit-llvm -O2
                hipdnn_ep_runtime.cpp -o runtime.bc
    )
    add_custom_command(
        OUTPUT runtime_ir_data.cpp
        COMMAND ${Python3_EXECUTABLE} xxd.py
                --var runtime_bc_data --output runtime_ir_data.cpp runtime.bc
    )
else()
    # Fallback: stub if Clang not available
    file(WRITE runtime_ir_data_stub.cpp ...)
endif()
```

**Requirements:** Clang compiler, Python 3, LLVM Linker API

---

## Implementation Reference

### RuntimeState Internal Structure

**Actual Implementation** (lib/Runtime/hipdnn_ep_runtime.cpp):
```cpp
struct RuntimeState {
  hipStream_t stream;                    // GPU stream
  miopenHandle_t miopen_handle;          // MIOpen for convolution
  hipblasLtHandle_t hipblas_handle;      // hipBLAS for GEMM
  void** gpu_constants;                  // Array of GPU pointers
  size_t num_constants;                  // Array size (known at compile time)
};
```

**Key Properties:**
- Size: 4 pointers + array pointer + size (~40 bytes + constant array)
- Allocation: Heap (`malloc` in init, `free` in cleanup)
- Thread safety: **NOT thread-safe** - one inference per state at a time
- Extensibility: Can add fields (rocFFT, rocRAND, etc.) without breaking generated code

### Error Codes

**hipdnn_ep_state_init():**
- `0` = success
- `1` = allocation failed
- `2` = stream creation failed
- `3` = MIOpen creation failed
- `4` = set stream failed
- `5` = hipBLAS creation failed

**Other functions:**
- `0` = success
- Negative = runtime error
- `hipdnn_ep_state_cleanup()` always returns `0` (best-effort)

### Core Functions

#### State Management
- **`hipdnn_ep_state_init(RuntimeState** out_state, size_t num_constants)`**
  - Creates stream, MIOpen handle, hipBLAS handle
  - Allocates constant array with given size
  - Returns error code (0 = success)

- **`hipdnn_ep_state_cleanup(RuntimeState* state)`**
  - Frees all constants
  - Destroys handles in LIFO order
  - Always returns 0 (best-effort)

- **`hipdnn_ep_get_stream(RuntimeState* state)`**
  - Returns stream as void*
  - Used by generated code to pass stream to operations

#### Constant Management
- **`hipdnn_ep_upload_constant(RuntimeState* state, int64_t index, const void* data, int64_t size)`**
  - Validates index range [0, num_constants)
  - Allocates GPU memory via hipMalloc
  - Copies data via hipMemcpyAsync
  - Stores GPU pointer in array

- **`hipdnn_ep_get_constant(RuntimeState* state, int64_t index)`**
  - Validates index range
  - Returns GPU pointer from array
  - NULL if index out of range

- **`hipdnn_ep_release_constant(RuntimeState* state, int64_t index)`**
  - Validates index range
  - Frees GPU memory via hipFree
  - Clears array entry

#### Operation Wrappers
- **`wrap_miopenConvolutionForward(...)`**
  - Full convolution (creates descriptors, finds algorithm, allocates workspace)
  - Currently hardcoded to float32 (miopenFloat)

- **`wrap_hipblasLtGemm(...)`**
  - Matrix multiplication (creates layout descriptors)
  - Currently hardcoded to float32 (HIPBLAS_R_32F)
  - Assumes column-major layout

**Current Limitations:**
- No descriptor caching (recreated per operation)
- Fixed to float32 data type
- GEMM assumes column-major layout

#### Memory Wrappers
- **`wrap_hipMalloc(void** ptr, int64_t size)`** - GPU allocation
- **`wrap_hipFree(void* ptr)`** - GPU deallocation
- **`wrap_hipMemcpyH2D(void* dst, const void* src, int64_t size, void* stream)`** - Async copy
- **`wrap_hipMemcpyD2H(void* dst, const void* src, int64_t size, void* stream)`** - Async copy
- **`wrap_hipStreamSynchronize(void* stream)`** - Synchronization

### Mock Runtime

**Purpose:** Enable testing without ROCm installation

**Configuration:** `cmake -DBUILD_MOCK_RUNTIME=ON`

**Behavior:**
- Prints diagnostic messages to stdout
- Uses malloc/free instead of GPU operations
- All functions return success codes

**Example:** See test/runtime/test_runtime_state.cpp

---

## Usage Examples

### API Usage (from test/runtime/test_runtime_state.cpp)

```cpp
// Initialize runtime state with 0 constants
RuntimeState *state = nullptr;
int result = hipdnn_ep_state_init(&state, 0);
assert(result == 0 && state != nullptr);

// Use state for inference
// ... (call inference_compute, operations, etc.)

// Cleanup
result = hipdnn_ep_state_cleanup(state);
assert(result == 0);
```

### MLIR Lowering Patterns (from lib/HipDialect/HipToLLVM.cpp)

**High-level HIP dialect:**
```mlir
hip.conv(%ctx, %input, %weights, %output) { ... }
```

**Lowered to LLVM dialect with runtime calls:**
```mlir
llvm.call @wrap_miopenConvolutionForward(
  %handle, %stream, %input_ptr, %input_shape,
  %weights_ptr, %weights_shape, %output_ptr, %output_shape,
  %pad_h, %pad_w, %stride_h, %stride_w, %dilation_h, %dilation_w
) : (...) -> i32
```

### Generated Interface (from lib/HipDialect/GenerateInterfacePass.cpp)

```mlir
llvm.func @inference_init(%out_state: !llvm.ptr) -> i32 {
  // Count constants at compile time
  %num_constants = llvm.call @get_constant_count() : () -> i64

  // Initialize runtime state
  %result = llvm.call @hipdnn_ep_state_init(%out_state, %num_constants)
    : (!llvm.ptr, i64) -> i32
  llvm.return %result : i32
}

llvm.func @inference_compute(%state: !llvm.ptr, %inputs: !llvm.ptr, %outputs: !llvm.ptr) -> i32 {
  %stream = llvm.call @hipdnn_ep_get_stream(%state) : (!llvm.ptr) -> !llvm.ptr
  // ... allocate GPU buffers, copy data, call @main, copy results ...
  llvm.call @wrap_hipStreamSynchronize(%stream) : (!llvm.ptr) -> i32
  llvm.return %c0_i32 : i32
}

llvm.func @inference_cleanup(%state: !llvm.ptr) -> i32 {
  %result = llvm.call @hipdnn_ep_state_cleanup(%state) : (!llvm.ptr) -> i32
  llvm.return %result : i32
}
```

---

## Extensibility

The opaque RuntimeState design enables adding GPU libraries without breaking the interface.

### Adding New Libraries

**Example:** Add rocFFT for FFT operations

**Required changes:**
```cpp
struct RuntimeState {
    // ... existing fields ...
    rocfft_plan fft_plan;  // NEW field
};
```

**No changes needed:**
- ✅ C interface unchanged (`void* state`)
- ✅ Generated code unchanged (opaque pointer)
- ✅ Existing compiled models unchanged (accessor functions)

### Extensible Libraries

The RuntimeState can be extended with handles for:
- **rocBLAS** - Basic linear algebra
- **rocFFT** - Fast Fourier Transform
- **rocRAND** - Random number generation
- **rocSPARSE** - Sparse linear algebra
- **rocSOLVER** - LAPACK functionality
- **RCCL** - Multi-GPU collective communication

### Other Extensible Fields

Beyond library handles, RuntimeState can include:
- Descriptor caches (avoid recreation overhead)
- Workspace memory pools (reduce allocation)
- Algorithm selection caches (avoid re-finding optimal algorithms)

Each addition is isolated to the runtime implementation. The opaque design ensures zero impact on existing code.

---

## Dependencies & Deployment

### Why Static Library?

The Runtime is compiled as `libHipDnnEpRuntime.a` and statically linked into each model DLL.

**Rationale:**
- Industry standard (TensorRT, TVM, IREE, XLA all use static linking)
- Simple deployment: 1 DLL = 1 model (self-contained)
- No DLL versioning conflicts between models

### Runtime Dependencies (External)

These must be installed on target system via ROCm:
- **amdhip64.dll** - HIP runtime
- **MIOpen.dll** - Convolution and DNN operations
- **hipblaslt.dll** - Matrix multiplication
