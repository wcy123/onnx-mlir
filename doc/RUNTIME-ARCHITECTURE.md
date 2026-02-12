<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Runtime Architecture

**Date**: 2026-02-12
**Status**: Design Document
**Related**: [ARCHITECTURE.md](ARCHITECTURE.md), [MLIR-COMPILATION-DESIGN.md](MLIR-COMPILATION-DESIGN.md), [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md)

---

## Overview

The Runtime is a static library providing runtime state management, constant (model weights) management, and wrapper functions for ROCm operations. It bridges the gap between MLIR-generated code and AMD GPU hardware.

**Key Design Principle**: The runtime state is **opaque** to all external code (including generated code). This enables evolution without breaking compatibility.

**Dual Perspective**: The same entity has two names depending on perspective:
- **External perspective** (C interface): Opaque `void*` called **"state"**
- **Internal perspective** (MLIR implementation): Concrete type `!hip.context` called **"context"**

This naming reflects the abstraction boundary between the generic interface and HIP-specific implementation.

---

## What is the Runtime?

### Core Functionality

The Runtime library (`lib/Runtime`) provides:

#### 1. Runtime State Management
- Opaque state pointer (`void*`) for external callers
- Lifecycle management: `inference_init`, `inference_cleanup`
- GPU resources persist across multiple inference calls
- See [Opaque RuntimeState Pattern](#opaque-runtimestate-pattern) for implementation details

#### 2. Constant Management
- `hipdnn_ep_upload_constant()` - Upload model weights to GPU memory
- `hipdnn_ep_get_constant()` - Retrieve GPU pointer for a constant by index
- `hipdnn_ep_release_constant()` - Free GPU memory for constants

#### 3. Operation Wrappers (Extensible)

- `wrap_wrap_miopenConvolutionForward()` - Full MIOpen convolution wrapper
  - Creates tensor/convolution descriptors
  - Finds optimal algorithm
  - Allocates workspace memory
  - Performs forward pass
- `wrap_hipblasLtGemm()` - Matrix multiplication wrapper
  - Creates matrix layout descriptors
  - Performs GEMM operation

**Currently supported:**
- MIOpen - convolution
- hipBLASLt - GEMM

**To add a new library:** Add its handle to RuntimeState, create wrapper functions, and add initialization/cleanup code. The opaque design ensures no impact on existing code.

#### 4. Memory Management Wrappers
- `wrap_hipMalloc()` - GPU memory allocation
- `wrap_hipFree()` - GPU memory deallocation
- `wrap_hipMemcpyH2D()` - Host-to-device async copy
- `wrap_hipMemcpyD2H()` - Device-to-host async copy
- `wrap_hipStreamSynchronize()` - Stream synchronization

---

## Why Do We Need a Runtime?

### Problems It Solves

1. **State Must Persist Across Inference Calls**
   - GPU handles (stream, MIOpen, hipBLAS) must outlive individual function calls
   - Constants uploaded once in `init`, reused across multiple `compute` calls
   - Cannot use stack allocation (destroyed on function return)
   - Heap allocation required, managed by runtime

2. **Generated Code Needs Clean Abstraction**
   - Generated LLVM IR shouldn't be coupled to internal struct layout
   - Adding new GPU library (rocFFT, rocRAND, etc.) shouldn't break compiled DLLs
   - Opaque interface allows runtime evolution without recompilation

3. **Flexibility to Evolve Without Breaking DLLs**
   - Can add fields to RuntimeState (e.g., rocFFT handle) without changing C interface
   - Can optimize internal layout (alignment, cache locality) independently
   - Each compiled model is self-contained and independent

### Why Not Alternatives?

#### Why Not Inline Everything?
- **Code bloat**: Descriptor creation code duplicated at every call site
- **Inflexibility**: Cannot change implementation without regenerating all DLLs
- **Optimization barrier**: Cannot share descriptors or caches across operations

#### Why Not Thread-Local Storage?

**Alternative considered**:
```c
__thread void* g_hip_context;

int inference_init() {
    g_hip_context = create_context();
    return 0;
}

int inference_compute(span_t* inputs, span_t* outputs) {
    // Implicitly uses g_hip_context
    return 0;
}
```

**Rejected because**:
- ❌ Hidden state makes code harder to reason about
- ❌ TLS overhead on every access
- ❌ Difficult to support multiple concurrent models
- ❌ Not compatible with MLIR function-passing style
- ❌ Harder to test (can't easily mock state)

**Explicit state passing is better**:
- ✅ Clear data flow (visible in function signatures)
- ✅ Zero overhead (direct pointer passing)
- ✅ Multiple models = multiple state pointers
- ✅ MLIR-friendly (SSA form, passes analyze data flow)
- ✅ Testable (inject mock state)

#### Why Not Global State?
- Multiple models cannot coexist (global state is singleton)
- Hidden dependencies make code brittle
- Thread safety issues with concurrent inference
- Testing becomes difficult (global mutable state)

---

## Design Decisions

### Opaque RuntimeState Pattern

**The Struct (INTERNAL ONLY - not exposed to ANYONE, even generated code)**:

```c
struct RuntimeState {
    // Field 0: GPU stream for asynchronous execution
    hipStream_t stream;

    // Field 1: MIOpen handle for DNN operations
    miopenHandle_t miopenHandle;

    // Field 2: hipBLASLt handle for matrix operations
    hipblasLtHandle_t hipblasHandle;

    // Field 3: Pre-uploaded constant pointers
    void** gpu_constants;  // Array of GPU pointers (size known at compile time)

    // SELF-CONTAINED DESIGN: This struct is private to the compiled DLL.
    // Can freely add/remove fields for any ROCm library without breaking anything:
    // - C interface only sees opaque void*
    // - Generated code only calls accessor functions
    // - No external code depends on struct layout
    // - Each compiled model is independent
    //
    // Examples of ROCm libraries that can be added as handles:
    //   rocBLAS       - basic linear algebra
    //   rocFFT        - Fast Fourier Transform
    //   rocRAND       - random number generation
    //   rocSPARSE     - sparse linear algebra
    //   rocSOLVER     - LAPACK functionality
    //   RCCL          - multi-GPU collective communication
    //
    // Other extensible fields:
    // - Descriptor cache
    // - Workspace memory
    // - Algorithm selection cache
};
```

**Size**: Known at compile time (depends on number of constants)

**Allocation**: Heap-allocated in `inference_init`, freed in `inference_cleanup`

### Why Opaque?

**Problem**: If generated code uses GEP (GetElementPtr) to access RuntimeState fields:
- Adding a new field (e.g., rocFFT handle) changes field offsets
- All previously compiled DLLs break (field 3 is now field 4)
- Cannot evolve runtime without recompiling all models

**Solution**: Opaque RuntimeState with accessor functions:
- Runtime owns struct layout (defined in runtime implementation, not generated code)
- Generated code calls `hipdnn_ep_get_constant(state, index)` - no field offsets
- Runtime can add fields without breaking generated code
- Clean separation: generated code doesn't know or care about internals

**Benefits**:
1. **ABI Stability**: Runtime can evolve without breaking compiled models
2. **Flexibility**: Add handles (rocFFT, rocRAND, etc.) without regenerating DLLs
3. **Optimization**: Runtime can optimize layout (alignment, cache locality) independently
4. **Simplicity**: Generated code is simpler (function calls vs complex GEP chains)
5. **Testability**: Can mock runtime functions for testing

**Trade-offs**:
- Function call overhead (negligible - optimized to single load instruction)
- Runtime must provide accessor implementations

### Access Pattern

**CORRECT - Use accessor functions**:
```mlir
// ✅ Getting GPU stream
%stream = llvm.call @hipdnn_ep_get_stream(%state) : (!llvm.ptr) -> !llvm.ptr

// ✅ Getting constants
%index_0 = llvm.mlir.constant(0 : i64) : i64
%weight_0_gpu = llvm.call @hipdnn_ep_get_constant(%state, %index_0) : (!llvm.ptr, i64) -> !llvm.ptr
```

**FORBIDDEN - Direct field access**:
```mlir
// ❌ Never use GEP on RuntimeState
%stream_ptr = llvm.getelementptr %state[0, 0] : (!llvm.ptr) -> !llvm.ptr
%stream = llvm.load %stream_ptr : !llvm.ptr

// ❌ Never access constants array directly
%gpu_constants_ptr = llvm.getelementptr %state[0, 3] : (!llvm.ptr) -> !llvm.ptr
%gpu_constants = llvm.load %gpu_constants_ptr : !llvm.ptr
```

### State vs Context Naming (Dual Perspective)

**External (C interface)**: Use "state"
```c
void* state;                              // ✅ Opaque state pointer
int inference_init(void** out_state);     // ✅ Create state
int inference_compute(void* state, ...);  // ✅ Use state
int inference_cleanup(void* state);       // ✅ Destroy state
```

**Why "state" and not "context"?**
- "State" is generic and neutral
- Avoids confusion with GPU-specific terms like "CUDA context" or "HIP context"
- Makes the interface usable across different backend implementations
- Clear ownership model: "state" implies lifecycle management (create, use, destroy)

**Internal (MLIR - HIP dialect)**: Use "context" with `!hip.context` type
```mlir
func.func @main(%ctx: !hip.context, ...) -> i32 {          // ✅ HIP context
  hip.conv(%ctx, ...) {...}                                 // ✅ Pass context
}
```

**Why `!hip.context` and not `!hip.state`?**
- Matches HIP/CUDA terminology conventions
- Clear that it's HIP-specific execution context
- Familiar to GPU programmers (analogous to `hipStream_t`, `cudaContext_t`)
- Type name documents the backend choice

**Internal (MLIR - LLVM dialect)**: Use "state" with `!llvm.ptr` (already lowered)
```mlir
func.func @inference_compute(%state: !llvm.ptr, ...) -> i32 {  // ✅ Lowered to opaque ptr
  %stream = llvm.call @hipdnn_ep_get_stream(%state) : ...        // ✅ Access via function
}
```

This naming reflects the abstraction boundary:
- High-level HIP dialect: backend-specific "context"
- Low-level LLVM dialect: opaque "state"
- C interface: opaque "state"

### Why Heap Allocation (Not Stack)?

**Decision**: State allocated with `malloc` (heap), not `alloca` (stack)

**Rationale**:
- Stack allocation destroyed when `inference_init` returns
- State must outlive function call
- Heap allocation persists until explicit `free` in `inference_cleanup`

**Implementation**:
```mlir
// ✅ CORRECT: Heap allocation
%state = llvm.call @malloc(%size) : (i64) -> !llvm.ptr

// ❌ WRONG: Stack allocation (would be destroyed on return)
%state = llvm.alloca %c1 x !llvm.struct<...> : (i64) -> !llvm.ptr
```

---

## Why Static Library?

### Decision Rationale

Static linking is the **industry standard** for ML inference frameworks:

**Similar frameworks use static linking**:
- **TensorRT** (NVIDIA): `libnvinfer_plugin.a` statically linked
- **TVM**: Runtime modules statically linked per model
- **IREE**: HAL runtime statically linked
- **XLA**: Runtime services statically linked

**Reasons**:
1. Deployment simplicity (1 DLL = 1 model)
2. No DLL versioning hell
3. Self-contained artifacts
4. Better for distribution/packaging

### Technical Trade-offs

#### Static Linking (Current) ✅

**Build Time**:
- Runtime compiled to `libHipDnnRuntime.a`
- Linker copies Runtime code INTO `model.dll`

**Deployment**:
- Single DLL per model
- Runtime code embedded in DLL

**Advantages**:
- ✅ Simple deployment (1 file per model)
- ✅ No version conflicts between models
- ✅ Each model can use different Runtime version
- ✅ Self-contained DLLs

**Disadvantages**:
- Code duplication (each DLL contains Runtime code)
- Slightly larger DLLs

#### Dynamic Linking (Alternative) ❌

**Build Time**:
- Runtime compiled to `hip_ep_runtime.dll`
- Model DLL links to runtime DLL at runtime

**Deployment**:
- Model DLL + runtime DLL (2 files)

**Advantages**:
- Shared Runtime code (smaller total size)
- Can update Runtime without recompiling models

**Disadvantages**:
- ❌ Must deploy 2 files per model
- ❌ Version conflicts if multiple models need different Runtime versions
- ❌ DLL dependency management complexity
- ❌ Runtime DLL must be in PATH or same directory

**Conclusion**: Static linking simplicity outweighs size cost.

---

## Optimization Opportunities

The runtime can be made smaller by generating operations directly in LLVM IR:

#### Keep in Runtime - **REQUIRED**
1. **Runtime State Management**
   - RuntimeState struct, handle lifecycle
   - State must persist between inference calls

2. **Constant Management**
   - upload/get/release functions
   - GPU pointer map must persist across calls

#### Generate Directly Instead
1. **Operation Wrappers**
   - Generate direct calls to `wrap_miopenConvolutionForward`
   - Generate direct calls to `hipblasLtMatmul`
   - Inline descriptor creation in generated IR

2. **Memory Wrappers**
   - Generate direct calls to `hipMalloc`/`hipFree`
   - Generate direct calls to `hipMemcpyAsync`
   - Generate direct calls to `hipStreamSynchronize`

### Benefits of Optimization
- Smaller static library
- Model-specific optimizations possible
- Direct ROCm API calls (no wrapper overhead)
- More flexible code generation

### Trade-off
Runtime size vs generated code size - generating wrappers inline trades smaller runtime for slightly larger generated code per model.

---

## Future Extensions

### Adding GPU Libraries

**Example**: Add rocFFT for FFT operations

**Required changes**:
```c
struct RuntimeState {
    // ... existing fields ...
    rocfft_plan fft_plan;  // NEW field
};
```

**No changes needed**:
- ✅ C interface unchanged (`void* state`)
- ✅ CustomOp code unchanged (opaque pointer)
- ✅ Existing generated code unchanged (accessor functions)

**Possible additions**:
- **rocBLAS** - basic linear algebra
- **rocFFT** - Fast Fourier Transform
- **rocRAND** - random number generation
- **rocSPARSE** - sparse linear algebra
- **rocSOLVER** - LAPACK functionality
- **RCCL** - multi-GPU communication

Each addition is isolated to the runtime implementation. The opaque design ensures zero impact on existing code.

### Multiple Backend Support

**Vision**: Same C interface, different implementations

**HIP implementation**:
```c
struct HipExecutionState {
    hipStream_t stream;
    miopenHandle_t miopenHandle;
    hipblasLtHandle_t hipblasHandle;
    // ...
};
```

**CUDA implementation**:
```c
struct CudaExecutionState {
    cudaStream_t stream;
    cudnnHandle_t cudnnHandle;
    cublasLtHandle_t cublasHandle;
    // ...
};
```

**Both compile to same C interface**:
```c
int inference_init(void** out_state);  // Creates HipExecutionState OR CudaExecutionState
int inference_compute(void* state, ...);
int inference_cleanup(void* state);
```

**CustomOp unchanged** - works with both backends. The opaque `void*` state abstracts the backend completely.

---

## Dependencies

### Runtime Dependencies (External DLLs)

These are **not** part of your DLL - they must be installed on the target system via ROCm:

- **HIP Runtime** (`amdhip64.dll`) - GPU compute abstraction
- **MIOpen** (`MIOpen.dll`) - Convolution and DNN operations
- **hipBLASLt** (`hipblaslt.dll`) - Matrix multiplication

### Deployment Example
```
User's System:
├── C:\Program Files\AMD\ROCm\bin\
│   ├── amdhip64.dll (from ROCm install)
│   ├── MIOpen.dll (from ROCm install)
│   └── hipblaslt.dll (from ROCm install)
│
└── C:\MyApp\
    └── resnet50.dll (your generated model)
        ↳ Calls functions in ROCm DLLs
```

Your generated DLL is self-contained except for these ROCm runtime dependencies.

---

## Lifecycle

### 1. Creation (inference_init)

**C interface**:
```c
void* state;
int ret = inference_init(&state);  // Creates and initializes state
```

**What happens**:
1. Allocate RuntimeState struct on heap
2. Create GPU handles (`hipStreamCreate`, `miopenCreate`, `hipblasLtCreate`)
3. Upload constants to GPU (via `hipdnn_ep_upload_constant`)
4. Store everything in state struct
5. Return state pointer

### 2. Usage (inference_compute)

**C interface**:
```c
int ret = inference_compute(state, inputs, outputs);  // Uses pre-initialized state
```

**What happens**:
1. Get stream via `hipdnn_ep_get_stream` (opaque - no GEP)
2. Get pre-uploaded constants via `hipdnn_ep_get_constant` (opaque - no GEP)
3. Execute operations using stream and weights
4. Return results

**Key point**: State is **opaque** - accessed only via runtime functions, never via GEP.

### 3. Destruction (inference_cleanup)

**C interface**:
```c
int ret = inference_cleanup(state);  // Frees all resources
```

**What happens**:
1. Free GPU constant memory (via `hipdnn_ep_release_constant`)
2. Destroy GPU handles (`miopenDestroy`, `hipStreamDestroy`, etc.)
3. Free state struct itself
4. State pointer becomes invalid

---

## Relationship to Constants

The state structure contains pointers to pre-uploaded constants (weights, biases).

**See [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md) for detailed design.**

**Summary**:
- Constants embedded in DLL as `llvm.mlir.global`
- Uploaded to GPU in `inference_init`
- GPU pointers stored in `state->gpu_constants[]`
- Accessed in `inference_compute` via `hipdnn_ep_get_constant`
- Freed in `inference_cleanup` via `hipdnn_ep_release_constant`

**Example**:
```mlir
// In initialize_constants (called from inference_init):
%weight_cpu = llvm.mlir.addressof @constant_0 : !llvm.ptr
%size = llvm.mlir.constant(6912 : i64) : i64
%index_0 = llvm.mlir.constant(0 : i64) : i64
// Upload via runtime function (opaque - no GEP)
llvm.call @hipdnn_ep_upload_constant(%state, %index_0, %weight_cpu, %size)
  : (!llvm.ptr, i64, !llvm.ptr, i64) -> i32

// In inference_compute:
%index_0 = llvm.mlir.constant(0 : i64) : i64
// Get via runtime function (opaque - no GEP)
%weight_gpu = llvm.call @hipdnn_ep_get_constant(%state, %index_0)
  : (!llvm.ptr, i64) -> !llvm.ptr
llvm.call @wrap_miopenConvolutionForward(..., %weight_gpu, ...)
```

---

## References

- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system architecture
- [MLIR-COMPILATION-DESIGN.md](MLIR-COMPILATION-DESIGN.md) - MLIR compilation pipeline
- [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md) - Constants design
- HIP Stream Management: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#stream-management

---

**Document Status**: Complete - consolidates STATE-AND-CONTEXT.md and RUNTIME-LIBRARY-ARCHITECTURE.md
