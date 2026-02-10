# State and Context Design
## Opaque State Management in Compiled Inference Code

**Date**: 2026-02-10
**Status**: Design Document
**Related**: [ARCHITECTURE.md](ARCHITECTURE.md), [MLIR-COMPILATION-DESIGN.md](MLIR-COMPILATION-DESIGN.md)

---

## Overview

This document explains the design of the opaque state pointer used throughout the compiled inference code interface, and how it maps to the internal HIP context implementation.

**Key Insight**: The same entity has two different names depending on perspective:
- **External perspective** (C interface user): Opaque `void*` called **"state"**
- **Internal perspective** (MLIR implementation): Concrete type `!hip.context` called **"context"**

This naming reflects the abstraction boundary between the generic interface and HIP-specific implementation.

**Self-Contained Design**: The internal state structure is private to each compiled DLL. This means:
- ✅ Can freely add/remove fields (e.g., handles for rocBLAS, rocFFT, rocRAND, etc.)
- ✅ No external dependencies on struct layout
- ✅ Each compiled model evolves independently
- ✅ Zero impact on C interface or CustomOp code

This isolation is fundamental to the architecture's extensibility and maintainability.

---

## Dual Perspective

### External Perspective: Opaque State

**From the CustomOp runtime's point of view:**

The compiled DLL exports a simple C interface with an opaque pointer:

```c
// User sees: opaque void* (implementation-agnostic)
int inference_init(void** out_state);       // Creates some state
int inference_compute(void* state, ...);     // Uses that state
int inference_cleanup(void* state);          // Destroys that state
```

**Key characteristics**:
- **Opaque**: User doesn't know what's inside
- **Implementation-agnostic**: Could be HIP, CUDA, SYCL, CPU, etc.
- **Generic naming**: "state" conveys "runtime execution state" without backend-specific implications
- **Clean abstraction**: CustomOp has zero dependencies on HIP headers

**Why "state" and not "context"?**
- "State" is generic and neutral
- Avoids confusion with GPU-specific terms like "CUDA context" or "HIP context"
- Makes the interface usable across different backend implementations
- Clear ownership model: "state" implies lifecycle management (create, use, destroy)

### Internal Perspective: HIP Context

**From the compiled code's implementation point of view:**

Inside the MLIR dialect and compiled code, this pointer refers to a concrete HIP execution context:

```mlir
// Implementation sees: concrete HIP context type
func.func @main(%ctx: !hip.context, %input: memref<...>, %output: memref<...>) -> i32 {
  // Operations know %ctx contains HIP stream, MIOpen handle, etc.
  hip.conv(%ctx, %input, %weights, %bias, %output) {...}
  return %c0_i32 : i32
}
```

**Key characteristics**:
- **Concrete type**: `!hip.context` is a specific MLIR dialect type
- **Backend-specific**: Contains HIP stream, MIOpen handles, GPU memory pointers
- **Implementation detail**: Not exposed in C interface
- **Type safety**: MLIR operations require `!hip.context`, can't pass wrong type

**Why `!hip.context` and not `!hip.state`?**
- Matches HIP/CUDA terminology conventions
- Clear that it's HIP-specific execution context
- Familiar to GPU programmers (analogous to `hipStream_t`, `cudaContext_t`)
- Type name documents the backend choice

---

## Terminology Standard

### In C Interface (External)

**Always use "state":**

```c
void* state;                              // ✅ Opaque state pointer
int inference_init(void** out_state);     // ✅ Create state
int inference_compute(void* state, ...);  // ✅ Use state
int inference_cleanup(void* state);       // ✅ Destroy state
```

**Never use "context" in C interface:**
```c
void* context;                            // ❌ Too HIP-specific
int inference_init(void** out_context);   // ❌ Exposes implementation
```

### In MLIR (Internal)

**HIP dialect: Use "context" with `!hip.context` type:**

```mlir
func.func @main(%ctx: !hip.context, ...) -> i32 {          // ✅ HIP context
  hip.conv(%ctx, ...) {...}                                 // ✅ Pass context
}
```

**LLVM dialect: Use "state" with `!llvm.ptr` (already lowered):**

```mlir
func.func @inference_compute(%state: !llvm.ptr, ...) -> i32 {  // ✅ Lowered to opaque ptr
  %stream_ptr = llvm.getelementptr %state[0, 0] : ...          // ✅ Access state fields
}
```

### In Documentation

**Use "state" when discussing:**
- C interface design
- External API contract
- CustomOp runtime behavior
- Generic execution state concepts

**Use "context" when discussing:**
- HIP dialect operations
- MLIR type system
- GPU execution model
- Internal implementation details

---

## State Structure

### Conceptual Layout

The opaque `void* state` pointer points to this concrete structure:

```c
// INTERNAL IMPLEMENTATION (not exposed in C interface)
struct HipExecutionState {
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

### MLIR Representation

**In HIP dialect:**
```mlir
!hip.context  // Opaque type (no internal structure exposed)
```

**In LLVM dialect:**
```mlir
!llvm.struct<(
  ptr,              // hip_stream
  ptr,              // miopen_handle
  ptr,              // hipblas_handle
  array<N x ptr>    // gpu_constants (N = constant count)
)>
```

### Field Access Pattern

**Accessing GPU handles:**
```mlir
// Get miopenHandle (field 1)
%miopen_ptr = llvm.getelementptr %state[0, 1] : (!llvm.ptr) -> !llvm.ptr
%miopen = llvm.load %miopen_ptr : !llvm.ptr

// Get hipStream (field 0)
%stream_ptr = llvm.getelementptr %state[0, 0] : (!llvm.ptr) -> !llvm.ptr
%stream = llvm.load %stream_ptr : !llvm.ptr
```

**Accessing constant GPU pointers:**
```mlir
// Get first constant (gpu_constants[0])
%weights_array = llvm.getelementptr %state[0, 3] : (!llvm.ptr) -> !llvm.ptr
%weight_0_ptr = llvm.getelementptr %weights_array[0] : (!llvm.ptr) -> !llvm.ptr
%weight_0_gpu = llvm.load %weight_0_ptr : !llvm.ptr
```

---

## Lifecycle

### 1. Creation (inference_init)

**C interface:**
```c
void* state;
int ret = inference_init(&state);  // Creates and initializes state
```

**MLIR implementation:**
```mlir
func.func @inference_init(%out_state: !llvm.ptr<!llvm.ptr>) -> i32 {
  // 1. Allocate state struct on heap
  %state = llvm.call @malloc(%state_size) : (i64) -> !llvm.ptr

  // 2. Create GPU handles
  %stream = ... // hipStreamCreate
  %miopen = ... // miopenCreate
  %hipblas = ... // hipblasLtCreate

  // 3. Upload constants to GPU
  %weight_gpu = ... // hipMalloc + hipMemcpy for each constant

  // 4. Store everything in state struct
  llvm.store %stream, %state[0, 0]
  llvm.store %miopen, %state[0, 1]
  llvm.store %weight_gpu, %state[0, 3, 0]
  // ...

  // 5. Return state pointer
  llvm.store %state, %out_state
  return %c0_i32 : i32
}
```

**Result**: `state` pointer is ready for use

### 2. Usage (inference_compute)

**C interface:**
```c
int ret = inference_compute(state, inputs, outputs);  // Uses pre-initialized state
```

**MLIR implementation:**
```mlir
func.func @inference_compute(%state: !llvm.ptr, %inputs: !llvm.ptr, %outputs: !llvm.ptr) -> i32 {
  // 1. Extract handles from state
  %miopen = ... // Load from state[0, 1]
  %stream = ... // Load from state[0, 0]

  // 2. Extract pre-uploaded constants
  %weights_gpu = ... // Load from state[0, 3, 0]

  // 3. Execute operations using handles and weights
  llvm.call @miopenConvolutionForward(%miopen, %stream, %weights_gpu, ...)

  return %c0_i32 : i32
}
```

**Key point**: State is read-only during compute (no allocation/deallocation)

### 3. Destruction (inference_cleanup)

**C interface:**
```c
int ret = inference_cleanup(state);  // Frees all resources
```

**MLIR implementation:**
```mlir
func.func @inference_cleanup(%state: !llvm.ptr) -> i32 {
  // 1. Free GPU constant memory
  %weight_gpu = ... // Load from state[0, 3, 0]
  llvm.call @hipFree(%weight_gpu)

  // 2. Destroy GPU handles
  llvm.call @miopenDestroy(%miopen)
  llvm.call @hipStreamDestroy(%stream)

  // 3. Free state struct itself
  llvm.call @free(%state)

  return %c0_i32 : i32
}
```

**Result**: All resources freed, `state` pointer invalid

---

## Design Rationale

### Why Opaque Pointer (Not Structured Type)?

**Decision**: C interface uses `void*`, not exposed struct.

**Rationale**:

1. **Encapsulation**: Implementation can change without breaking interface
   - Adding new GPU library → add field to struct, no interface change
   - Optimization (descriptor cache) → internal change only

2. **Portability**: Same interface works for different backends
   - HIP implementation: struct contains `miopenHandle_t`
   - CUDA implementation: struct contains `cudnnHandle_t`
   - CPU implementation: struct contains thread pool
   - CustomOp code unchanged

3. **ABI Stability**: Struct layout changes don't break binary compatibility
   - Compiled DLLs can evolve independently
   - No header file dependency between CustomOp and compiled code

4. **Type Safety Where It Matters**: MLIR enforces `!hip.context` type internally
   - External simplicity (void*) + internal safety (!hip.context)

### Why Not Thread-Local Storage?

**Alternative considered**: Global state with thread-local storage (TLS)

```c
// Alternative: TLS-based approach
__thread void* g_hip_context;

int inference_init() {
    g_hip_context = create_context();
    return 0;
}

int inference_compute(span_t inputs, span_t outputs) {
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

**Alternative**: `hipMallocHost` for pinned memory (better CPU-GPU transfer performance)

---

## Relationship to Constants

The state structure contains pointers to pre-uploaded constants (weights, biases).

**See [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md) for detailed design.**

**Summary**:
- Constants embedded in DLL as `llvm.mlir.global`
- Uploaded to GPU in `inference_init`
- GPU pointers stored in `state->gpu_constants[]`
- Accessed in `inference_compute` via state
- Freed in `inference_cleanup`

**Example**:
```mlir
// In inference_init:
%weight_cpu = llvm.mlir.addressof @constant_0 : !llvm.ptr
%weight_gpu = ... // hipMalloc + hipMemcpy
llvm.store %weight_gpu, %state[0, 3, 0]  // Store in state

// In inference_compute:
%weight_gpu = llvm.load %state[0, 3, 0]  // Load from state
llvm.call @miopenConvolutionForward(..., %weight_gpu, ...)
```

---

## Cross-Reference Table

| Concept | C Interface | HIP Dialect MLIR | LLVM Dialect MLIR | Purpose |
|---------|-------------|------------------|-------------------|---------|
| **Name** | state | context | state | Execution state |
| **Type** | `void*` | `!hip.context` | `!llvm.ptr` | Opaque → Typed → Lowered |
| **Parameter** | `void* state` | `%ctx: !hip.context` | `%state: !llvm.ptr` | Function parameter |
| **Creation** | `inference_init(&state)` | N/A (created in LLVM) | `llvm.call @malloc` | Heap allocation |
| **Usage** | `inference_compute(state, ...)` | `hip.conv(%ctx, ...)` | `llvm.getelementptr %state` | Pass to operations |
| **Destruction** | `inference_cleanup(state)` | N/A (freed in LLVM) | `llvm.call @free` | Resource cleanup |

---

## Naming Conventions Summary

### C Interface (External API)
```c
void* state;                              // ✅ Always "state"
inference_init(&state);
inference_compute(state, inputs, outputs);
inference_cleanup(state);
```

### HIP Dialect (High-Level MLIR)
```mlir
func.func @main(%ctx: !hip.context, ...) -> i32 {  // ✅ Always "ctx" or "context"
  hip.conv(%ctx, ...) {...}
}
```

### LLVM Dialect (Low-Level MLIR)
```mlir
func.func @inference_compute(%state: !llvm.ptr, ...) -> i32 {  // ✅ Always "state"
  %handle = llvm.getelementptr %state[0, 1] : ...
}
```

### Documentation
- **"state"** = C interface, opaque pointer, external perspective
- **"context"** = HIP context, MLIR type, internal perspective
- **"HIP execution state"** = full name when clarification needed

---

## Future Extensions

### Adding New GPU Libraries

**Example**: Add rocFFT for FFT operations

**Required changes**:
1. **State struct**: Add field
   ```c
   struct HipExecutionState {
       // ... existing fields ...
       rocfft_plan fft_plan;  // NEW field
   };
   ```

2. **MLIR struct type**: Add field
   ```mlir
   !llvm.struct<(ptr, ptr, ptr, array<N x ptr>, ptr)>  // Added 5th field
   ```

3. **inference_init**: Create rocFFT plan
   ```mlir
   %fft_plan = llvm.call @rocfft_plan_create(...)
   llvm.store %fft_plan, %state[0, 4]  // Store at new offset
   ```

4. **inference_cleanup**: Destroy rocFFT plan
   ```mlir
   %fft_plan = llvm.load %state[0, 4]
   llvm.call @rocfft_plan_destroy(%fft_plan)
   ```

**No changes needed**:
- ✅ C interface (`void* state` unchanged)
- ✅ CustomOp code (opaque pointer)
- ✅ HIP dialect operations (still use `!hip.context`)

### Supporting Multiple Backends

**Vision**: Same C interface, different implementations

**HIP implementation**:
```c
struct HipExecutionState {
    hipStream_t stream;
    miopenHandle_t miopenHandle;
    // ...
};
```

**CUDA implementation**:
```c
struct CudaExecutionState {
    cudaStream_t stream;
    cudnnHandle_t cudnnHandle;
    // ...
};
```

**Both compile to same C interface**:
```c
int inference_init(void** out_state);  // Creates HipExecutionState OR CudaExecutionState
int inference_compute(void* state, ...);
int inference_cleanup(void* state);
```

**CustomOp unchanged** - works with both backends.

---

## References

- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system architecture
- [MLIR-COMPILATION-DESIGN.md](MLIR-COMPILATION-DESIGN.md) - MLIR module structure
- [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md) - How constants are stored in state
- HIP Stream Management: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#stream-management

---

**Document Status**: Complete - defines terminology standard for all documents
