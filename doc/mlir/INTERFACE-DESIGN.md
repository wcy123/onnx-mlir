# C Interface Design and GenerateInterfacePass Prerequisites

**Related:** [../MLIR-COMPILATION-OVERVIEW.md](../MLIR-COMPILATION-OVERVIEW.md)

---

## Overview

This document describes:
1. The two-layer architecture (C interface vs internal computation)
2. Prerequisites for implementing GenerateInterfacePass
3. Detailed function designs for inference_init/compute/cleanup

---

## Two-Layer Architecture

The compiled DLL has **two layers of functions**:

**Layer 1: C Interface (Public API for CustomOp)**
- `inference_init(void** out_state)` - Exported from DLL
- `inference_compute(void* state, span_t inputs, span_t outputs)` - Exported from DLL
- `inference_cleanup(void* state)` - Exported from DLL

**Layer 2: Internal MLIR Functions (Private)**
- `@main(context, inputs, outputs) -> i32` - Actual computation
- `initialize_constants(context) -> i32` - Upload constants to GPU
- `release_constants(context) -> i32` - Free GPU constant memory
- `get_constant_count() -> i64` - Metadata helper

### Why Two Layers?

**Problem:** Type system impedance mismatch
- **CustomOp** (C code) uses: `span_t` (array of `tensor_t` structs with dynamic shapes)
- **MLIR** (compiled code) uses: `memref` structs (typed descriptors)

**Solution:** Wrapper functions bridge the gap
- `inference_compute` parses `span_t` → builds `memref` descriptors → calls `@main`
- `@main` operates on memrefs (natural for MLIR, works with existing passes)

---

## GenerateInterfacePass Prerequisites

### Overview

Before implementing the `GenerateInterfacePass`, we must establish clear contracts and design decisions that this pass depends on. This section documents the prerequisites that must be satisfied by prior passes (OnnxToHip, HipToLLVM) and the assumptions the GenerateInterfacePass can rely on.

**CRITICAL REQUIREMENT: Dynamic Shape Support**

All prerequisites MUST support **dynamic shapes from Day 1**. This means:
- ✅ Tensor **rank** is compile-time known (e.g., 4D tensor)
- ✅ Dimension **values** are runtime (loaded from tensor_t.shape pointer)
- ✅ No interface changes needed for dynamic shapes
- ✅ All memref operations must work with runtime dimension values

See [../DYNAMIC-SHAPE-DESIGN.md](../DYNAMIC-SHAPE-DESIGN.md) for comprehensive details.

### Prerequisite 1: @main Function Signature (Dynamic Shape Ready)

**Requirement:** The `@main` function must exist with a well-defined signature that supports multiple inputs and outputs with **dynamic shapes**.

**Design Decision:**
- **Calling convention:** Struct-by-value memrefs (NOT unpacked descriptors)
- **Multiple I/O support:** Arrays of memref structs
- **Dynamic shape support:** Memref size/stride arrays contain **runtime values**
- **Signature format:**

```mlir
llvm.func @main(%context: !llvm.ptr,
                %inputs: !llvm.ptr,   // Pointer to array of input memref structs
                %outputs: !llvm.ptr)  // Pointer to array of output memref structs
                -> i32 {
  // Computation logic here
}
```

**Rationale:**
- ✅ **Scalable:** Supports N inputs and M outputs without signature changes
- ✅ **Consistent with MLIR best practices:** Struct-by-value for memrefs
- ✅ **Future-proof:** Can handle models with varying numbers of I/O tensors
- ✅ **Clean interface:** 3 parameters instead of 3 + (11×N) + (11×M) unpacked parameters
- ✅ **Dynamic shape ready:** Memref structs contain runtime dimension values

**What GenerateInterfacePass expects:**
1. `@main` function exists in the module
2. Signature matches the format above exactly
3. First parameter is state/context pointer
4. Second parameter points to array of input memref structs with **runtime dimensions**
5. Third parameter points to array of output memref structs with **runtime dimensions**
6. Returns i32 status code (0 = success, non-zero = error)
7. **Critical:** Memref size/stride arrays populated with runtime values (from tensor_t.shape)

**Example usage in @main (with dynamic shapes):**
```mlir
llvm.func @main(%context: !llvm.ptr,
                %inputs: !llvm.ptr,
                %outputs: !llvm.ptr) -> i32 {
  // Access input 0 (memref struct at inputs[0])
  %input_0_ptr = llvm.getelementptr %inputs[0] : (!llvm.ptr) -> !llvm.ptr
  // Load memref struct - size array contains RUNTIME dimension values!
  %input_0 = llvm.load %input_0_ptr : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>

  // Extract runtime dimensions from memref struct
  %n = llvm.extractvalue %input_0[3, 0] : !llvm.struct<...> -> i64  // Batch size (runtime!)
  %c = llvm.extractvalue %input_0[3, 1] : !llvm.struct<...> -> i64  // Channels (runtime!)
  %h = llvm.extractvalue %input_0[3, 2] : !llvm.struct<...> -> i64  // Height (runtime!)
  %w = llvm.extractvalue %input_0[3, 3] : !llvm.struct<...> -> i64  // Width (runtime!)

  // Access output 0 (memref struct at outputs[0])
  %output_0_ptr = llvm.getelementptr %outputs[0] : (!llvm.ptr) -> !llvm.ptr
  %output_0 = llvm.load %output_0_ptr : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>

  // Computation using input_0 and output_0
  // Wrapper functions extract dimensions from memref structs at runtime
  // ...

  %c0_i32 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %c0_i32 : i32
}
```

**Dynamic Shape Flow:**
```
tensor_t.shape (runtime int64_t* in C)
    ↓ loaded by inference_compute
Runtime dimension values (N, C, H, W)
    ↓ inserted into memref struct
Memref struct.sizes[4] = {N, C, H, W}  (runtime values!)
    ↓ passed to @main
@main extracts dimensions via llvm.extractvalue
    ↓ passed to wrapper functions
Wrappers pass dimensions to MIOpen (runtime!)
```

### Prerequisite 2: Constant Management Function Contracts

**Requirement:** Three helper functions for managing constants must exist with well-defined contracts.

#### Function: `get_constant_count()`

```mlir
llvm.func @get_constant_count() -> i64 {
  %count = llvm.mlir.constant(4 : i64) : i64  // Example: 4 constants
  llvm.return %count : i64
}
```

**Contract:**
- **Inputs:** None
- **Outputs:** Number of constants in the model (i64)
- **Side effects:** None (pure function)
- **Guarantees:** Returns compile-time constant count

#### Function: `initialize_constants(context)`

```mlir
llvm.func @initialize_constants(%context: !llvm.ptr) -> i32 {
  // Upload all constants to GPU
  // Store GPU pointers in context.gpu_constants array
  llvm.return %status : i32
}
```

**Contract:**
- **Inputs:** Fully initialized context pointer
- **Preconditions (what initialize_constants expects):**
  1. ✅ Context struct already allocated (by inference_init)
  2. ✅ GPU handles already created:
     - `context.stream` (hipStream_t) created and valid
     - `context.miopenHandle` (miopenHandle_t) created and set to use stream
     - `context.hipblasHandle` (hipblasLtHandle_t) created
  3. ✅ `context.gpu_constants` pointer already allocated:
     - Array size = `get_constant_count() × sizeof(void*)`
     - Array is uninitialized (initialize_constants fills it)
- **Postconditions (what initialize_constants guarantees):**
  1. All constants uploaded to GPU memory
  2. `context.gpu_constants[i]` points to GPU memory for constant i
  3. GPU memory allocated with `hipMalloc` on `context.stream`
  4. Returns 0 on success, non-zero on error
- **Side effects:** Allocates GPU memory, modifies context.gpu_constants array

#### Function: `release_constants(context)`

```mlir
llvm.func @release_constants(%context: !llvm.ptr) -> i32 {
  // Free all GPU constant memory
  llvm.return %status : i32
}
```

**Contract:**
- **Inputs:** Context pointer with initialized constants
- **Preconditions:**
  1. ✅ `initialize_constants` was called successfully
  2. ✅ `context.gpu_constants` array contains valid GPU pointers
- **Postconditions:**
  1. All GPU memory freed (via hipFree)
  2. `context.gpu_constants` array is in undefined state (caller should free array itself)
  3. Returns 0 on success, non-zero on error
- **Side effects:** Frees GPU memory

### Prerequisite 3: Context Struct Layout

**Requirement:** The runtime context struct must have a well-defined layout.

**Design Decision (from STATE-AND-CONTEXT.md):**

```c
// C struct (for reference - not in MLIR)
struct HipExecutionContext {
    hipStream_t stream;              // field 0: GPU stream for async operations
    miopenHandle_t miopenHandle;     // field 1: MIOpen library handle
    hipblasLtHandle_t hipblasHandle; // field 2: hipBLAS library handle
    void** gpu_constants;            // field 3: POINTER to dynamically allocated array
};
```

**MLIR type representation:**
```mlir
// Context is opaque !llvm.ptr at LLVM level
// Access fields via getelementptr:
%stream_ptr = llvm.getelementptr %context[0, 0] : (!llvm.ptr) -> !llvm.ptr
%miopen_ptr = llvm.getelementptr %context[0, 1] : (!llvm.ptr) -> !llvm.ptr
%hipblas_ptr = llvm.getelementptr %context[0, 2] : (!llvm.ptr) -> !llvm.ptr
%gpu_constants_ptr_ptr = llvm.getelementptr %context[0, 3] : (!llvm.ptr) -> !llvm.ptr
```

**Key Points:**
- ✅ **Terminology:** Use "context" internally, "state" externally (C interface)
- ✅ **gpu_constants is a POINTER:** Not a fixed-size array
  - Allocated dynamically: `malloc(get_constant_count() × sizeof(void*))`
  - Freed by inference_cleanup after calling release_constants
- ✅ **All handles created before initialize_constants:**
  - Stream created first
  - MIOpen/hipBLAS handles created and associated with stream
  - Then initialize_constants can safely use handles

### Prerequisite 4: Error Handling Strategy

**Requirement:** Clear error handling policy for GenerateInterfacePass-generated functions.

**Design Decision:**

#### Error Codes
```c
#define HIPDNN_SUCCESS 0
#define HIPDNN_ERROR_ALLOCATION 1      // malloc/hipMalloc failed
#define HIPDNN_ERROR_HANDLE_CREATION 2 // Stream/handle creation failed
#define HIPDNN_ERROR_CONSTANT_INIT 3   // initialize_constants failed
#define HIPDNN_ERROR_COMPUTATION 4     // @main returned error
#define HIPDNN_ERROR_INVALID_INPUT 5   // Invalid span_t/tensor_t data
```

#### inference_init Error Handling Example
```mlir
llvm.func @inference_init(%out_state: !llvm.ptr<!llvm.ptr>) -> i32 {
  // 1. Allocate context
  %context = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
  %is_null = llvm.icmp "eq" %context, %null : !llvm.ptr
  llvm.cond_br %is_null, ^error_alloc, ^cont1

^cont1:
  // 2. Create stream
  %stream_ret = llvm.call @hipStreamCreate(%stream_ptr) : ...
  %stream_failed = llvm.icmp "ne" %stream_ret, %c0 : i32
  llvm.cond_br %stream_failed, ^error_stream, ^cont2

  // ... more operations

^success:
  llvm.store %context, %out_state : !llvm.ptr
  llvm.return %c0 : i32

^error_init:
  // Cleanup: destroy handles, free context
  llvm.call @hipblasLtDestroy(%hipblas)
  llvm.call @miopenDestroy(%miopen)
  llvm.call @hipStreamDestroy(%stream)
  llvm.call @free(%context)
  llvm.return %c3_i32 : i32  // HIPDNN_ERROR_CONSTANT_INIT
}
```

**Policy:**
- ✅ **Fail fast:** Return error immediately on failure, don't continue
- ✅ **Cleanup on error:** Free all resources allocated before error
- ✅ **Propagate errors:** Pass through error codes from @main and helpers
- ✅ **No exceptions:** Pure C ABI, use integer error codes
- ✅ **Validate inputs:** Check span_t/tensor_t pointers are non-null

### Prerequisite 5: Tensor Interface (span_t and tensor_t)

**Requirement:** C interface types for dynamic tensor data.

**Design (defined in custom-op header):**

```c
// C struct for tensor metadata
typedef struct {
    void* data;        // Pointer to tensor data (CPU or GPU)
    int64_t* shape;    // Pointer to shape array (runtime dimensions)
    int rank;          // Number of dimensions (compile-time known rank)
    int data_type;     // Enum: FLOAT32=0, FLOAT16=1, INT8=2, etc.
} tensor_t;

// C struct for array of tensors
typedef struct {
    tensor_t* data;    // Pointer to array of tensor_t
    size_t count;      // Number of tensors in array
} span_t;
```

**What GenerateInterfacePass must do (CRITICAL: Dynamic Shape Support):**
1. Parse `span_t` to get `tensor_t` array
2. For each `tensor_t`:
   - Extract `data` pointer (cast to !llvm.ptr<1> for GPU address space)
   - Extract `shape` pointer (**LOAD RUNTIME DIMENSION VALUES**)
   - Extract `rank` (compile-time constant for this model)
3. Build memref struct with **runtime dimensions** - see code example in LOWERING-PIPELINE.md

**Key Design Decisions:**
- ✅ **Keep tensor_t simple:** Don't match memref structure exactly
- ✅ **CRITICAL: Support dynamic shapes from Day 1:** `shape` pointer provides runtime dimensions
- ✅ **No interface changes needed:** Same interface works for static and dynamic shapes
- ✅ **Runtime stride calculation:** Compute strides from runtime dimension values
- ✅ **Type system:** Rank (4D) is compile-time, dimension values (N, C, H, W) are runtime

### Summary of Prerequisites

**CRITICAL: All prerequisites MUST support dynamic shapes from Day 1!**

**Before GenerateInterfacePass can be implemented, the module must have:**

1. ✅ **@main function** with signature: `(context, inputs, outputs) -> i32`
   - **Dynamic shape ready:** Memref structs contain runtime dimension values
2. ✅ **Constant helpers:**
   - `get_constant_count() -> i64`
   - `initialize_constants(context) -> i32`
   - `release_constants(context) -> i32`
3. ✅ **Context struct layout:** stream, miopenHandle, hipblasHandle, gpu_constants*
4. ✅ **Error handling policy:** Fail fast, cleanup on error, integer error codes
5. ✅ **Tensor interface:** span_t and tensor_t structs (defined in custom-op)
   - **Dynamic shape ready:** tensor_t.shape provides runtime dimensions

**What GenerateInterfacePass generates:**

1. ✅ **inference_init:** Allocate context, create handles, call initialize_constants
2. ✅ **inference_compute:** Parse span_t, **load runtime dimensions**, build memrefs, call @main
3. ✅ **inference_cleanup:** Call release_constants, destroy handles, free context

**Dynamic Shape Support Summary:**
- Rank: Compile-time known (e.g., 4D tensor)
- Dimensions: Runtime values loaded from tensor_t.shape
- Strides: Calculated at runtime from dimension values
- No interface changes: Same C API for static and dynamic shapes
- See [../DYNAMIC-SHAPE-DESIGN.md](../DYNAMIC-SHAPE-DESIGN.md) for full details

---

## Related Documents

- [MODULE-STRUCTURE.md](MODULE-STRUCTURE.md) - MLIR module organization
- [LOWERING-PIPELINE.md](LOWERING-PIPELINE.md) - Detailed implementations of inference functions
- [CONSTANT-MANAGEMENT.md](CONSTANT-MANAGEMENT.md) - Constant handling details
- [../STATE-AND-CONTEXT.md](../STATE-AND-CONTEXT.md) - Runtime state structure
- [../DYNAMIC-SHAPE-DESIGN.md](../DYNAMIC-SHAPE-DESIGN.md) - Dynamic shape support
