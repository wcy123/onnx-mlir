<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

**Date:** 2026-02-12
**Status:** Self-Reviewed
**Branch:** `mlir-integration`
**Related:** [HipToLLVM.md](HipToLLVM.md), [OnnxToHip.md](OnnxToHip.md), [../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md)

---

## Table of Contents

- [GenerateInterfacePass](#generateinterfacepass)
- [Overview](#overview)
- [Stable Contract Summary](#stable-contract-summary)
- [Prerequisites](#prerequisites)
  - [A. Verified Prerequisites](#a-verified-prerequisites)
  - [B. Design Contracts](#b-design-contracts)
  - [Prerequisites Summary Table](#prerequisites-summary-table)
- [Detailed MLIR Implementations](#detailed-mlir-implementations)
  - [Function 1: inference_init](#function-1-inference_init)
  - [Function 2: inference_compute](#function-2-inference_compute)
  - [Function 3: inference_cleanup](#function-3-inference_cleanup)
- [Verifying C-ABI Compliance](#verifying-c-abi-compliance)
- [Dynamic Shape Support](#dynamic-shape-support)
- [Implementation Strategy](#implementation-strategy)
- [Related Documents](#related-documents)

---

# GenerateInterfacePass

**Input:** LLVM dialect module with @main + constant helpers
**Output:** LLVM dialect module + C interface wrappers

---

## Overview

The GenerateInterfacePass generates the three C interface functions that are exported from the compiled DLL. These functions implement the interface defined in [../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md) and wrap the internal @main and constant helper functions.

**Generated functions:**
1. `inference_init` - Allocates context, creates GPU handles, uploads constants
2. `inference_compute` - Parses span_t*, builds memrefs, calls @main
3. `inference_cleanup` - Frees GPU resources, destroys handles

**This document describes:** Implementation details - how to generate the MLIR code

**For design rationale:** See [../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md) for WHAT and WHY

---

## Why This Pass Exists

This pass serves as an **adapter layer** between two incompatible representations:
- **MLIR world**: Statically-typed memref structs with compile-time rank information
- **C API world**: Dynamic span_t/tensor_t structures with runtime rank information

The pass cannot be eliminated and moved to runtime because MLIR's type system requires compile-time knowledge of tensor ranks to generate memref struct types (`{ptr, ptr, i64, [N x i64], [N x i64]}`). Moving this to runtime would require hardcoding MLIR's internal memref layout in C++ code, creating fragile coupling and losing type safety.

**For detailed analysis of alternatives and trade-offs**, see [../../WHY-GENERATEINTERFACEPASS.md](../../WHY-GENERATEINTERFACEPASS.md).

---

## Stable Contract Summary

This pass depends on a simple contract from prior passes. **Any future pass that satisfies this contract can work with GenerateInterfacePass.**

### Required Functions:

1. **@main function** (from HipToLLVM)
   - Signature: `llvm.func @main(%ctx: !llvm.ptr, %inputs: !llvm.ptr, %outputs: !llvm.ptr) -> i32`
   - Behavior: Processes arrays of input/output memref structs, returns status code

2. **get_constant_registry function** (from OnnxToHip)
   - Signature: `llvm.func @get_constant_registry() -> !llvm.ptr`
   - Returns: Pointer to ConstantRegistry struct with model weights metadata

### Required Module Attributes:

3. **I/O Metadata** (from OnnxToHip)
   - `hipdnn.input_count: i64` - Number of model inputs
   - `hipdnn.input_ranks: dense<[...]>` - Rank of each input tensor
   - `hipdnn.output_count: i64` - Number of model outputs
   - `hipdnn.output_ranks: dense<[...]>` - Rank of each output tensor

**Extensibility:** Future passes can replace OnnxToHip/HipToLLVM as long as they produce these functions and attributes.

---

## Prerequisites

Before the `GenerateInterfacePass` can run, prior passes must establish certain prerequisites. This section is divided into:

1. **A. Verified Prerequisites** - What `verifyPrerequisites()` actually checks in the code
2. **B. Design Contracts** - Important conventions not enforced by code but critical for correct implementation

**CRITICAL REQUIREMENT: Dynamic Shape Support**

All prerequisites MUST support **dynamic shapes from Day 1**. For complete dynamic shape design and rationale, see [../DYNAMIC-SHAPE-DESIGN.md](../DYNAMIC-SHAPE-DESIGN.md).

Summary:
- ✅ Tensor **rank** is compile-time known (e.g., 4D tensor)
- ✅ Dimension **values** are runtime (loaded from tensor_t.shape pointer)
- ✅ No interface changes needed for dynamic shapes
- ✅ All memref operations must work with runtime dimension values

---

## A. Verified Prerequisites

The following prerequisites are **enforced by code** via the `verifyPrerequisites()` function. The pass will fail with an error message if any of these are missing.

### Prerequisite 0: Idempotency Check

**What the code checks:** Pass verifies that `inference_init`, `inference_compute`, and `inference_cleanup` don't already exist.

**Why it matters:** Prevents duplicate interface generation if pass runs multiple times.

**Error message:**
```
[GenerateInterface] Interface functions already exist. Pass already ran.
```

### Prerequisite 1: @main Function

**Required signature:**
```mlir
llvm.func @main(%context: !llvm.ptr,
                %inputs: !llvm.ptr,
                %outputs: !llvm.ptr) -> i32
```

**What the code checks:**
1. `@main` exists as `llvm.func` (NOT `func.func`)
2. Has exactly 3 parameters, all `!llvm.ptr`
3. Returns `i32`

**Satisfied by:** HipToLLVM pass (`transformMainFunction()` method)

**Error messages:**
```
[GenerateInterface] @main (llvm.func) not found
[GenerateInterface] @main is func.func, needs llvm.func. Run --convert-hip-to-llvm first.
[GenerateInterface] @main has wrong signature. Expected: (ptr, ptr, ptr) -> i32
```

**See also:** [HipToLLVM.md](HipToLLVM.md) for implementation details

### Prerequisite 2: get_constant_registry Function

**Required signature (LLVM dialect):**
```mlir
llvm.func @get_constant_registry() -> !llvm.ptr
```

**Returns:** Pointer to static `ConstantRegistry` struct containing:
```c
struct ConstantInfo {
    const void* cpu_data;      // CPU pointer to constant in DLL .data section
    size_t size_bytes;         // Size in bytes
    size_t element_size;       // Element size (4 for float32, 2 for float16, etc.)
    size_t num_elements;       // Number of elements
};

struct ConstantRegistry {
    const ConstantInfo* constants;  // Array of constant descriptors
    size_t count;                   // Number of constants
};
```

**What the code checks:**
1. `@get_constant_registry` exists as `llvm.func`
2. Has no parameters
3. Returns `!llvm.ptr`

**Satisfied by:** OnnxToHip pass

**Usage:** Runtime calls this in `inference_init` to get metadata about all constants, then uploads them to GPU using its own strategy (batch upload, pinned memory, etc.). Runtime owns complete GPU memory lifecycle.

**Error messages:**
```
[GenerateInterface] get_constant_registry (llvm.func) not found
[GenerateInterface] get_constant_registry has wrong signature. Expected: () -> ptr
```

**Required attributes:**
```mlir
module attributes {
  hipdnn.input_count = 2 : i64,
  hipdnn.input_ranks = dense<[4, 2]> : tensor<2xi64>,
  hipdnn.output_count = 1 : i64,
  hipdnn.output_ranks = dense<[2]> : tensor<1xi64>
}
```

**What the code checks:**
1. `hipdnn.input_count` attribute exists
2. `hipdnn.input_ranks` attribute exists
3. `hipdnn.output_count` attribute exists
4. `hipdnn.output_ranks` attribute exists

**Satisfied by:** OnnxToHip pass

**Error messages:**
```
[GenerateInterface] hipdnn.input_count attribute missing
[GenerateInterface] hipdnn.input_ranks attribute missing
[GenerateInterface] hipdnn.output_count attribute missing
[GenerateInterface] hipdnn.output_ranks attribute missing
```

**Why this matters:** When @main signature becomes `(context, inputs, outputs) → i32`, type information is lost. Metadata preserves this for memref struct construction.

---

## B. Design Contracts

The following are **NOT enforced by code** in `verifyPrerequisites()`, but are critical design contracts that the GenerateInterfacePass implementation depends on. These document important conventions and requirements.

### @main Function Behavioral Contract

The code only checks that @main exists with the correct signature `(ptr, ptr, ptr) -> i32`. However, the implementation relies on the following behavioral contracts:

**Design decisions:**
- **Calling convention:** Struct-by-value memrefs (NOT unpacked descriptors)
- **Multiple I/O support:** Arrays of memref structs
- **Dynamic shape support:** Memref size/stride arrays contain **runtime values**

**What GenerateInterfacePass expects:**
1. First parameter is state/context pointer
2. Second parameter points to array of input memref structs with **runtime dimensions**
3. Third parameter points to array of output memref structs with **runtime dimensions**
4. Returns i32 status code (0 = success, non-zero = error)
5. **Critical:** Memref size/stride arrays populated with runtime values (from tensor_t.shape)

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

  // Computation using runtime dimensions...
  %c0_i32 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %c0_i32 : i32
}
```

**See also:** [HipToLLVM.md](HipToLLVM.md) for call chain walkthrough and LLVM optimization details.

### Constant Registry Contract

**get_constant_registry() Contract:**

The code verifies this function exists with correct signature, but its internal behavior (documented below) is NOT verified.

**Postconditions:**
1. Returns pointer to static `ConstantRegistry` struct (valid for DLL lifetime)
2. Registry contains metadata for all constants in the model
3. `cpu_data` pointers point to LLVM globals in DLL .data section (valid for DLL lifetime)
4. Runtime uses this metadata to allocate/upload/free GPU memory

**Runtime ownership:**
- **DLL owns:** Constant data in .data section, ConstantRegistry struct (static lifetime)
- **Runtime owns:** GPU memory allocation, upload strategy, cleanup strategy
- **Separation:** DLL provides metadata, runtime controls GPU memory lifecycle

**See also:** [../../CONSTANT-HANDLING-DESIGN.md](../../CONSTANT-HANDLING-DESIGN.md) for complete constant handling details.

### RuntimeState Contract (Opaque Handle Design)

**Reference:** [../../RUNTIME-ARCHITECTURE.md](../../RUNTIME-ARCHITECTURE.md) - Opaque Handle Design

**RuntimeState structure:**
```c
// C struct - layout known only to inference_init and inference_cleanup
struct RuntimeState {
    hipStream_t stream;              // GPU stream for async operations
    miopenHandle_t miopenHandle;     // MIOpen library handle
    hipblasLtHandle_t hipblasHandle; // hipBLAS library handle
    void** gpu_constants;            // POINTER to dynamically allocated array
};
```

**MLIR representation:**
```mlir
// RuntimeState is opaque !llvm.ptr
!llvm.ptr
```

**Design:**
- **Constructor/destructor** (`inference_init`, `inference_cleanup`): Access fields directly via GEP
- **Regular computation code** (@main, wrappers): Use accessor functions (`runtime_get_stream`, `hip_get_constant`)
- **Benefit**: Runtime can change struct layout without breaking computation code (only constructor/destructor need updates)

**Initialization order:**
1. Stream created first
2. MIOpen/hipBLAS handles created and associated with stream
3. Constants uploaded to GPU using handles

### Error Handling Contract

**Required policy (not enforced by code):**
- Fail fast on errors
- Cleanup all allocated resources on error paths
- Return distinct error codes for each failure type
- No exceptions (pure C-ABI)

**Error code scheme:**

See [../INTERFACE-DESIGN.md - Error Codes](../INTERFACE-DESIGN.md#33-error-codes-consolidated) for the complete error code specification.

**Example error handling pattern in inference_init:**
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
  llvm.return %c3_i32 : i32  // ERROR_CONSTANT_INIT
}
```

**Policy:**
- ✅ **Fail fast:** Return error immediately on failure
- ✅ **Cleanup on error:** Free all resources allocated before error
- ✅ **Propagate errors:** Pass through error codes from @main and helpers
- ✅ **No exceptions:** Pure C ABI, use integer error codes
- ✅ **Validate inputs:** Check span_t/tensor_t pointers are non-null

### Tensor Interface Contract (span_t and tensor_t)

**Defined in:** CustomOp header (external to MLIR compilation)

**Required structures:**
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
   - Map `data_type` to element size (FLOAT32=4, FLOAT16=2, INT8=1)
3. Build memref struct with **runtime dimensions**
4. Calculate **runtime strides** from dimension values

**Key Design Decisions:**
- ✅ **Keep tensor_t simple:** Don't match memref structure exactly
- ✅ **CRITICAL: Support dynamic shapes from Day 1:** `shape` pointer provides runtime dimensions
- ✅ **No interface changes needed:** Same interface works for static and dynamic shapes
- ✅ **Runtime stride calculation:** Compute strides from runtime dimension values
- ✅ **Type system:** Rank (4D) is compile-time, dimension values (N, C, H, W) are runtime

**See:** [../INTERFACE-DESIGN.md - Data Structures](../INTERFACE-DESIGN.md#32-data-structures) for complete tensor interface specification.

---

## Prerequisites Summary Table

**Verified Prerequisites (enforced by code):**

| # | Prerequisite | Satisfied By | Code Check |
|---|--------------|--------------|------------|
| 0 | Idempotency | (generated code) | Functions don't exist yet |
| 1 | @main function | HipToLLVM | Signature: `(ptr, ptr, ptr) -> i32` |
| 2 | get_constant_registry | OnnxToHip | Signature: `() -> ptr` |
| 3 | Module metadata | OnnxToHip | 4 attributes exist |

**Design Contracts (NOT enforced by code):**

| Contract | Critical For | Documented In |
|----------|--------------|---------------|
| @main behavioral contract | Correct memref handling | HipToLLVM.md |
| Constant registry contract | GPU memory management | ../../CONSTANT-HANDLING-DESIGN.md |
| RuntimeState opaque design | ABI stability | RUNTIME-ARCHITECTURE.md |
| Error handling policy | Robust error paths | INTERFACE-DESIGN.md |
| Tensor interface (span_t/tensor_t) | C-ABI compatibility | INTERFACE-DESIGN.md |
| Dynamic shape support | Runtime flexibility | DYNAMIC-SHAPE-DESIGN.md |

---

## Detailed MLIR Implementations

### Function 1: inference_init

**C Signature:**
```c
int inference_init(void** out_state);
```

**For design and rationale:** See [../INTERFACE-DESIGN.md - inference_init](../INTERFACE-DESIGN.md#inference_init)

**MLIR Implementation:**

**NOTE:** This function is a CONSTRUCTOR for RuntimeState. It directly accesses RuntimeState fields via GEP operations to initialize them. This is an EXCEPTION to the opaque handle design - regular computation code (@main, wrappers) MUST use accessor functions instead.

```mlir
llvm.func @inference_init(%out_state: !llvm.ptr<!llvm.ptr>) -> i32
    attributes {
      llvm.emit_c_interface,
      sym_visibility = "public"
    } {
  // 1. Allocate context
  %context_size = llvm.mlir.constant(32 : i64) : i64
  %context = llvm.call @malloc(%context_size) : (i64) -> !llvm.ptr
  %is_null = llvm.icmp "eq" %context, %null : !llvm.ptr
  llvm.cond_br %is_null, ^error_alloc, ^cont1

^cont1:
  // 2. Create stream
  %stream_ptr = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr
  %stream_ret = llvm.call @hipStreamCreate(%stream_ptr) : (!llvm.ptr) -> i32
  %stream_failed = llvm.icmp "ne" %stream_ret, %c0 : i32
  llvm.cond_br %stream_failed, ^error_stream, ^cont2

^cont2:
  %stream = llvm.load %stream_ptr : !llvm.ptr
  %stream_field = llvm.getelementptr %context[0, 0] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %stream, %stream_field : !llvm.ptr

  // 3. Create MIOpen handle
  %miopen_ptr = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr
  llvm.call @miopenCreate(%miopen_ptr) : (!llvm.ptr) -> i32
  %miopen = llvm.load %miopen_ptr : !llvm.ptr
  llvm.call @miopenSetStream(%miopen, %stream) : (!llvm.ptr, !llvm.ptr) -> i32
  %miopen_field = llvm.getelementptr %context[0, 1] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %miopen, %miopen_field : !llvm.ptr

  // 4. Create hipBLAS handle
  %hipblas_ptr = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr
  llvm.call @hipblasLtCreate(%hipblas_ptr) : (!llvm.ptr) -> i32
  %hipblas = llvm.load %hipblas_ptr : !llvm.ptr
  %hipblas_field = llvm.getelementptr %context[0, 2] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %hipblas, %hipblas_field : !llvm.ptr

  // 5. Get constant registry and allocate gpu_constants array
  %registry_ptr = llvm.call @get_constant_registry() : () -> !llvm.ptr
  %count_ptr = llvm.getelementptr %registry_ptr[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %count = llvm.load %count_ptr : !llvm.ptr -> i64
  %ptr_size = llvm.mlir.constant(8 : i64) : i64
  %array_size = llvm.mul %count, %ptr_size : i64
  %gpu_constants = llvm.call @malloc(%array_size) : (i64) -> !llvm.ptr
  %gpu_constants_field = llvm.getelementptr %context[0, 3] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %gpu_constants, %gpu_constants_field : !llvm.ptr

  // 6. Upload constants to GPU (runtime owns strategy)
  %constants_array_ptr_ptr = llvm.getelementptr %registry_ptr[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %constants_array_ptr = llvm.load %constants_array_ptr_ptr : !llvm.ptr -> !llvm.ptr

  // Loop through all constants and upload
  // (In actual implementation, this would be an unrolled loop or scf.for)
  // For each i in 0..count:
  //   info = constants_array_ptr[i]
  //   hipMalloc(&gpu_constants[i], info.size_bytes)
  //   hipMemcpy(gpu_constants[i], info.cpu_data, info.size_bytes, H2D)

  %upload_success = <constant upload loop> : i32
  %upload_failed = llvm.icmp "ne" %upload_success, %c0 : i32
  llvm.cond_br %upload_failed, ^error_init, ^success

^success:
  llvm.store %context, %out_state : !llvm.ptr
  llvm.return %c0 : i32

^error_init:
  // Cleanup: destroy handles, free context
  llvm.call @hipblasLtDestroy(%hipblas)
  llvm.call @miopenDestroy(%miopen)
  llvm.call @hipStreamDestroy(%stream)
  llvm.call @free(%gpu_constants)
  llvm.call @free(%context)
  %c3 = llvm.mlir.constant(3 : i32) : i32
  llvm.return %c3 : i32

^error_stream:
  llvm.call @free(%context)
  %c2 = llvm.mlir.constant(2 : i32) : i32
  llvm.return %c2 : i32

^error_alloc:
  %c1 = llvm.mlir.constant(1 : i32) : i32
  llvm.return %c1 : i32
}
```

**Error codes:**
- 0: Success
- 1: Context allocation failed
- 2: Handle creation failed
- 3: Constant initialization failed

### Function 2: inference_compute

**C Signature:**
```c
int inference_compute(void* state, span_t* inputs, span_t* outputs);
```

**For design and rationale:** See [../INTERFACE-DESIGN.md - inference_compute](../INTERFACE-DESIGN.md#inference_compute)

**MLIR Implementation:**

**NOTE:** This is a simplified example showing the key steps for a single 4D input and single 2D output.
For multiple inputs/outputs, the pass would generate loops or unrolled code for each tensor.

```mlir
llvm.func @inference_compute(%state: !llvm.ptr,
                              %inputs: !llvm.ptr,   // span_t* (CPU memory)
                              %outputs: !llvm.ptr)  // span_t* (CPU memory)
                              -> i32
    attributes {
      llvm.emit_c_interface,
      sym_visibility = "public"
    } {
  // Define constants
  %c0_i32 = llvm.mlir.constant(0 : i32) : i32
  %c0_i64 = llvm.mlir.constant(0 : i64) : i64
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64
  %null = llvm.mlir.zero : !llvm.ptr

  // These come from module metadata (hipdnn.input_count, hipdnn.input_ranks, etc.)
  %expected_input_count = llvm.mlir.constant(1 : i64) : i64
  %expected_input_rank = llvm.mlir.constant(4 : i32) : i32  // 4D tensor
  %expected_output_count = llvm.mlir.constant(1 : i64) : i64
  %expected_output_rank = llvm.mlir.constant(2 : i32) : i32  // 2D tensor

  // ============================================================================
  // Step 1: Validate input span_t
  // ============================================================================
  // Extract inputs.count (span_t layout: {tensor_t* data, size_t count})
  %inputs_count_ptr = llvm.getelementptr %inputs[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %inputs_count = llvm.load %inputs_count_ptr : !llvm.ptr -> i64

  // Check count matches expected
  %count_ok = llvm.icmp "eq" %inputs_count, %expected_input_count : i64
  llvm.cond_br %count_ok, ^validate_output_count, ^error

^validate_output_count:
  // Extract outputs.count
  %outputs_count_ptr = llvm.getelementptr %outputs[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %outputs_count = llvm.load %outputs_count_ptr : !llvm.ptr -> i64

  // Check count matches expected
  %out_count_ok = llvm.icmp "eq" %outputs_count, %expected_output_count : i64
  llvm.cond_br %out_count_ok, ^validate_ranks, ^error

^validate_ranks:
  // ============================================================================
  // Step 2: Get tensor_t arrays and validate ranks
  // ============================================================================
  // Get inputs.data (tensor_t* array)
  %inputs_data_ptr = llvm.getelementptr %inputs[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %inputs_data = llvm.load %inputs_data_ptr : !llvm.ptr -> !llvm.ptr

  // Get pointer to input tensor 0 (tensor_t layout: {void* data, int64_t* shape, int rank, int data_type})
  %input_0_ptr = llvm.getelementptr %inputs_data[0] : (!llvm.ptr) -> !llvm.ptr

  // Validate rank (field 2 of tensor_t)
  %input_rank_ptr = llvm.getelementptr %input_0_ptr[0, 2] : (!llvm.ptr) -> !llvm.ptr
  %input_rank = llvm.load %input_rank_ptr : !llvm.ptr -> i32
  %rank_ok = llvm.icmp "eq" %input_rank, %expected_input_rank : i32
  llvm.cond_br %rank_ok, ^allocate_gpu_buffers, ^error

^allocate_gpu_buffers:
  // ============================================================================
  // Step 3: Load runtime dimensions and calculate buffer sizes
  // ============================================================================
  // Extract shape pointer (field 1 of tensor_t)
  %input_shape_ptr_ptr = llvm.getelementptr %input_0_ptr[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %input_shape_ptr = llvm.load %input_shape_ptr_ptr : !llvm.ptr -> !llvm.ptr

  // Load RUNTIME dimension values
  %dim0_ptr = llvm.getelementptr %input_shape_ptr[0] : (!llvm.ptr) -> !llvm.ptr
  %dim0 = llvm.load %dim0_ptr : !llvm.ptr -> i64  // Batch size (runtime!)

  %dim1_ptr = llvm.getelementptr %input_shape_ptr[1] : (!llvm.ptr) -> !llvm.ptr
  %dim1 = llvm.load %dim1_ptr : !llvm.ptr -> i64  // Channels (runtime!)

  %dim2_ptr = llvm.getelementptr %input_shape_ptr[2] : (!llvm.ptr) -> !llvm.ptr
  %dim2 = llvm.load %dim2_ptr : !llvm.ptr -> i64  // Height (runtime!)

  %dim3_ptr = llvm.getelementptr %input_shape_ptr[3] : (!llvm.ptr) -> !llvm.ptr
  %dim3 = llvm.load %dim3_ptr : !llvm.ptr -> i64  // Width (runtime!)

  // Calculate total elements and byte size
  %elem_count_01 = llvm.mul %dim0, %dim1 : i64
  %elem_count_012 = llvm.mul %elem_count_01, %dim2 : i64
  %total_elements = llvm.mul %elem_count_012, %dim3 : i64
  %element_size = llvm.mlir.constant(4 : i64) : i64  // sizeof(float) = 4
  %input_byte_size = llvm.mul %total_elements, %element_size : i64

  // Allocate GPU memory for input (H2D transfer target)
  %input_gpu_ptr_ptr = llvm.alloca %c1_i64 x !llvm.ptr : (i64) -> !llvm.ptr
  %malloc_ret = llvm.call @hipMalloc(%input_gpu_ptr_ptr, %input_byte_size)
    : (!llvm.ptr, i64) -> i32

  // Check if allocation succeeded
  %malloc_failed = llvm.icmp "ne" %malloc_ret, %c0_i32 : i32
  llvm.cond_br %malloc_failed, ^error, ^copy_input_h2d

^copy_input_h2d:
  // ============================================================================
  // Step 4: Copy input data from CPU to GPU (H2D)
  // ============================================================================
  %input_gpu_ptr = llvm.load %input_gpu_ptr_ptr : !llvm.ptr -> !llvm.ptr

  // Extract CPU data pointer (field 0 of tensor_t)
  %input_cpu_ptr_ptr = llvm.getelementptr %input_0_ptr[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %input_cpu_ptr = llvm.load %input_cpu_ptr_ptr : !llvm.ptr -> !llvm.ptr

  // Get stream from state via accessor (OPAQUE - no GEP)
  %stream = llvm.call @runtime_get_stream(%state) : (!llvm.ptr) -> !llvm.ptr

  // hipMemcpyAsync(dst=GPU, src=CPU, size, hipMemcpyHostToDevice, stream)
  %hipMemcpyHostToDevice = llvm.mlir.constant(1 : i32) : i32
  %memcpy_ret = llvm.call @hipMemcpyAsync(
    %input_gpu_ptr, %input_cpu_ptr, %input_byte_size, %hipMemcpyHostToDevice, %stream
  ) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32

  %memcpy_failed = llvm.icmp "ne" %memcpy_ret, %c0_i32 : i32
  llvm.cond_br %memcpy_failed, ^error_free_input, ^build_input_memref

^build_input_memref:
  // ============================================================================
  // Step 5: Build memref descriptor for input (pointing to GPU memory)
  // ============================================================================
  // Calculate runtime strides (row-major: stride[i] = product(dims[i+1:]))
  %stride3 = llvm.mlir.constant(1 : i64) : i64
  %stride2 = llvm.mul %dim3, %stride3 : i64
  %stride1 = llvm.mul %dim2, %stride2 : i64
  %stride0 = llvm.mul %dim1, %stride1 : i64

  // Build memref struct (4D tensor)
  %input_gpu_addrspace = llvm.addrspacecast %input_gpu_ptr : !llvm.ptr to !llvm.ptr<1>
  %input_memref = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %input_gpu_addrspace, %input_memref[0]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %input_gpu_addrspace, %input_memref[1]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %c0_i64, %input_memref[2]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %dim0, %input_memref[3, 0]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %dim1, %input_memref[3, 1]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %dim2, %input_memref[3, 2]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %dim3, %input_memref[3, 3]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %stride0, %input_memref[4, 0]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %stride1, %input_memref[4, 1]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %stride2, %input_memref[4, 2]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %stride3, %input_memref[4, 3]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>

  // Store in array to pass to @main
  %input_array = llvm.alloca %c1_i64 x !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
    : (i64) -> !llvm.ptr
  llvm.store %input_memref, %input_array : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>, !llvm.ptr

  // ============================================================================
  // Step 6: Allocate GPU buffer for output (similar process)
  // ============================================================================
  // NOTE: Similar code for output tensor (2D in this example)
  // - Get outputs.data array
  // - Load output dimensions from output tensor_t
  // - Allocate GPU memory for output
  // - Build output memref (NO H2D copy - output buffer starts empty)
  // ... (omitted for brevity - same pattern as input)

  %output_array = <similar construction for 2D output>

  // ============================================================================
  // Step 7: Call @main to perform GPU computation
  // ============================================================================
  %main_ret = llvm.call @main(%state, %input_array, %output_array)
    : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

  // Check if @main succeeded
  %main_failed = llvm.icmp "ne" %main_ret, %c0_i32 : i32
  llvm.cond_br %main_failed, ^error_free_all, ^copy_output_d2h

^copy_output_d2h:
  // ============================================================================
  // Step 8: Copy output data from GPU back to CPU (D2H)
  // ============================================================================
  %output_gpu_ptr = <extract from output_memref>
  %output_cpu_ptr = <extract from output tensor_t>
  %output_byte_size = <calculated from output dimensions>

  // hipMemcpyAsync(dst=CPU, src=GPU, size, hipMemcpyDeviceToHost, stream)
  %hipMemcpyDeviceToHost = llvm.mlir.constant(2 : i32) : i32
  %d2h_ret = llvm.call @hipMemcpyAsync(
    %output_cpu_ptr, %output_gpu_ptr, %output_byte_size, %hipMemcpyDeviceToHost, %stream
  ) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32

  %d2h_failed = llvm.icmp "ne" %d2h_ret, %c0_i32 : i32
  llvm.cond_br %d2h_failed, ^error_free_all, ^synchronize

^synchronize:
  // ============================================================================
  // Step 9: Wait for all GPU operations to complete
  // ============================================================================
  %sync_ret = llvm.call @hipStreamSynchronize(%stream) : (!llvm.ptr) -> i32

  %sync_failed = llvm.icmp "ne" %sync_ret, %c0_i32 : i32
  llvm.cond_br %sync_failed, ^error_free_all, ^cleanup

^cleanup:
  // ============================================================================
  // Step 10: Free temporary GPU buffers and return success
  // ============================================================================
  llvm.call @hipFree(%input_gpu_ptr) : (!llvm.ptr) -> i32
  llvm.call @hipFree(%output_gpu_ptr) : (!llvm.ptr) -> i32

  llvm.return %c0_i32 : i32

// ==============================================================================
// ERROR PATHS
// ==============================================================================

^error_free_all:
  // Computation or D2H transfer failed - free both buffers
  llvm.call @hipFree(%input_gpu_ptr) : (!llvm.ptr) -> i32
  llvm.call @hipFree(%output_gpu_ptr) : (!llvm.ptr) -> i32
  %c8_i32 = llvm.mlir.constant(8 : i32) : i32
  llvm.return %c8_i32 : i32

^error_free_input:
  // H2D copy failed - free input buffer only
  llvm.call @hipFree(%input_gpu_ptr) : (!llvm.ptr) -> i32
  %c9_i32 = llvm.mlir.constant(9 : i32) : i32
  llvm.return %c9_i32 : i32

^error:
  // Validation or allocation failed before any GPU memory allocated
  %c5_i32 = llvm.mlir.constant(5 : i32) : i32  // ERROR_INVALID_INPUT
  llvm.return %c5_i32 : i32
}
```

**Key implementation notes:**
- ✅ Validates input/output counts and ranks
- ✅ Loads **runtime dimensions** from tensor_t.shape pointers
- ✅ Calculates **runtime strides** from dimensions (row-major layout)
- ✅ **Allocates GPU buffers** for inputs and outputs
- ✅ **H2D transfer**: Copies input data from CPU to GPU via hipMemcpyAsync
- ✅ Builds memref structs pointing to **GPU memory** with runtime dimensions
- ✅ Calls @main with memref arrays
- ✅ **D2H transfer**: Copies output data from GPU back to CPU
- ✅ **Synchronizes stream** to ensure transfers complete
- ✅ **Frees temporary GPU buffers** (model weights stay in state, only I/O buffers freed)
- ✅ Proper error handling with cleanup paths

### Function 3: inference_cleanup

**C Signature:**
```c
int inference_cleanup(void* state);
```

**For design and rationale:** See [../INTERFACE-DESIGN.md - inference_cleanup](../INTERFACE-DESIGN.md#inference_cleanup)

**MLIR Implementation:**

**NOTE:** This function is a DESTRUCTOR for RuntimeState. It directly accesses RuntimeState fields via GEP operations to read and cleanup resources. This is an EXCEPTION to the opaque handle design - regular computation code (@main, wrappers) MUST use accessor functions instead.

```mlir
llvm.func @inference_cleanup(%state: !llvm.ptr) -> i32
    attributes {
      llvm.emit_c_interface,
      sym_visibility = "public"
    } {
  // Define constants
  %c0_i32 = llvm.mlir.constant(0 : i32) : i32

  // ============================================================================
  // Step 1: Synchronize stream before cleanup
  // ============================================================================
  // CRITICAL: Wait for all pending GPU operations to complete
  // If we destroy handles while GPU is still working, we'll get crashes
  %stream_ptr = llvm.getelementptr %state[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %stream = llvm.load %stream_ptr : !llvm.ptr

  %sync_ret = llvm.call @hipStreamSynchronize(%stream) : (!llvm.ptr) -> i32

  // Check if synchronization succeeded
  %sync_failed = llvm.icmp "ne" %sync_ret, %c0_i32 : i32
  llvm.cond_br %sync_failed, ^error_sync, ^free_gpu_constants

^free_gpu_constants:
  // ============================================================================
  // Step 2: Free GPU constant memory (weights, biases)
  // ============================================================================
  // Get constant count and gpu_constants array
  %registry_ptr = llvm.call @get_constant_registry() : () -> !llvm.ptr
  %count_ptr = llvm.getelementptr %registry_ptr[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %count = llvm.load %count_ptr : !llvm.ptr -> i64

  %gpu_constants_ptr_load = llvm.getelementptr %state[0, 3] : (!llvm.ptr) -> !llvm.ptr
  %gpu_constants_load = llvm.load %gpu_constants_ptr_load : !llvm.ptr

  // Loop through all constants and free GPU memory
  // (In actual implementation, this would be an unrolled loop or scf.for)
  // For each i in 0..count:
  //   hipFree(gpu_constants[i])

  %free_success = <constant free loop> : i32
  %free_failed = llvm.icmp "ne" %free_success, %c0_i32 : i32
  llvm.cond_br %free_failed, ^error_release, ^destroy_handles

^destroy_handles:
  // ============================================================================
  // Step 3: Destroy GPU handles in REVERSE creation order
  // ============================================================================
  // Order: hipBLAS (created last) → MIOpen → stream (created first)

  // Load hipBLAS handle (field 2)
  %hipblas_ptr = llvm.getelementptr %state[0, 2] : (!llvm.ptr) -> !llvm.ptr
  %hipblas = llvm.load %hipblas_ptr : !llvm.ptr

  // Destroy hipBLAS handle
  %hipblas_ret = llvm.call @hipblasLtDestroy(%hipblas) : (!llvm.ptr) -> i32
  %hipblas_failed = llvm.icmp "ne" %hipblas_ret, %c0_i32 : i32
  llvm.cond_br %hipblas_failed, ^error_hipblas, ^destroy_miopen

^destroy_miopen:
  // Load MIOpen handle (field 1)
  %miopen_ptr = llvm.getelementptr %state[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %miopen = llvm.load %miopen_ptr : !llvm.ptr

  // Destroy MIOpen handle
  %miopen_ret = llvm.call @miopenDestroy(%miopen) : (!llvm.ptr) -> i32
  %miopen_failed = llvm.icmp "ne" %miopen_ret, %c0_i32 : i32
  llvm.cond_br %miopen_failed, ^error_miopen, ^destroy_stream

^destroy_stream:
  // Stream already loaded in Step 1
  // Destroy stream
  %stream_ret = llvm.call @hipStreamDestroy(%stream) : (!llvm.ptr) -> i32
  %stream_failed = llvm.icmp "ne" %stream_ret, %c0_i32 : i32
  llvm.cond_br %stream_failed, ^error_stream, ^free_memory

^free_memory:
  // ============================================================================
  // Step 4: Free host memory allocations
  // ============================================================================
  // Free gpu_constants array pointer (field 3)
  %gpu_constants_ptr = llvm.getelementptr %state[0, 3] : (!llvm.ptr) -> !llvm.ptr
  %gpu_constants = llvm.load %gpu_constants_ptr : !llvm.ptr
  llvm.call @free(%gpu_constants) : (!llvm.ptr) -> ()

  // Free context struct itself
  llvm.call @free(%state) : (!llvm.ptr) -> ()

  // Success - all resources released
  llvm.return %c0_i32 : i32

// ==============================================================================
// ERROR PATHS
// ==============================================================================
// Note: Even if cleanup fails, we still free memory to prevent leaks
// Return error codes to inform caller, but don't leave resources dangling

^error_stream:
  // Stream destruction failed - still free memory
  %gpu_constants_ptr_es = llvm.getelementptr %state[0, 3] : (!llvm.ptr) -> !llvm.ptr
  %gpu_constants_es = llvm.load %gpu_constants_ptr_es : !llvm.ptr
  llvm.call @free(%gpu_constants_es) : (!llvm.ptr) -> ()
  llvm.call @free(%state) : (!llvm.ptr) -> ()

  %c10_i32 = llvm.mlir.constant(10 : i32) : i32
  llvm.return %c10_i32 : i32

^error_miopen:
  // MIOpen destruction failed - still destroy stream and free memory
  llvm.call @hipStreamDestroy(%stream) : (!llvm.ptr) -> i32

  %gpu_constants_ptr_em = llvm.getelementptr %state[0, 3] : (!llvm.ptr) -> !llvm.ptr
  %gpu_constants_em = llvm.load %gpu_constants_ptr_em : !llvm.ptr
  llvm.call @free(%gpu_constants_em) : (!llvm.ptr) -> ()
  llvm.call @free(%state) : (!llvm.ptr) -> ()

  %c11_i32 = llvm.mlir.constant(11 : i32) : i32
  llvm.return %c11_i32 : i32

^error_hipblas:
  // hipBLAS destruction failed - still destroy remaining handles and free memory
  %miopen_ptr_eh = llvm.getelementptr %state[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %miopen_eh = llvm.load %miopen_ptr_eh : !llvm.ptr
  llvm.call @miopenDestroy(%miopen_eh) : (!llvm.ptr) -> i32
  llvm.call @hipStreamDestroy(%stream) : (!llvm.ptr) -> i32

  %gpu_constants_ptr_eh = llvm.getelementptr %state[0, 3] : (!llvm.ptr) -> !llvm.ptr
  %gpu_constants_eh = llvm.load %gpu_constants_ptr_eh : !llvm.ptr
  llvm.call @free(%gpu_constants_eh) : (!llvm.ptr) -> ()
  llvm.call @free(%state) : (!llvm.ptr) -> ()

  %c12_i32 = llvm.mlir.constant(12 : i32) : i32
  llvm.return %c12_i32 : i32

^error_release:
  // Constant free loop failed - still destroy handles and free memory
  // Note: GPU memory for constants may be leaked, but we free handles
  %hipblas_ptr_er = llvm.getelementptr %state[0, 2] : (!llvm.ptr) -> !llvm.ptr
  %hipblas_er = llvm.load %hipblas_ptr_er : !llvm.ptr
  llvm.call @hipblasLtDestroy(%hipblas_er) : (!llvm.ptr) -> i32

  %miopen_ptr_er = llvm.getelementptr %state[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %miopen_er = llvm.load %miopen_ptr_er : !llvm.ptr
  llvm.call @miopenDestroy(%miopen_er) : (!llvm.ptr) -> i32

  llvm.call @hipStreamDestroy(%stream) : (!llvm.ptr) -> i32

  %gpu_constants_ptr_er = llvm.getelementptr %state[0, 3] : (!llvm.ptr) -> !llvm.ptr
  %gpu_constants_er = llvm.load %gpu_constants_ptr_er : !llvm.ptr
  llvm.call @free(%gpu_constants_er) : (!llvm.ptr) -> ()
  llvm.call @free(%state) : (!llvm.ptr) -> ()

  %c13_i32 = llvm.mlir.constant(13 : i32) : i32
  llvm.return %c13_i32 : i32

^error_sync:
  // Stream synchronization failed - this is serious, but still try cleanup
  // Try to free GPU constants despite sync failure (best effort)
  // (GPU constant free loop would go here - omitted for brevity)

  %hipblas_ptr_sync = llvm.getelementptr %state[0, 2] : (!llvm.ptr) -> !llvm.ptr
  %hipblas_sync = llvm.load %hipblas_ptr_sync : !llvm.ptr
  llvm.call @hipblasLtDestroy(%hipblas_sync) : (!llvm.ptr) -> i32

  %miopen_ptr_sync = llvm.getelementptr %state[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %miopen_sync = llvm.load %miopen_ptr_sync : !llvm.ptr
  llvm.call @miopenDestroy(%miopen_sync) : (!llvm.ptr) -> i32

  llvm.call @hipStreamDestroy(%stream) : (!llvm.ptr) -> i32

  %gpu_constants_ptr_sync = llvm.getelementptr %state[0, 3] : (!llvm.ptr) -> !llvm.ptr
  %gpu_constants_sync = llvm.load %gpu_constants_ptr_sync : !llvm.ptr
  llvm.call @free(%gpu_constants_sync) : (!llvm.ptr) -> ()
  llvm.call @free(%state) : (!llvm.ptr) -> ()

  %c14_i32 = llvm.mlir.constant(14 : i32) : i32
  llvm.return %c14_i32 : i32
}
```

**Key implementation notes:**
- ✅ **Synchronizes stream first** - Critical to ensure no GPU work is pending
- ✅ **Checks all return values** - Every API call is validated
- ✅ **Destroys in reverse order** - hipBLAS → MIOpen → stream (opposite of creation)
- ✅ **Best-effort cleanup on error** - Even if one step fails, continues to free remaining resources
- ✅ **Prevents memory leaks** - Always frees CPU memory even on error
- ✅ **Clear error codes** - Distinct codes for each failure point (10-14)

**Error codes:**
- 0: Success - all resources released cleanly
- 10: Stream destruction failed
- 11: MIOpen destruction failed
- 12: hipBLAS destruction failed
- 13: Constant release failed (potential GPU memory leak)
- 14: Stream synchronization failed (GPU may still be working)

---

## Verifying C-ABI Compliance

After generating LLVM IR and compiling to DLL, verify exports:

**Windows:**
```bash
dumpbin /EXPORTS inference.dll
# Should show:
#   inference_init
#   inference_compute
#   inference_cleanup
```

**Linux:**
```bash
nm -D inference.so | grep inference
# Should show:
#   T inference_init
#   T inference_compute
#   T inference_cleanup
```

**Calling from C:**
```c
// Load DLL
HMODULE dll = LoadLibrary("inference.dll");  // Windows
// void* dll = dlopen("inference.so", RTLD_NOW);  // Linux

// Get function pointers
typedef int (*inference_init_t)(void**);
inference_init_t init = (inference_init_t)GetProcAddress(dll, "inference_init");

// Call
void* state = NULL;
int ret = init(&state);  // Must work without stack corruption
```

**Common Issues:**
- Missing exports → Symbol not found at runtime
- Wrong calling convention → Stack corruption, crashes
- Name mangling → Can't find symbol (looks for `_Z14inference_initPPv` instead of `inference_init`)

---

## Dynamic Shape Support

**CRITICAL:** This pass implements dynamic shape support!

**How it works:**
1. User provides tensor with shape [2, 3, 256, 256]
2. `inference_compute` loads [2, 3, 256, 256] from `tensor_t.shape` (**runtime!**)
3. Calculates strides: [196608, 65536, 256, 1] (**runtime!**)
4. Builds memref struct with these **runtime values**
5. Passes to @main
6. @main passes to wrappers
7. Wrappers extract dimensions and pass to MIOpen

**No interface changes needed** - same C API for static and dynamic shapes!

See [../DYNAMIC-SHAPE-DESIGN.md](../DYNAMIC-SHAPE-DESIGN.md) for complete flow.

---

## Implementation Strategy

```cpp
class GenerateInterfacePass : public PassWrapper<GenerateInterfacePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // 1. Verify prerequisites
    if (!verifyPrerequisites(module)) {
      signalPassFailure();
      return;
    }

    // 2. Read module metadata
    auto inputCount = module->getAttr("hipdnn.input_count");
    auto inputRanks = module->getAttr("hipdnn.input_ranks");
    auto outputCount = module->getAttr("hipdnn.output_count");
    auto outputRanks = module->getAttr("hipdnn.output_ranks");

    // 3. Generate interface functions
    generateInferenceInit(module);
    generateInferenceCompute(module, inputCount, inputRanks, outputCount, outputRanks);
    generateInferenceCleanup(module);
  }

  bool verifyPrerequisites(ModuleOp module) {
    // Check @main exists and has correct signature
    auto mainFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("main");
    if (!mainFunc) return false;

    // Check module metadata exists
    if (!module->getAttr("hipdnn.input_count")) return false;

    // Check constant registry function exists
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("get_constant_registry")) return false;

    return true;
  }
};
```

---

## Related Documents

**Design:**
- [../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md) - What and why (design rationale, function contracts)
- [../DYNAMIC-SHAPE-DESIGN.md](../DYNAMIC-SHAPE-DESIGN.md) - Dynamic shape architecture

**Prerequisites:**
- [OnnxToHip.md](OnnxToHip.md) - Generates constant helpers and metadata
- [HipToLLVM.md](HipToLLVM.md) - Transforms @main to final signature

**Supporting:**
- [../MODULE-STRUCTURE.md](../MODULE-STRUCTURE.md) - MLIR module organization
- [../LOWERING-PIPELINE.md](../LOWERING-PIPELINE.md) - Complete lowering flow
- [../../CONSTANT-HANDLING-DESIGN.md](../../CONSTANT-HANDLING-DESIGN.md) - Constant handling details
- [../STATE-AND-CONTEXT.md](../STATE-AND-CONTEXT.md) - Runtime state structure
