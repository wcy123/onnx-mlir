# GenerateInterfacePass

**Location:** To be implemented
**Input:** LLVM dialect module with @main + constant helpers
**Output:** LLVM dialect module + C interface wrappers

---

## Overview

The GenerateInterfacePass generates the three C interface functions that are exported from the compiled DLL. These functions wrap the internal @main and constant helper functions.

**Generated functions:**
1. `inference_init` - Allocates context, creates GPU handles, uploads constants
2. `inference_compute` - Parses span_t*, builds memrefs, calls @main
3. `inference_cleanup` - Frees GPU resources, destroys handles

---

## Prerequisites

This pass requires 5 prerequisites to be met by prior passes. For detailed specifications, see [../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md#generateinterfacepass-prerequisites).

**Summary:**

1. **@main function signature**: `(context, inputs, outputs) -> i32` with struct-by-value memrefs supporting dynamic shapes
2. **Module metadata**: Attributes specifying input/output counts and tensor ranks
3. **Constant management functions**: `get_constant_count`, `initialize_constants`, `release_constants`
4. **Context struct layout**: Stream, MIOpen/hipBLAS handles, GPU constants array
5. **Tensor interface types**: `tensor_t` and `span_t` structs for C interface

**Why prerequisites matter:** GenerateInterfacePass bridges between user-provided `span_t` arrays and MLIR's internal `@main` function. When @main signature becomes `(context, inputs, outputs) → i32`, type information is lost. Metadata and helper functions preserve this information to enable correct validation and memref construction.

---

## Generated Code Overview

GenerateInterfacePass generates three C-compatible functions that form the public interface of the compiled DLL. These functions manage the complete lifecycle of GPU inference execution.

### High-Level Function Roles

**1. `inference_init` - One-Time Setup**

Creates and initializes all GPU resources needed for inference:
- Allocates an opaque state structure (the "context")
- Creates GPU handles (stream, MIOpen, hipBLAS)
- Uploads model constants (weights, biases) to GPU memory
- Stores GPU constant pointers in the state structure
- Returns the state pointer to the caller

**Purpose:** Do all expensive initialization work once, so subsequent inference calls are fast.

**Analogy:** Like loading a program into memory and initializing it - done once at startup.

---

**2. `inference_compute` - The Fast Path**

Executes the actual inference using pre-initialized resources:
- Receives input/output tensors from caller (via `span_t` structures - CPU memory)
- Validates tensor counts and ranks match model expectations
- Loads runtime dimension values from `tensor_t.shape` pointers
- **Copies input data from CPU to GPU (H2D transfer)** via hipMemcpy
- Builds MLIR memref descriptors with runtime dimensions (pointing to GPU memory)
- Calls the internal `@main` function to perform GPU computation
- **Copies output data from GPU back to CPU (D2H transfer)** via hipMemcpy
- Returns status code (0 = success)

**Purpose:** Fast execution path - all resources already created, just run the model.

**Analogy:** Like calling an already-loaded function - minimal overhead, maximum speed.

**Critical features:**
- Supports dynamic shapes - dimension values are loaded at runtime from the caller's tensor metadata
- Handles data movement transparently - caller provides CPU pointers, GPU computation happens automatically

---

**3. `inference_cleanup` - Teardown**

Releases all GPU resources created by `inference_init`:
- Calls `release_constants` to free GPU memory for weights/biases
- Destroys GPU handles (hipBLAS, MIOpen, stream) in reverse order
- Frees the gpu_constants array
- Frees the state structure itself
- State pointer becomes invalid after this call

**Purpose:** Clean shutdown - prevent GPU memory leaks.

**Analogy:** Like shutting down a program - release all resources back to the OS.

---

### Usage Pattern

```c
// Once at startup
void* state;
inference_init(&state);

// Many times (hot path)
for (int i = 0; i < num_requests; i++) {
  inference_compute(state, inputs, outputs);
}

// Once at shutdown
inference_cleanup(state);
```

---

## C-ABI Safety

**CRITICAL:** All generated functions MUST be C-ABI compatible for cross-language DLL calling.

**Required attributes:**
```mlir
llvm.func @function_name(...) -> i32 attributes {
  llvm.emit_c_interface,     // Use C calling convention (cdecl/sysv)
  sym_visibility = "public"  // Export symbol from DLL
}
```

**Platform-specific export (alternative):**
- **Windows**: `passthrough = ["dllexport"]`
- **Linux**: `sym_visibility = "public"` (default visibility)

**Why this matters:**
- Without `llvm.emit_c_interface`: May use wrong calling convention, causing stack corruption
- Without export attributes: Symbol won't be visible in DLL export table
- Name mangling: LLVM functions use mangled names by default; C interface ensures `extern "C"` semantics

---

## Detailed MLIR Implementations

### Function 1: inference_init

**C Signature:**
```c
int inference_init(void** out_state);
```

**MLIR Implementation:**
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

  // 5. Allocate gpu_constants array
  %count = llvm.call @get_constant_count() : () -> i64
  %ptr_size = llvm.mlir.constant(8 : i64) : i64
  %array_size = llvm.mul %count, %ptr_size : i64
  %gpu_constants = llvm.call @malloc(%array_size) : (i64) -> !llvm.ptr
  %gpu_constants_field = llvm.getelementptr %context[0, 3] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %gpu_constants, %gpu_constants_field : !llvm.ptr

  // 6. Initialize constants
  %init_ret = llvm.call @initialize_constants(%context) : (!llvm.ptr) -> i32
  %init_failed = llvm.icmp "ne" %init_ret, %c0 : i32
  llvm.cond_br %init_failed, ^error_init, ^success

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

  // Get stream from context for asynchronous transfer
  %stream_ptr = llvm.getelementptr %state[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %stream = llvm.load %stream_ptr : !llvm.ptr -> !llvm.ptr

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
  %c5_i32 = llvm.mlir.constant(5 : i32) : i32  // HIPDNN_ERROR_INVALID_INPUT
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

**MLIR Implementation:**
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
  llvm.cond_br %sync_failed, ^error_sync, ^release_constants

^release_constants:
  // ============================================================================
  // Step 2: Free GPU constant memory (weights, biases)
  // ============================================================================
  %release_ret = llvm.call @release_constants(%state) : (!llvm.ptr) -> i32

  // Check if constant release succeeded
  %release_failed = llvm.icmp "ne" %release_ret, %c0_i32 : i32
  llvm.cond_br %release_failed, ^error_release, ^destroy_handles

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
  // release_constants failed - still destroy handles and free memory
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
  // Continue with release_constants despite sync failure
  %release_ret_sync = llvm.call @release_constants(%state) : (!llvm.ptr) -> i32

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

See [../../DYNAMIC-SHAPE-DESIGN.md](../../DYNAMIC-SHAPE-DESIGN.md) for complete flow.

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

    // Check constant helpers exist
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("get_constant_count")) return false;
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("initialize_constants")) return false;
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("release_constants")) return false;

    return true;
  }
};
```

---

## Related Documents

- [OnnxToHip.md](OnnxToHip.md) - Generates @main and constant helpers
- [HipToLLVM.md](HipToLLVM.md) - Transforms @main to final signature
- [../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md) - Complete prerequisite details
- [../../DYNAMIC-SHAPE-DESIGN.md](../../DYNAMIC-SHAPE-DESIGN.md) - Dynamic shape flow
- [../CONSTANT-MANAGEMENT.md](../CONSTANT-MANAGEMENT.md) - Constant handling
