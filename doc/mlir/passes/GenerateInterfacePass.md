# GenerateInterfacePass

**Location:** To be implemented
**Input:** LLVM dialect module with @main + constant helpers
**Output:** LLVM dialect module + C interface wrappers

---

## Overview

The GenerateInterfacePass generates the three C interface functions that are exported from the compiled DLL. These functions wrap the internal @main and constant helper functions.

**Generated functions:**
1. `inference_init` - Allocates context, creates GPU handles, uploads constants
2. `inference_compute` - Parses span_t, builds memrefs, calls @main
3. `inference_cleanup` - Frees GPU resources, destroys handles

---

## Prerequisites

Before this pass can run, the module MUST satisfy these requirements.

**Why prerequisites matter:** [GenerateInterfacePass](GenerateInterfacePass.md) generates C interface functions that bridge between user-provided `span_t` arrays and MLIR's internal `@main` function. This requires precise knowledge of:
- How many inputs/outputs to expect (for validation)
- What rank each tensor has (for memref struct construction)
- Where to find constants and GPU handles (for initialization)

When @main signature becomes `(context, inputs, outputs) → i32` in [HipToLLVM pass](HipToLLVM.md), type information is lost. Module metadata and helper functions preserve this information.

### Prerequisite 1: @main Function Signature

**Required signature:**
```mlir
llvm.func @main(%context: !llvm.ptr,
                %inputs: !llvm.ptr,   // Pointer to array of input memref structs
                %outputs: !llvm.ptr)  // Pointer to array of output memref structs
                -> i32
```

**Requirements:**
- Function named `@main` exists
- First parameter is context pointer
- Second parameter is array of input memref structs (struct-by-value)
- Third parameter is array of output memref structs (struct-by-value)
- Returns i32 status code
- Memref structs contain runtime dimension values (dynamic shape support)

### Prerequisite 2: Module Metadata

**Required module attributes:**
```mlir
module attributes {
  hipdnn.input_count = 2 : i64,              // N inputs
  hipdnn.input_ranks = dense<[4, 2]> : tensor<2xi64>,  // ranks for each input
  hipdnn.output_count = 2 : i64,             // M outputs
  hipdnn.output_ranks = dense<[2, 1]> : tensor<2xi64>  // ranks for each output
}
```

**Why metadata is critical:** When @main signature becomes `(context, inputs, outputs) → i32`, type information is lost (arrays have no compile-time size). Metadata compensates for this loss by preserving:
- **Input/output counts**: How many tensors to validate and process
- **Tensor ranks**: Determines memref struct layout (`rank 4` → `array<4xi64>` for sizes/strides)
- **Loop bounds**: How many iterations when building memref arrays

This enables the pass to:
1. Generate validation code (check user provides correct number of tensors)
2. Build memref structs with correct rank-dependent layout
3. Allocate arrays of correct size
4. Iterate over inputs/outputs correctly

### Prerequisite 3: Constant Management Helpers

**Required functions:**

```mlir
llvm.func @get_constant_count() -> i64
llvm.func @initialize_constants(%context: !llvm.ptr) -> i32
llvm.func @release_constants(%context: !llvm.ptr) -> i32
```

**Contracts:**
- `get_constant_count()`: Pure function, returns number of constants
- `initialize_constants(context)`: Expects context with handles created and gpu_constants allocated
- `release_constants(context)`: Frees GPU constant memory

### Prerequisite 4: Interface Functions Must NOT Exist

**CRITICAL: The pass must fail if interface functions already exist.**

The following functions must NOT be present in the module:
- `inference_init`
- `inference_compute`
- `inference_cleanup`

**Why?** These are the OUTPUT of this pass. If they already exist:
- The pass has already run (accidental double-run)
- Another pass/tool generated them (naming conflict)
- Attempting to generate them again would cause duplicate symbol errors

**What to do:** The pass should check for these symbols in `verifyPrerequisites()` and fail if found.

See [../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md) for complete prerequisite details.

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
- Receives input/output tensors from caller (via `span_t` structures)
- Validates tensor counts and ranks match model expectations
- Loads runtime dimension values from `tensor_t.shape` pointers
- Builds MLIR memref descriptors with runtime dimensions
- Calls the internal `@main` function to perform GPU computation
- Returns status code (0 = success)

**Purpose:** Fast execution path - all resources already created, just run the model.

**Analogy:** Like calling an already-loaded function - minimal overhead, maximum speed.

**Critical feature:** Supports dynamic shapes - dimension values are loaded at runtime from the caller's tensor metadata.

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

## Detailed MLIR Implementations

### Function 1: inference_init

**C Signature:**
```c
int inference_init(void** out_state);
```

**MLIR Implementation:**
```mlir
llvm.func @inference_init(%out_state: !llvm.ptr<!llvm.ptr>) -> i32 {
  // Define constants used throughout function
  %c0_i32 = llvm.mlir.constant(0 : i32) : i32
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64
  %null = llvm.mlir.zero : !llvm.ptr

  // ============================================================================
  // Step 1: Allocate context struct on heap
  // ============================================================================
  %context_size = llvm.mlir.constant(32 : i64) : i64  // 4 pointers × 8 bytes = 32
  %context = llvm.call @malloc(%context_size) : (i64) -> !llvm.ptr

  // Check if allocation failed (malloc returns NULL on failure)
  %is_null = llvm.icmp "eq" %context, %null : !llvm.ptr
  llvm.cond_br %is_null, ^error_alloc, ^cont1

^cont1:
  // ============================================================================
  // Step 2: Create HIP stream for asynchronous GPU operations
  // ============================================================================
  %stream_ptr = llvm.alloca %c1_i64 x !llvm.ptr : (i64) -> !llvm.ptr  // Stack allocate pointer to receive handle
  %stream_ret = llvm.call @hipStreamCreate(%stream_ptr) : (!llvm.ptr) -> i32

  // Check if stream creation failed (returns non-zero on error)
  %stream_failed = llvm.icmp "ne" %stream_ret, %c0_i32 : i32
  llvm.cond_br %stream_failed, ^error_stream, ^cont2

^cont2:
  // Stream created successfully - load it and store in context.stream (field 0)
  %stream = llvm.load %stream_ptr : !llvm.ptr
  %stream_field = llvm.getelementptr %context[0, 0] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %stream, %stream_field : !llvm.ptr

  // ============================================================================
  // Step 3: Create MIOpen handle for DNN operations
  // ============================================================================
  %miopen_ptr = llvm.alloca %c1_i64 x !llvm.ptr : (i64) -> !llvm.ptr
  %miopen_ret = llvm.call @miopenCreate(%miopen_ptr) : (!llvm.ptr) -> i32

  // Check if MIOpen handle creation failed
  %miopen_failed = llvm.icmp "ne" %miopen_ret, %c0_i32 : i32
  llvm.cond_br %miopen_failed, ^error_miopen, ^cont3

^cont3:
  // Load MIOpen handle and associate it with our stream
  %miopen = llvm.load %miopen_ptr : !llvm.ptr
  %setstream_ret = llvm.call @miopenSetStream(%miopen, %stream) : (!llvm.ptr, !llvm.ptr) -> i32

  // Check if stream association failed
  %setstream_failed = llvm.icmp "ne" %setstream_ret, %c0_i32 : i32
  llvm.cond_br %setstream_failed, ^error_miopen_stream, ^cont4

^cont4:
  // Store MIOpen handle in context.miopenHandle (field 1)
  %miopen_field = llvm.getelementptr %context[0, 1] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %miopen, %miopen_field : !llvm.ptr

  // ============================================================================
  // Step 4: Create hipBLASLt handle for matrix operations
  // ============================================================================
  %hipblas_ptr = llvm.alloca %c1_i64 x !llvm.ptr : (i64) -> !llvm.ptr
  %hipblas_ret = llvm.call @hipblasLtCreate(%hipblas_ptr) : (!llvm.ptr) -> i32

  // Check if hipBLAS handle creation failed
  %hipblas_failed = llvm.icmp "ne" %hipblas_ret, %c0_i32 : i32
  llvm.cond_br %hipblas_failed, ^error_hipblas, ^cont5

^cont5:
  // Store hipBLAS handle in context.hipblasHandle (field 2)
  %hipblas = llvm.load %hipblas_ptr : !llvm.ptr
  %hipblas_field = llvm.getelementptr %context[0, 2] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %hipblas, %hipblas_field : !llvm.ptr

  // ============================================================================
  // Step 5: Allocate array to hold GPU constant pointers
  // ============================================================================
  %count = llvm.call @get_constant_count() : () -> i64
  %ptr_size = llvm.mlir.constant(8 : i64) : i64
  %array_size = llvm.mul %count, %ptr_size : i64  // count × sizeof(void*)
  %gpu_constants = llvm.call @malloc(%array_size) : (i64) -> !llvm.ptr

  // Check if allocation failed
  %constants_null = llvm.icmp "eq" %gpu_constants, %null : !llvm.ptr
  llvm.cond_br %constants_null, ^error_constants_alloc, ^cont6

^cont6:
  // Store gpu_constants array pointer in context.gpu_constants (field 3)
  %gpu_constants_field = llvm.getelementptr %context[0, 3] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %gpu_constants, %gpu_constants_field : !llvm.ptr

  // ============================================================================
  // Step 6: Upload constants to GPU and populate gpu_constants array
  // ============================================================================
  %init_ret = llvm.call @initialize_constants(%context) : (!llvm.ptr) -> i32

  // Check if constant initialization failed
  %init_failed = llvm.icmp "ne" %init_ret, %c0_i32 : i32
  llvm.cond_br %init_failed, ^error_init, ^success

// ==============================================================================
// SUCCESS PATH: Return context pointer to caller
// ==============================================================================
^success:
  llvm.store %context, %out_state : !llvm.ptr
  llvm.return %c0_i32 : i32

// ==============================================================================
// ERROR PATHS: Clean up what was created, then return error code
// ==============================================================================

^error_init:
  // initialize_constants failed - need to destroy ALL handles and free everything
  // NOTE: initialize_constants cleans up its own partial work, so we don't
  // need to call release_constants

  // Load handles from context struct (they were stored successfully)
  %hipblas_ptr_err = llvm.getelementptr %context[0, 2] : (!llvm.ptr) -> !llvm.ptr
  %hipblas_err = llvm.load %hipblas_ptr_err : !llvm.ptr
  llvm.call @hipblasLtDestroy(%hipblas_err) : (!llvm.ptr) -> i32

  %miopen_ptr_err = llvm.getelementptr %context[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %miopen_err = llvm.load %miopen_ptr_err : !llvm.ptr
  llvm.call @miopenDestroy(%miopen_err) : (!llvm.ptr) -> i32

  %stream_ptr_err = llvm.getelementptr %context[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %stream_err = llvm.load %stream_ptr_err : !llvm.ptr
  llvm.call @hipStreamDestroy(%stream_err) : (!llvm.ptr) -> i32

  // Free gpu_constants array (was allocated successfully)
  %gpu_constants_ptr_err = llvm.getelementptr %context[0, 3] : (!llvm.ptr) -> !llvm.ptr
  %gpu_constants_err = llvm.load %gpu_constants_ptr_err : !llvm.ptr
  llvm.call @free(%gpu_constants_err) : (!llvm.ptr) -> ()

  llvm.call @free(%context) : (!llvm.ptr) -> ()
  %c3_i32 = llvm.mlir.constant(3 : i32) : i32
  llvm.return %c3_i32 : i32

^error_constants_alloc:
  // gpu_constants allocation failed - destroy all handles, free context
  %hipblas_ptr_ca = llvm.getelementptr %context[0, 2] : (!llvm.ptr) -> !llvm.ptr
  %hipblas_ca = llvm.load %hipblas_ptr_ca : !llvm.ptr
  llvm.call @hipblasLtDestroy(%hipblas_ca) : (!llvm.ptr) -> i32

  %miopen_ptr_ca = llvm.getelementptr %context[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %miopen_ca = llvm.load %miopen_ptr_ca : !llvm.ptr
  llvm.call @miopenDestroy(%miopen_ca) : (!llvm.ptr) -> i32

  %stream_ptr_ca = llvm.getelementptr %context[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %stream_ca = llvm.load %stream_ptr_ca : !llvm.ptr
  llvm.call @hipStreamDestroy(%stream_ca) : (!llvm.ptr) -> i32

  llvm.call @free(%context) : (!llvm.ptr) -> ()
  %c4_i32 = llvm.mlir.constant(4 : i32) : i32
  llvm.return %c4_i32 : i32

^error_hipblas:
  // hipBLAS creation failed - destroy MIOpen handle and stream, free context
  %miopen_ptr_hb = llvm.getelementptr %context[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %miopen_hb = llvm.load %miopen_ptr_hb : !llvm.ptr
  llvm.call @miopenDestroy(%miopen_hb) : (!llvm.ptr) -> i32

  %stream_ptr_hb = llvm.getelementptr %context[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %stream_hb = llvm.load %stream_ptr_hb : !llvm.ptr
  llvm.call @hipStreamDestroy(%stream_hb) : (!llvm.ptr) -> i32

  llvm.call @free(%context) : (!llvm.ptr) -> ()
  %c5_i32 = llvm.mlir.constant(5 : i32) : i32
  llvm.return %c5_i32 : i32

^error_miopen_stream:
  // miopenSetStream failed - destroy MIOpen handle, destroy stream, free context
  // Note: %miopen is in scope (loaded before setStream call)
  llvm.call @miopenDestroy(%miopen) : (!llvm.ptr) -> i32

  %stream_ptr_ms = llvm.getelementptr %context[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %stream_ms = llvm.load %stream_ptr_ms : !llvm.ptr
  llvm.call @hipStreamDestroy(%stream_ms) : (!llvm.ptr) -> i32

  llvm.call @free(%context) : (!llvm.ptr) -> ()
  %c6_i32 = llvm.mlir.constant(6 : i32) : i32
  llvm.return %c6_i32 : i32

^error_miopen:
  // miopenCreate failed - destroy stream, free context
  %stream_ptr_m = llvm.getelementptr %context[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %stream_m = llvm.load %stream_ptr_m : !llvm.ptr
  llvm.call @hipStreamDestroy(%stream_m) : (!llvm.ptr) -> i32

  llvm.call @free(%context) : (!llvm.ptr) -> ()
  %c7_i32 = llvm.mlir.constant(7 : i32) : i32
  llvm.return %c7_i32 : i32

^error_stream:
  // Stream creation failed - only need to free context
  llvm.call @free(%context) : (!llvm.ptr) -> ()
  %c2_i32 = llvm.mlir.constant(2 : i32) : i32
  llvm.return %c2_i32 : i32

^error_alloc:
  // Context allocation failed - nothing to clean up
  %c1_i32 = llvm.mlir.constant(1 : i32) : i32
  llvm.return %c1_i32 : i32
}
```

**Error codes:**
- 0: Success
- 1: Context allocation failed (malloc returned NULL)
- 2: Stream creation failed (hipStreamCreate failed)
- 3: Constant initialization failed (initialize_constants failed)
- 4: GPU constants array allocation failed
- 5: hipBLAS handle creation failed
- 6: MIOpen stream association failed (miopenSetStream failed)
- 7: MIOpen handle creation failed (miopenCreate failed)

### Function 2: inference_compute

**C Signature:**
```c
int inference_compute(void* state, span_t inputs, span_t outputs);
```

**MLIR Implementation:**
```mlir
llvm.func @inference_compute(%state: !llvm.ptr,
                              %inputs: !llvm.ptr,   // span_t*
                              %outputs: !llvm.ptr)  // span_t*
                              -> i32 {
  // Read module metadata
  %expected_input_count = <from hipdnn.input_count attribute>
  %expected_input_ranks = <from hipdnn.input_ranks attribute>

  // 1. Validate input count
  %inputs_count_ptr = llvm.getelementptr %inputs[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %inputs_count = llvm.load %inputs_count_ptr : i64
  %count_ok = llvm.icmp "eq" %inputs_count, %expected_input_count : i64
  llvm.cond_br %count_ok, ^validate_ranks, ^error

^validate_ranks:
  // 2. Get tensor_t array
  %inputs_data_ptr = llvm.getelementptr %inputs[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %inputs_data = llvm.load %inputs_data_ptr : !llvm.ptr  // tensor_t*

  // 3. Validate rank of input 0
  %tensor_0 = llvm.getelementptr %inputs_data[0] : (!llvm.ptr) -> !llvm.ptr
  %rank_ptr = llvm.getelementptr %tensor_0[0, 2] : (!llvm.ptr) -> !llvm.ptr
  %rank = llvm.load %rank_ptr : i32
  %expected_rank = <from hipdnn.input_ranks[0]>
  %rank_ok = llvm.icmp "eq" %rank, %expected_rank : i32
  llvm.cond_br %rank_ok, ^build_memrefs, ^error

^build_memrefs:
  // 4. Extract tensor_t fields
  %data_ptr_ptr = llvm.getelementptr %tensor_0[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %data = llvm.load %data_ptr_ptr : !llvm.ptr

  %shape_ptr_ptr = llvm.getelementptr %tensor_0[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %shape_ptr = llvm.load %shape_ptr_ptr : !llvm.ptr  // int64_t*

  // 5. Load RUNTIME dimensions
  %dim0_ptr = llvm.getelementptr %shape_ptr[0] : (!llvm.ptr) -> !llvm.ptr
  %dim0 = llvm.load %dim0_ptr : i64  // RUNTIME!
  %dim1_ptr = llvm.getelementptr %shape_ptr[1] : (!llvm.ptr) -> !llvm.ptr
  %dim1 = llvm.load %dim1_ptr : i64  // RUNTIME!
  %dim2_ptr = llvm.getelementptr %shape_ptr[2] : (!llvm.ptr) -> !llvm.ptr
  %dim2 = llvm.load %dim2_ptr : i64  // RUNTIME!
  %dim3_ptr = llvm.getelementptr %shape_ptr[3] : (!llvm.ptr) -> !llvm.ptr
  %dim3 = llvm.load %dim3_ptr : i64  // RUNTIME!

  // 6. Calculate RUNTIME strides (row-major)
  %stride3 = llvm.mlir.constant(1 : i64) : i64
  %stride2 = llvm.mul %dim3, %stride3 : i64  // Runtime calculation!
  %stride1 = llvm.mul %dim2, %stride2 : i64
  %stride0 = llvm.mul %dim1, %stride1 : i64

  // 7. Build memref struct
  %data_gpu = llvm.addrspacecast %data : !llvm.ptr to !llvm.ptr<1>
  %memref = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %memref = llvm.insertvalue %data_gpu, %memref[0] : ...     // allocated_ptr
  %memref = llvm.insertvalue %data_gpu, %memref[1] : ...     // aligned_ptr
  %c0_i64 = llvm.mlir.constant(0 : i64) : i64
  %memref = llvm.insertvalue %c0_i64, %memref[2] : ...       // offset
  %memref = llvm.insertvalue %dim0, %memref[3, 0] : ...      // size[0] = RUNTIME
  %memref = llvm.insertvalue %dim1, %memref[3, 1] : ...      // size[1] = RUNTIME
  %memref = llvm.insertvalue %dim2, %memref[3, 2] : ...      // size[2] = RUNTIME
  %memref = llvm.insertvalue %dim3, %memref[3, 3] : ...      // size[3] = RUNTIME
  %memref = llvm.insertvalue %stride0, %memref[4, 0] : ...   // stride[0] = RUNTIME
  %memref = llvm.insertvalue %stride1, %memref[4, 1] : ...   // stride[1] = RUNTIME
  %memref = llvm.insertvalue %stride2, %memref[4, 2] : ...   // stride[2] = RUNTIME
  %memref = llvm.insertvalue %stride3, %memref[4, 3] : ...   // stride[3] = RUNTIME

  // 8. Allocate array and store memref struct
  %input_array = llvm.alloca %c1 x !llvm.struct<...> : (i64) -> !llvm.ptr
  llvm.store %memref, %input_array : !llvm.ptr

  // 9. Do similar for outputs
  // ... (same process for output tensors)

  // 10. Call @main
  %ret = llvm.call @main(%state, %input_array, %output_array)
    : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

  llvm.return %ret : i32

^error:
  %c5 = llvm.mlir.constant(5 : i32) : i32  // HIPDNN_ERROR_INVALID_INPUT
  llvm.return %c5 : i32
}
```

**Key aspects:**
- ✅ Validates input/output counts and ranks
- ✅ Loads **runtime dimensions** from tensor_t.shape
- ✅ Calculates **runtime strides** from dimensions
- ✅ Builds memref structs with runtime values
- ✅ Calls @main with arrays of memref structs

### Function 3: inference_cleanup

**C Signature:**
```c
int inference_cleanup(void* state);
```

**MLIR Implementation:**
```mlir
llvm.func @inference_cleanup(%state: !llvm.ptr) -> i32 {
  // 1. Release constants
  llvm.call @release_constants(%state) : (!llvm.ptr) -> i32

  // 2. Load handles from context
  %hipblas_ptr = llvm.getelementptr %state[0, 2] : (!llvm.ptr) -> !llvm.ptr
  %hipblas = llvm.load %hipblas_ptr : !llvm.ptr

  %miopen_ptr = llvm.getelementptr %state[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %miopen = llvm.load %miopen_ptr : !llvm.ptr

  %stream_ptr = llvm.getelementptr %state[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %stream = llvm.load %stream_ptr : !llvm.ptr

  // 3. Destroy handles
  llvm.call @hipblasLtDestroy(%hipblas) : (!llvm.ptr) -> i32
  llvm.call @miopenDestroy(%miopen) : (!llvm.ptr) -> i32
  llvm.call @hipStreamDestroy(%stream) : (!llvm.ptr) -> i32

  // 4. Free gpu_constants array
  %gpu_constants_ptr = llvm.getelementptr %state[0, 3] : (!llvm.ptr) -> !llvm.ptr
  %gpu_constants = llvm.load %gpu_constants_ptr : !llvm.ptr
  llvm.call @free(%gpu_constants) : (!llvm.ptr) -> ()

  // 5. Free context
  llvm.call @free(%state) : (!llvm.ptr) -> ()

  %c0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %c0 : i32
}
```

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

    // Check interface functions DON'T exist (prevent double-run)
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("inference_init")) return false;
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("inference_compute")) return false;
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("inference_cleanup")) return false;

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
