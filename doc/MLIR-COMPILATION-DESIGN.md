# MLIR Compilation Design

**Date:** 2026-02-09
**Status:** Design Document
**Related:** [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Overview

This document describes the detailed design of how ONNX models are compiled through MLIR to native DLL code that implements the 3-function interface (`inference_init`, `inference_compute`, `inference_cleanup`).

**Key Design Decisions:**
1. **Inline lowering**: Each ONNX operation is lowered inline (no function-per-node)
2. **Explicit state passing**: State is passed as function parameter (no thread-local globals)
3. **Constants in DLL**: Weights embedded in `.data` section, uploaded to GPU in `init`
4. **Direct C interface mapping**: MLIR function signatures match C interface exactly

---

## MLIR Module Structure

### Overall Structure

After lowering ONNX → HIP → LLVM, the MLIR module has this structure:

```mlir
module {
  // ============================================================================
  // Constants (weights, biases) - embedded in DLL .data section
  // ============================================================================
  llvm.mlir.global constant @conv_weight(dense<...> : tensor<64x3x3x3xf32>)
    : !llvm.array<1728 x f32>
  llvm.mlir.global constant @conv_bias(dense<...> : tensor<64xf32>)
    : !llvm.array<64 x f32>
  // ... more weights for other layers

  // ============================================================================
  // Function 1: inference_init
  // C signature: int inference_init(void** out_state);
  // ============================================================================
  func.func @inference_init(%out_state: !llvm.ptr<!llvm.ptr>) -> i32 {
    // See detailed design below
  }

  // ============================================================================
  // Function 2: inference_compute
  // C signature: int inference_compute(void* state, span_t inputs, span_t outputs);
  // ============================================================================
  func.func @inference_compute(%state: !llvm.ptr,
                                %inputs: !llvm.ptr,
                                %outputs: !llvm.ptr) -> i32 {
    // See detailed design below
  }

  // ============================================================================
  // Function 3: inference_cleanup
  // C signature: int inference_cleanup(void* state);
  // ============================================================================
  func.func @inference_cleanup(%state: !llvm.ptr) -> i32 {
    // See detailed design below
  }
}
```

**Key Points:**
- Exactly 3 public functions (exported from DLL)
- All ONNX operations are lowered **inline** within `inference_compute`
- No separate `@node_0_conv`, `@node_1_relu` functions
- Constants are global data, not code

---

## Lowering Pipeline

### Stage 1: ONNX-MLIR (Input from MorphiZen)

```mlir
module {
  func.func @main(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %weights = arith.constant dense<[...]> : tensor<64x3x3x3xf32>
    %bias = arith.constant dense<[...]> : tensor<64xf32>

    %0 = "onnx.Conv"(%arg0, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 1
    } : (tensor<1x3x224x224xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>)
        -> tensor<1x64x224x224xf32>

    %1 = "onnx.Relu"(%0) : (tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32>

    return %1 : tensor<1x64x224x224xf32>
  }
}
```

### Stage 2: After ONNX → HIP Lowering

```mlir
module {
  // Constants extracted to globals
  llvm.mlir.global constant @conv_weight(...) : !llvm.array<...>
  llvm.mlir.global constant @conv_bias(...) : !llvm.array<...>

  func.func @main(%state: !hip.context,  // NEW: state parameter added
                   %arg0: memref<1x3x224x224xf32>) -> memref<1x64x224x224xf32> {

    // ONNX operations replaced with HIP operations (inline)
    %0 = hip.conv(%state, %arg0, @conv_weight, @conv_bias) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 1
    } : (!hip.context, memref<...>, ...) -> memref<1x64x224x224xf32>

    %1 = hip.relu(%state, %0) : (!hip.context, memref<...>) -> memref<...>

    return %1 : memref<1x64x224x224xf32>
  }
}
```

**Key changes:**
- Operations: `onnx.Conv` → `hip.conv`, `onnx.Relu` → `hip.relu`
- Constants: Moved to `llvm.mlir.global`
- State: Added `%state: !hip.context` parameter
- Inline: All operations still in one function

### Stage 3: After HIP → LLVM Lowering

```mlir
module {
  // Constants remain as globals
  llvm.mlir.global constant @conv_weight(...) : !llvm.array<1728 x f32>
  llvm.mlir.global constant @conv_bias(...) : !llvm.array<64 x f32>

  // Main computation function signature changes to match C interface
  func.func @inference_compute(%state: !llvm.ptr,
                                %inputs: !llvm.ptr,
                                %outputs: !llvm.ptr) -> i32 {

    // Extract handles from state
    %miopen_ptr = llvm.getelementptr %state[0, 1] : (!llvm.ptr) -> !llvm.ptr
    %miopen = llvm.load %miopen_ptr : !llvm.ptr

    %stream_ptr = llvm.getelementptr %state[0, 0] : (!llvm.ptr) -> !llvm.ptr
    %stream = llvm.load %stream_ptr : !llvm.ptr

    // Extract weight GPU pointers from state (uploaded in init)
    %weights_ptr = llvm.getelementptr %state[0, 3, 0] : (!llvm.ptr) -> !llvm.ptr
    %weights_gpu = llvm.load %weights_ptr : !llvm.ptr

    %bias_ptr = llvm.getelementptr %state[0, 3, 1] : (!llvm.ptr) -> !llvm.ptr
    %bias_gpu = llvm.load %bias_ptr : !llvm.ptr

    // Parse input tensors from span_t (TODO: detailed design)
    // For now, assume %input_memref is parsed

    // HIP operations become direct LLVM calls to MIOpen (inline)
    %conv_out = llvm.call @miopenConvolutionForward(
      %miopen, %stream,
      %input_memref, %weights_gpu, %bias_gpu,
      %conv_desc, %algo, ...
    ) : (...) -> !llvm.ptr

    %relu_out = llvm.call @miopenActivationForward(
      %miopen, %stream,
      %conv_out, %relu_desc, ...
    ) : (...) -> !llvm.ptr

    // Write output to span_t (TODO: detailed design)

    %c0 = llvm.mlir.constant(0 : i32) : i32
    return %c0 : i32
  }
}
```

**Key changes:**
- Types: `!hip.context` → `!llvm.ptr`, `memref<...>` → LLVM pointers
- Operations: `hip.conv` → `llvm.call @miopenConvolutionForward`
- State access: Explicit `getelementptr` + `load` to extract handles/weights
- Signature: Matches C interface exactly

---

## HIP Context Type Design

### Type: `!hip.context` (Opaque)

The `!hip.context` type represents runtime execution state for HIP operations. It is designed as an **opaque type** following industry conventions (CUDA's `cudaStream_t`, HIP's `hipStream_t`, OpenCL's `cl_context`).

**Definition in HipDialect.td:**
```tablegen
def Hip_ContextType : DialectType<HipDialect, "context"> {
  let summary = "Opaque HIP execution context";
  let description = [{
    Represents runtime state including stream, library handles (MIOpen,
    hipBLASLT), and GPU memory. Lowered to !llvm.ptr in HIP→LLVM conversion.

    The context is an opaque pointer to a runtime-managed struct that contains:
    - HIP stream for asynchronous execution
    - Library handles (miopenHandle_t, hipblasLtHandle_t, etc.)
    - GPU pointers to uploaded weights
    - Cached descriptors and workspace buffers

    This type is intentionally opaque to separate high-level IR from low-level
    implementation details, enabling easy extension without IR changes.
  }];
}
```

**Design Rationale:**

1. **Separation of Concerns**
   - HIP dialect IR = high-level semantic operations
   - Context internals = low-level runtime implementation
   - IR doesn't encode "miopenHandle is at offset 8" - that's lowering detail

2. **Extensibility**
   - Adding new library support (e.g., rocFFT) only requires:
     - Updating runtime struct definition
     - Modifying HIP→LLVM lowering offset calculations
   - No changes to HIP dialect IR or operation definitions

3. **Type Safety Where It Matters**
   - Operations verify context type: `hip.conv` requires `!hip.context`
   - Can't accidentally pass wrong type
   - Lowering extracts correct library handle based on operation

4. **Industry Standard Pattern**
   - Matches CUDA, HIP, OpenCL conventions
   - Developers familiar with GPU programming recognize the pattern
   - Natural mapping to C API expectations

**Usage in IR:**
```mlir
// Context is first parameter of every HIP function
func @main_graph(%ctx: !hip.context, %input: tensor<...>) {
  // Passed to all operations
  %output = hip.conv(%ctx, %input, %weights, %bias) {...}
  %result = hip.gemm(%ctx, %output, %matrix) {...}
  return %result
}
```

**Lowering to LLVM:**
```mlir
// HIP dialect (before lowering)
%result = hip.conv(%ctx, %input, ...) : (!hip.context, ...) -> ...

// LLVM dialect (after lowering)
// %ctx is now !llvm.ptr, extract miopenHandle at offset 8
%miopen_ptr = llvm.getelementptr %ctx[0, 1] : (!llvm.ptr) -> !llvm.ptr
%miopen = llvm.load %miopen_ptr : !llvm.ptr
%result = llvm.call @miopenConvolutionForward(%miopen, ...) : ...
```

**Alternative Designs Considered (and rejected):**

- **Structured type** `!hip.context<{stream, miopen, hipblas, weights}>`: Too rigid, exposes implementation
- **Multiple parameters** `func(..., %stream, %miopen, %hipblas, ...)`: Doesn't scale, verbose
- **Global state**: Not thread-safe, harder to reason about

---

## State Structure Design

### State Struct Layout

The state is an opaque pointer to this struct:

```c
// Conceptual layout (not actual C code, but LLVM struct definition)
struct InferenceState {
  void* hip_stream;          // field 0: hipStream_t
  void* miopen_handle;       // field 1: miopenHandle_t
  void* hipblas_handle;      // field 2: hipblasLtHandle_t
  void* weight_pointers[N];  // field 3: array of GPU pointers to weights
  // ... potentially more fields (workspace buffers, algorithm caches, etc.)
};
```

**In MLIR (LLVM dialect):**

```mlir
// Define state type (private, not exposed to C)
!llvm.struct<(
  ptr,              // hip_stream
  ptr,              // miopen_handle
  ptr,              // hipblas_handle
  array<10 x ptr>   // weight_pointers (10 = number of weight tensors)
)>
```

### Accessing State Fields

```mlir
// Get miopenHandle_t from state
%miopen_ptr = llvm.getelementptr %state[0, 1] : (!llvm.ptr) -> !llvm.ptr
%miopen = llvm.load %miopen_ptr : !llvm.ptr

// Get weight GPU pointer (index 0 = first weight tensor)
%weights_array_ptr = llvm.getelementptr %state[0, 3] : (!llvm.ptr) -> !llvm.ptr
%weight_0_ptr = llvm.getelementptr %weights_array_ptr[0] : (!llvm.ptr) -> !llvm.ptr
%weight_0_gpu = llvm.load %weight_0_ptr : !llvm.ptr
```

---

## Detailed Function Designs

### Function 1: `inference_init`

**C Signature:**
```c
int inference_init(void** out_state);
```

**MLIR Implementation:**

```mlir
func.func @inference_init(%out_state: !llvm.ptr<!llvm.ptr>) -> i32 {
  // ============================================================================
  // Step 1: Allocate state struct
  // ============================================================================
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %state = llvm.alloca %c1 x !llvm.struct<(ptr, ptr, ptr, array<10 x ptr>)>
    : (i64) -> !llvm.ptr

  // ============================================================================
  // Step 2: Create GPU handles
  // ============================================================================

  // Create HIP stream
  %stream_ptr = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr
  %hip_success = llvm.call @hipStreamCreate(%stream_ptr) : (!llvm.ptr) -> i32
  // TODO: Check error code
  %stream = llvm.load %stream_ptr : !llvm.ptr

  // Create MIOpen handle
  %miopen_ptr = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr
  %miopen_success = llvm.call @miopenCreate(%miopen_ptr) : (!llvm.ptr) -> i32
  // TODO: Check error code
  %miopen = llvm.load %miopen_ptr : !llvm.ptr

  // Set stream for MIOpen
  llvm.call @miopenSetStream(%miopen, %stream) : (!llvm.ptr, !llvm.ptr) -> i32

  // Create hipBLAS handle
  %hipblas_ptr = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr
  %hipblas_success = llvm.call @hipblasLtCreate(%hipblas_ptr) : (!llvm.ptr) -> i32
  %hipblas = llvm.load %hipblas_ptr : !llvm.ptr

  // ============================================================================
  // Step 3: Upload constants (weights) to GPU
  // ============================================================================

  // Upload conv_weight
  %weight_cpu = llvm.mlir.addressof @conv_weight : !llvm.ptr
  %weight_size = llvm.mlir.constant(1728 : i64) : i64  // 64*3*3*3 floats
  %weight_bytes = llvm.mlir.constant(6912 : i64) : i64 // 1728 * 4 bytes

  %weight_gpu_ptr = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr
  llvm.call @hipMalloc(%weight_gpu_ptr, %weight_bytes) : (!llvm.ptr, i64) -> i32
  %weight_gpu = llvm.load %weight_gpu_ptr : !llvm.ptr

  %memcpy_kind = llvm.mlir.constant(1 : i32) : i32  // hipMemcpyHostToDevice
  llvm.call @hipMemcpy(%weight_gpu, %weight_cpu, %weight_bytes, %memcpy_kind)
    : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32

  // ... repeat for other weights (conv_bias, etc.)

  // ============================================================================
  // Step 4: Store everything in state struct
  // ============================================================================

  // Store hip_stream (field 0)
  %stream_field_ptr = llvm.getelementptr %state[0, 0] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %stream, %stream_field_ptr : !llvm.ptr

  // Store miopen_handle (field 1)
  %miopen_field_ptr = llvm.getelementptr %state[0, 1] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %miopen, %miopen_field_ptr : !llvm.ptr

  // Store hipblas_handle (field 2)
  %hipblas_field_ptr = llvm.getelementptr %state[0, 2] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %hipblas, %hipblas_field_ptr : !llvm.ptr

  // Store weight GPU pointers (field 3)
  %weights_array_ptr = llvm.getelementptr %state[0, 3] : (!llvm.ptr) -> !llvm.ptr

  %weight_0_slot = llvm.getelementptr %weights_array_ptr[0] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %weight_gpu, %weight_0_slot : !llvm.ptr

  // ... store other weight pointers

  // ============================================================================
  // Step 5: Return state pointer via out parameter
  // ============================================================================
  llvm.store %state, %out_state : !llvm.ptr<!llvm.ptr>

  // Return success
  %c0 = llvm.mlir.constant(0 : i32) : i32
  return %c0 : i32
}
```

### Function 2: `inference_compute`

**C Signature:**
```c
int inference_compute(void* state, span_t inputs, span_t outputs);
```

**MLIR Implementation:**

```mlir
func.func @inference_compute(%state: !llvm.ptr,
                              %inputs: !llvm.ptr,   // span_t* (pointer to span)
                              %outputs: !llvm.ptr)  // span_t* (pointer to span)
                              -> i32 {

  // ============================================================================
  // Step 1: Extract handles from state
  // ============================================================================
  %miopen_ptr = llvm.getelementptr %state[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %miopen = llvm.load %miopen_ptr : !llvm.ptr

  %stream_ptr = llvm.getelementptr %state[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %stream = llvm.load %stream_ptr : !llvm.ptr

  // ============================================================================
  // Step 2: Extract weight GPU pointers from state
  // ============================================================================
  %weights_array_ptr = llvm.getelementptr %state[0, 3] : (!llvm.ptr) -> !llvm.ptr

  %weight_0_ptr = llvm.getelementptr %weights_array_ptr[0] : (!llvm.ptr) -> !llvm.ptr
  %weight_0_gpu = llvm.load %weight_0_ptr : !llvm.ptr

  %weight_1_ptr = llvm.getelementptr %weights_array_ptr[1] : (!llvm.ptr) -> !llvm.ptr
  %weight_1_gpu = llvm.load %weight_1_ptr : !llvm.ptr

  // ============================================================================
  // Step 3: Parse input tensors from span_t
  // ============================================================================
  // TODO: Detailed design of span_t parsing
  // For now, simplified version:

  // inputs is span_t* -> {void* data, size_t count}
  // inputs.data points to array of tensor_t
  %inputs_data_ptr = llvm.getelementptr %inputs[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %inputs_data = llvm.load %inputs_data_ptr : !llvm.ptr  // tensor_t*

  // Get first tensor: inputs.data[0]
  %input_tensor_0 = llvm.getelementptr %inputs_data[0] : (!llvm.ptr) -> !llvm.ptr

  // Extract tensor.data pointer
  %input_data_ptr = llvm.getelementptr %input_tensor_0[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %input_data = llvm.load %input_data_ptr : !llvm.ptr

  // TODO: Extract shape, rank, validate against expected

  // ============================================================================
  // Step 4: Execute operations (inline, one after another)
  // ============================================================================

  // Operation 1: Convolution (originally onnx.Conv)
  // Create descriptors (TODO: cache in state for efficiency)
  %conv_desc = llvm.call @miopenCreateConvolutionDescriptor(...) : ... -> !llvm.ptr

  // Call MIOpen convolution
  %conv_out = llvm.call @miopenConvolutionForward(
    %miopen,
    %alpha_ptr,          // scaling factor (1.0)
    %input_desc,         // input tensor descriptor
    %input_data,         // input data pointer
    %weight_desc,        // weight tensor descriptor
    %weight_0_gpu,       // weight GPU pointer from state
    %conv_desc,          // convolution descriptor
    %conv_algo,          // algorithm (TODO: find best)
    %beta_ptr,           // beta (0.0)
    %output_desc,        // output tensor descriptor
    %conv_output_ptr,    // output buffer
    %workspace,          // workspace
    %workspace_size      // workspace size
  ) : (...) -> i32

  // Operation 2: ReLU (originally onnx.Relu)
  %relu_desc = llvm.call @miopenCreateActivationDescriptor(...) : ... -> !llvm.ptr

  %relu_out = llvm.call @miopenActivationForward(
    %miopen,
    %relu_desc,
    %alpha_ptr,
    %conv_output_desc,
    %conv_output_ptr,
    %beta_ptr,
    %relu_output_desc,
    %relu_output_ptr
  ) : (...) -> i32

  // ... more operations inline ...

  // ============================================================================
  // Step 5: Write outputs to span_t
  // ============================================================================
  // TODO: Detailed design of span_t writing
  // For now, simplified:

  %outputs_data_ptr = llvm.getelementptr %outputs[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %outputs_data = llvm.load %outputs_data_ptr : !llvm.ptr

  %output_tensor_0 = llvm.getelementptr %outputs_data[0] : (!llvm.ptr) -> !llvm.ptr
  %output_data_field = llvm.getelementptr %output_tensor_0[0, 0] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %relu_output_ptr, %output_data_field : !llvm.ptr

  // Return success
  %c0 = llvm.mlir.constant(0 : i32) : i32
  return %c0 : i32
}
```

**Key Design Note:**
- All ONNX operations are **lowered inline** within this function
- No separate `@node_conv`, `@node_relu` functions
- State is passed as parameter, extracted once at the top

### Function 3: `inference_cleanup`

**C Signature:**
```c
int inference_cleanup(void* state);
```

**MLIR Implementation:**

```mlir
func.func @inference_cleanup(%state: !llvm.ptr) -> i32 {
  // ============================================================================
  // Step 1: Free GPU weight memory
  // ============================================================================
  %weights_array_ptr = llvm.getelementptr %state[0, 3] : (!llvm.ptr) -> !llvm.ptr

  // Free weight 0
  %weight_0_ptr = llvm.getelementptr %weights_array_ptr[0] : (!llvm.ptr) -> !llvm.ptr
  %weight_0_gpu = llvm.load %weight_0_ptr : !llvm.ptr
  llvm.call @hipFree(%weight_0_gpu) : (!llvm.ptr) -> i32

  // ... free other weights

  // ============================================================================
  // Step 2: Destroy handles
  // ============================================================================

  // Destroy hipBLAS handle
  %hipblas_ptr = llvm.getelementptr %state[0, 2] : (!llvm.ptr) -> !llvm.ptr
  %hipblas = llvm.load %hipblas_ptr : !llvm.ptr
  llvm.call @hipblasLtDestroy(%hipblas) : (!llvm.ptr) -> i32

  // Destroy MIOpen handle
  %miopen_ptr = llvm.getelementptr %state[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %miopen = llvm.load %miopen_ptr : !llvm.ptr
  llvm.call @miopenDestroy(%miopen) : (!llvm.ptr) -> i32

  // Destroy HIP stream
  %stream_ptr = llvm.getelementptr %state[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %stream = llvm.load %stream_ptr : !llvm.ptr
  llvm.call @hipStreamDestroy(%stream) : (!llvm.ptr) -> i32

  // ============================================================================
  // Step 3: Free state struct (TODO: should we? it's stack-allocated in init)
  // ============================================================================
  // NOTE: Current design allocates state on stack (llvm.alloca in init)
  // This is a design issue - state should be heap-allocated!
  // TODO Phase 2: Use hipMallocHost or malloc for state

  // Return success
  %c0 = llvm.mlir.constant(0 : i32) : i32
  return %c0 : i32
}
```

---

## TODO Items for Phase 2

### High-Level Type System (Future Refinement)

**Current (Phase 1):** Work directly in LLVM dialect
- ✅ Pro: Simple, direct mapping to C
- ❌ Con: Verbose, lots of getelementptr

**Future (Phase 2):** Define high-level types and operations
```mlir
// New types in HipTypes.td
def Hip_StateType : HipType<"State", "state">;
def Hip_MiopenHandleType : HipType<"MiopenHandle", "miopen_handle">;
def Hip_StreamType : HipType<"Stream", "stream">;

// New operations in HipOps.td
def Hip_StateCreateOp : Hip_Op<"state.create"> {
  let results = (outs Hip_StateType:$state);
}

def Hip_StateGetMiopenOp : Hip_Op<"state.get_miopen"> {
  let arguments = (ins Hip_StateType:$state);
  let results = (outs Hip_MiopenHandleType:$handle);
}

def Hip_StateGetWeightOp : Hip_Op<"state.get_weight"> {
  let arguments = (ins Hip_StateType:$state, I64Attr:$index);
  let results = (outs AnyMemRef:$weight);
}

// Usage in MLIR (cleaner!)
%state = hip.state.create() : !hip.state
%miopen = hip.state.get_miopen(%state) : (!hip.state) -> !hip.miopen_handle
%weight = hip.state.get_weight(%state) {index = 0} : (!hip.state) -> memref<...>
```

**Benefits:**
- ✅ Cleaner, more readable MLIR
- ✅ Type safety (can't pass wrong handle type)
- ✅ Easier pattern matching and transformations
- ✅ Better error messages

**Downside:**
- ❌ More work (define types, operations, lowering passes)

**Decision:** Start with LLVM dialect (Phase 1), refine with high-level types later (Phase 2)

### State Memory Management (Current Issue)

**Problem:** Current design uses `llvm.alloca` for state in `inference_init`:
```mlir
%state = llvm.alloca %c1 x !llvm.struct<...> : (i64) -> !llvm.ptr
```

This allocates on the **stack**, which is **destroyed when `inference_init` returns**!

**Solution (TODO):** Heap-allocate state:
```mlir
func.func @inference_init(%out_state: !llvm.ptr<!llvm.ptr>) -> i32 {
  // Allocate state on heap
  %state_size = llvm.mlir.constant(256 : i64) : i64  // sizeof(InferenceState)
  %state = llvm.call @malloc(%state_size) : (i64) -> !llvm.ptr

  // ... initialize ...

  llvm.store %state, %out_state : !llvm.ptr<!llvm.ptr>
  return %c0 : i32
}

func.func @inference_cleanup(%state: !llvm.ptr) -> i32 {
  // ... cleanup resources ...

  // Free state memory
  llvm.call @free(%state) : (!llvm.ptr) -> ()

  return %c0 : i32
}
```

**Alternative:** Use `hipMallocHost` for pinned memory (better performance for CPU-GPU transfers)

### Span and Tensor Parsing (TODO)

**Current:** Simplified pseudocode in `inference_compute`

**Needed:** Full implementation of:
1. Parse `span_t` struct to get `tensor_t*` array
2. For each `tensor_t`, extract `data`, `shape`, `rank`
3. Validate shape/rank against expected (compile-time known)
4. Handle dynamic shapes if needed

**Implementation Strategy:**
- Generate parsing code in LLVM dialect
- Inline for simplicity (no separate parsing function)
- Add assertions for shape validation

### Descriptor Caching (Performance)

**Current:** Create descriptors on every `inference_compute` call:
```mlir
%conv_desc = llvm.call @miopenCreateConvolutionDescriptor(...) : ...
```

**Problem:** Descriptor creation has overhead (~1-5ms)

**Solution (TODO):** Cache descriptors in state:
```c
struct InferenceState {
  void* hip_stream;
  void* miopen_handle;
  void* hipblas_handle;
  void* weight_pointers[N];
  void* descriptors[M];  // NEW: cached descriptors
};
```

Create in `inference_init`, reuse in `inference_compute`, destroy in `inference_cleanup`.

### Workspace Memory (TODO)

MIOpen operations need workspace memory. Current options:

**Option A:** Allocate per-operation (simple, but slow)
```mlir
%workspace = llvm.call @hipMalloc(%workspace_size) : ...
%result = llvm.call @miopenConvolutionForward(..., %workspace, ...) : ...
llvm.call @hipFree(%workspace) : ...
```

**Option B:** Pre-allocate workspace in `inference_init` (faster)
```c
struct InferenceState {
  // ...
  void* workspace;      // Pre-allocated workspace
  size_t workspace_size; // Size of workspace
};
```

**Recommendation:** Use Option B (pre-allocate). Need to:
1. Determine max workspace size needed across all operations
2. Allocate in `inference_init`
3. Reuse in all operations
4. Free in `inference_cleanup`

---

## Compilation to DLL

### LLVM IR Generation

After all MLIR lowering passes, translate LLVM dialect to LLVM IR:

```cpp
// In Level-1 Pass
mlir::registerLLVMDialectTranslation(*context);
auto llvmModule = mlir::translateModuleToLLVMIR(mlirModule, llvmContext);
```

**Output:** LLVM IR (`.ll` file or in-memory)

### Native Code Generation

Use LLVM backend to compile to native code:

```cpp
// Set target triple (e.g., x86_64-pc-windows-msvc)
llvmModule->setTargetTriple(TargetTriple.normalize());

// Set data layout
llvmModule->setDataLayout(targetMachine->createDataLayout());

// Emit object file
llvm::SmallString<128> objPath;
llvm::sys::fs::createTemporaryFile("inference", "obj", objPath);

std::error_code EC;
llvm::raw_fd_ostream dest(objPath, EC);

llvm::legacy::PassManager pass;
targetMachine->addPassesToEmitFile(pass, dest, nullptr,
                                    llvm::CGFT_ObjectFile);
pass.run(*llvmModule);
dest.flush();
```

**Output:** Object file (`.obj` or `.o`)

### Linking to DLL

Link object file with HIP/MIOpen/hipBLAS libraries:

```bash
# Windows (MSVC)
link.exe /DLL /OUT:inference.dll inference.obj \
  hip.lib miopen.lib hipblaslt.lib

# Linux (GCC)
gcc -shared -o inference.so inference.o \
  -lhip -lmiopen -lhipblaslt
```

**Output:** DLL (`.dll` on Windows, `.so` on Linux)

**Exported symbols:**
- `inference_init`
- `inference_compute`
- `inference_cleanup`

### Embedding in EPContext

CustomOp loads DLL bytes from EPContext and uses MemoryModule to load without disk I/O (see ARCHITECTURE.md for details).

---

## Summary

**MLIR Module Structure:**
- 3 functions: `init`, `compute`, `cleanup`
- Constants: `llvm.mlir.global` in `.data` section
- Inline lowering: All operations within `inference_compute`

**State Design:**
- Opaque pointer to struct
- Contains: handles (stream, miopen, hipblas) + weight GPU pointers
- Heap-allocated in `init`, freed in `cleanup`

**Lowering Pipeline:**
- ONNX-MLIR → HIP dialect → LLVM dialect → LLVM IR → DLL
- Each stage preserves inline structure (no function explosion)

**Key Design Principles:**
1. **Simplicity:** Start with LLVM dialect directly (no high-level types yet)
2. **Explicit state:** Pass as parameter, no globals
3. **Inline operations:** One function for all computation
4. **C compatibility:** MLIR signatures match C interface exactly

**Phase 2 TODOs:**
- Heap-allocate state (fix current stack allocation bug)
- Implement span_t/tensor_t parsing
- Add descriptor caching for performance
- Pre-allocate workspace memory
- Refine with high-level types (!hip.state, etc.)

---

**Related Documents:**
- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system architecture
- [ONNX-MLIR-INTEGRATION.md](ONNX-MLIR-INTEGRATION.md) - How onnx-mlir is integrated
