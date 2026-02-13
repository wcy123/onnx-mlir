<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Dynamic Shape Support Design

**Note:** This is the authoritative source for dynamic shape design. Other documents reference this for details.

**Date:** 2026-02-10
**Document Type:** Design
**Review Status:** Draft
**Related:** [ARCHITECTURE.md](ARCHITECTURE.md), [MLIR-COMPILATION-DESIGN.md](MLIR-COMPILATION-DESIGN.md)

---

## Overview

This document explains how the MLIR-based AOT compilation pipeline supports dynamic shapes (tensors with runtime-determined dimensions) while maintaining efficient execution and clean architecture.

**Key Design Principle:** The C interface (`tensor_t` with `shape` pointer) already provides runtime shape information. All layers of the system simply propagate and use this information without requiring interface changes.

---

## The Challenge: Static Types vs Dynamic Values

### MLIR Type System

MLIR's type system distinguishes between:
- **Rank** (number of dimensions) - compile-time constant
- **Dimension values** (size of each dimension) - can be static or dynamic

**Static shape:**
```mlir
memref<1x3x224x224xf32, 1>  // All dimensions known at compile time
```

**Dynamic shape:**
```mlir
memref<?x?x224x224xf32, 1>  // First two dimensions unknown (? = dynamic)
```

**Fully dynamic:**
```mlir
memref<?x?x?x?xf32, 1>  // All dimensions dynamic
```

### Lowering to LLVM

Both static and dynamic shapes lower to the **same struct type** (determined by rank):

```mlir
// 4D tensor (rank=4) - regardless of which dimensions are static/dynamic
!llvm.struct<(
  ptr<1>,           // allocated_ptr
  ptr<1>,           // aligned_ptr
  i64,              // offset
  array<4 x i64>,   // sizes[4] - ACTUAL dimension values
  array<4 x i64>    // strides[4]
)>
```

The difference is **how the sizes array is populated**:
- Static: `llvm.insertvalue %c224, ...` (constant)
- Dynamic: `llvm.insertvalue %runtime_value, ...` (variable)

---

## Interface Design: Already Dynamic-Ready

### C Interface (From ARCHITECTURE.md)

```c
typedef struct {
    void* data;          // Pointer to tensor data (CPU or GPU)
    int64_t* shape;      // Runtime dimension values
    int rank;            // Number of dimensions
} tensor_t;

typedef struct {
    void* data;          // Pointer to tensor_t array
    size_t count;        // Number of tensors
} span_t;

int inference_compute(void* state, span_t* inputs, span_t* outputs);
```

**Key observation:** The interface **already provides runtime shapes** via `tensor_t.shape` pointer!

This design choice was intentional:
- CustomOp can pass static shapes (constant array)
- CustomOp can pass dynamic shapes (runtime-computed array)
- `inference_compute` treats both cases identically

---

## How Each Layer Handles Dynamic Shapes

### Layer 1: CustomOp (Caller)

**Static shape example:**
```cpp
void MyCustomOp::Compute(OrtKernelContext* context) {
  // Shape is compile-time constant
  int64_t input_shape[] = {1, 3, 224, 224};

  tensor_t input = {
    .data = GetInputPointer(context, 0),
    .shape = input_shape,
    .rank = 4
  };

  span_t inputs = {.data = &input, .count = 1};
  inference_compute(state_, inputs, outputs);
}
```

**Dynamic shape example:**
```cpp
void MyCustomOp::Compute(OrtKernelContext* context) {
  // Shape determined at runtime
  int64_t batch = GetInputDim(context, 0, 0);  // Could be 1, 8, 16, etc.
  int64_t channels = GetInputDim(context, 0, 1);
  int64_t input_shape[] = {batch, channels, 224, 224};

  tensor_t input = {
    .data = GetInputPointer(context, 0),
    .shape = input_shape,  // Runtime values
    .rank = 4
  };

  span_t inputs = {.data = &input, .count = 1};
  inference_compute(state_, inputs, outputs);  // Same call!
}
```

**No interface change needed** - the same function call handles both cases.

### Layer 2: inference_compute Wrapper (GenerateInterfacePass)

```mlir
llvm.func @inference_compute(%state: !llvm.ptr,
                              %inputs: !llvm.ptr,   // span_t*
                              %outputs: !llvm.ptr)  // span_t*
                              -> i32 {

  // ============================================================================
  // Parse span_t to get tensor_t
  // ============================================================================
  %inputs_data_ptr = llvm.getelementptr %inputs[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %inputs_data = llvm.load %inputs_data_ptr : !llvm.ptr  // tensor_t* array

  %input_tensor_0 = llvm.getelementptr %inputs_data[0] : (!llvm.ptr) -> !llvm.ptr

  // ============================================================================
  // Extract tensor_t fields
  // ============================================================================
  // tensor_t layout: {void* data, int64_t* shape, int rank}

  // Get data pointer
  %data_field_ptr = llvm.getelementptr %input_tensor_0[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %data = llvm.load %data_field_ptr : !llvm.ptr

  // Get shape pointer
  %shape_field_ptr = llvm.getelementptr %input_tensor_0[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %shape = llvm.load %shape_field_ptr : !llvm.ptr  // int64_t* array

  // Get rank
  %rank_field_ptr = llvm.getelementptr %input_tensor_0[0, 2] : (!llvm.ptr) -> !llvm.ptr
  %rank = llvm.load %rank_field_ptr : i32

  // ============================================================================
  // Build memref descriptor from runtime shape
  // ============================================================================
  // Read dimension values from shape array (works for both static and dynamic!)
  %shape_0_ptr = llvm.getelementptr %shape[0] : (!llvm.ptr) -> !llvm.ptr
  %size_0 = llvm.load %shape_0_ptr : i64  // Load actual value

  %shape_1_ptr = llvm.getelementptr %shape[1] : (!llvm.ptr) -> !llvm.ptr
  %size_1 = llvm.load %shape_1_ptr : i64  // Load actual value

  %shape_2_ptr = llvm.getelementptr %shape[2] : (!llvm.ptr) -> !llvm.ptr
  %size_2 = llvm.load %shape_2_ptr : i64

  %shape_3_ptr = llvm.getelementptr %shape[3] : (!llvm.ptr) -> !llvm.ptr
  %size_3 = llvm.load %shape_3_ptr : i64

  // Compute strides (row-major): stride[i] = product(shape[i+1:])
  %stride_3 = llvm.mlir.constant(1 : i64) : i64
  %stride_2 = llvm.mul %size_3, %stride_3 : i64
  %stride_1 = llvm.mul %size_2, %stride_2 : i64
  %stride_0 = llvm.mul %size_1, %stride_1 : i64

  // Build memref struct (4D)
  %desc = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>

  %data_gpu = llvm.addrspacecast %data : !llvm.ptr to !llvm.ptr<1>
  %desc = llvm.insertvalue %data_gpu, %desc[0] : !llvm.struct<...>  // allocated_ptr
  %desc = llvm.insertvalue %data_gpu, %desc[1] : !llvm.struct<...>  // aligned_ptr

  %offset = llvm.mlir.constant(0 : i64) : i64
  %desc = llvm.insertvalue %offset, %desc[2] : !llvm.struct<...>

  // Insert sizes (could be constants or runtime values - doesn't matter!)
  %desc = llvm.insertvalue %size_0, %desc[3, 0] : !llvm.struct<...>
  %desc = llvm.insertvalue %size_1, %desc[3, 1] : !llvm.struct<...>
  %desc = llvm.insertvalue %size_2, %desc[3, 2] : !llvm.struct<...>
  %desc = llvm.insertvalue %size_3, %desc[3, 3] : !llvm.struct<...>

  // Insert strides
  %desc = llvm.insertvalue %stride_0, %desc[4, 0] : !llvm.struct<...>
  %desc = llvm.insertvalue %stride_1, %desc[4, 1] : !llvm.struct<...>
  %desc = llvm.insertvalue %stride_2, %desc[4, 2] : !llvm.struct<...>
  %desc = llvm.insertvalue %stride_3, %desc[4, 3] : !llvm.struct<...>

  // ============================================================================
  // Call @main with memref struct (same for static or dynamic shapes!)
  // ============================================================================
  %ret = llvm.call @main(%state, %desc, %output_desc) : (...) -> i32

  llvm.return %ret : i32
}
```

**Key point:** Whether `%size_0` is a constant (from static shape) or a variable (from dynamic shape), the code is **identical**. We just load from the `shape` pointer.

### Layer 3: @main Function

```mlir
llvm.func @main(%ctx: !llvm.ptr,
                %input: !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>,
                %output: !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>)
                -> i32
  attributes {
    inference.num_inputs = 1 : i64,
    inference.num_outputs = 1 : i64
  } {

  // Get constants
  %weights = [hip.get_constant(...)]
  %bias = [hip.get_constant(...)]

  // Allocate temporary buffer
  // ... (build temp memref struct)

  // Call wrapper - passes memref structs (contain actual dimension values)
  %ret = llvm.call @hip_conv_wrapper(
    %ctx, %input, %weights, %bias, %temp
  ) : (...) -> i32

  llvm.return %ret : i32
}
```

**@main doesn't care about static vs dynamic** - it just passes memref structs to wrapper functions.

### Layer 4: Wrapper Functions (HipToLLVM Pass)

```mlir
llvm.func @hip_conv_wrapper(
    %ctx: !llvm.ptr,
    %input: !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>,
    %weights: !llvm.struct<...>,
    %bias: !llvm.struct<...>,
    %output: !llvm.struct<...>,
    %kernel_h: i64, %kernel_w: i64,
    %stride_h: i64, %stride_w: i64,
    ...) -> i32 {

  // ============================================================================
  // Extract dimension values from memref struct
  // ============================================================================
  %input_ptr = llvm.extractvalue %input[1] : !llvm.struct<...>

  // These could be constants (static) or variables (dynamic) - doesn't matter!
  %input_n = llvm.extractvalue %input[3, 0] : !llvm.struct<...>  // sizes[0]
  %input_c = llvm.extractvalue %input[3, 1] : !llvm.struct<...>  // sizes[1]
  %input_h = llvm.extractvalue %input[3, 2] : !llvm.struct<...>  // sizes[2]
  %input_w = llvm.extractvalue %input[3, 3] : !llvm.struct<...>  // sizes[3]

  // ============================================================================
  // Create MIOpen tensor descriptor with ACTUAL dimension values
  // ============================================================================
  %xDesc_ptr = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr
  llvm.call @miopenCreateTensorDescriptor(%xDesc_ptr) : (!llvm.ptr) -> i32
  %xDesc = llvm.load %xDesc_ptr : !llvm.ptr

  %dataType = llvm.mlir.constant(0 : i32) : i32  // miopenFloat
  llvm.call @miopenSet4dTensorDescriptor(
    %xDesc, %dataType,
    %input_n,  // Could be constant(1) or runtime variable
    %input_c,  // Could be constant(3) or runtime variable
    %input_h,  // Could be constant(224) or runtime variable
    %input_w   // Could be constant(224) or runtime variable
  ) : (!llvm.ptr, i32, i64, i64, i64, i64) -> i32

  // Similar for weights, output descriptors...

  // ============================================================================
  // Call MIOpen with runtime dimension values
  // ============================================================================
  %ret = llvm.call @miopenConvolutionForward(
    %miopen, %alpha, %xDesc, %input_ptr,
    %wDesc, %weights_ptr, %convDesc, %algo,
    %beta, %yDesc, %output_ptr, %workspace, %workspace_size
  ) : (...) -> i32

  // Cleanup descriptors
  llvm.call @miopenDestroyTensorDescriptor(%xDesc) : (!llvm.ptr) -> i32
  // ...

  llvm.return %ret : i32
}
```

**The wrapper is shape-agnostic!** It:
1. Extracts dimension values from memref struct (SSA values)
2. Passes them to MIOpen

Whether those values are compile-time constants or runtime variables is **irrelevant** - MIOpen just gets `i64` values either way.

---

## Concrete Examples

### Example 1: Static Shape ResNet50

**ONNX Model:**
```
Input: tensor<1x3x224x224xf32>  // Fixed batch=1
Output: tensor<1x1000xf32>
```

**CustomOp provides:**
```cpp
int64_t input_shape[] = {1, 3, 224, 224};  // Constants
tensor_t input = {data, input_shape, 4};
```

**What happens:**
- `inference_compute` loads: `%size_0 = load [1]`, `%size_1 = load [3]`, etc.
- LLVM optimizer sees these are constants from a global array
- **Optimization:** LLVM inlines the constants, MIOpen descriptors created with constant dimensions
- **Result:** Nearly identical performance to pure static compilation

### Example 2: Dynamic Batch Size

**ONNX Model:**
```
Input: tensor<?x3x224x224xf32>  // Batch size varies
Output: tensor<?x1000xf32>
```

**CustomOp provides (batch=8 this time):**
```cpp
int64_t batch = GetBatchSize();  // Runtime: 8
int64_t input_shape[] = {batch, 3, 224, 224};
tensor_t input = {data, input_shape, 4};
```

**What happens:**
- `inference_compute` loads: `%size_0 = load [8]`, `%size_1 = load [3]`, etc.
- These are runtime values (LLVM can't optimize away)
- MIOpen descriptors created with runtime batch size
- **Result:** Slightly slower descriptor creation (~microseconds), but convolution performance identical

### Example 3: Fully Dynamic (Variable Sequence Length)

**ONNX Model:**
```
Input: tensor<?x?x?x?xf32>  // NLP model with variable sequence length
```

**CustomOp provides:**
```cpp
int64_t shape[] = {batch, seq_len, hidden_dim, num_heads};  // All runtime
tensor_t input = {data, shape, 4};
```

**What happens:**
- All dimensions loaded at runtime
- MIOpen descriptors created with full runtime dimensions
- **Performance:** Descriptor creation overhead (~1-2 microseconds per operation), negligible compared to GPU compute time

---

## Performance Implications

### Static Shape Advantages

**Compile-time optimizations:**
```mlir
// Static: LLVM can constant-fold
%size = llvm.mlir.constant(224 : i64) : i64
%stride = llvm.mul %size, %size : i64  // LLVM computes: 224*224 = 50176 at compile time
```

**LLVM optimizations:**
- Constant propagation through MIOpen descriptor creation
- Dead code elimination for unused dimensions
- Loop unrolling based on known sizes

**Estimated overhead:** ~0 (compiled away)

### Dynamic Shape Overhead

**Runtime operations:**
```mlir
// Dynamic: LLVM must emit actual loads and multiplies
%size = llvm.load %shape_ptr : i64     // Runtime load
%stride = llvm.mul %size, %other : i64  // Runtime multiply
```

**Estimated overhead per tensor:**
- Loads: ~4 loads × 1-2 cycles = 4-8 cycles
- Stride computation: ~3 multiplies × 1-3 cycles = 3-9 cycles
- MIOpen descriptor creation: ~500-1000 cycles (dominates)

**Total overhead:** ~1-2 microseconds per operation

**Context:** GPU convolution takes milliseconds
- ResNet50 conv layer: ~500 microseconds
- Dynamic shape overhead: ~1 microsecond (~0.2%)
- **Negligible impact on end-to-end performance**

---

## Design Decisions

### Decision 1: Support Dynamic Shapes from Day 1

**Rationale:**
- C interface already provides runtime shapes (no redesign needed)
- Implementation cost is low (just load from shape pointer instead of constants)
- Handles all use cases (static shapes work as special case of dynamic)
- More future-proof (users expect dynamic shapes)

**Trade-off:**
- Slightly more complex code generation (but not significantly)
- Negligible runtime overhead (~0.2% on GPU-bound workloads)

**Conclusion:** ✅ Support dynamic shapes in Phase 1

### Decision 2: Memref Struct Type Determined by Rank Only

**Rationale:**
- MLIR's type system requires compile-time rank
- `!llvm.struct<(..., array<4 x i64>, ...)>` vs `!llvm.struct<(..., array<2 x i64>, ...)>` are different types
- Cannot have fully rank-polymorphic `@main` in LLVM dialect

**Implication:**
- `@main` signature must be generated for specific tensor ranks
- ONNX→HIP pass must know input/output ranks when generating `@main`

**Metadata needed:**
```mlir
llvm.func @main(...) attributes {
  inference.num_inputs = 1 : i64,
  inference.num_outputs = 1 : i64,
  inference.input_ranks = [4] : i64,   // Input is 4D
  inference.output_ranks = [2] : i64   // Output is 2D
}
```

### Decision 3: Strides Computed in inference_compute

**Rationale:**
- Row-major layout is standard (stride[i] = product(shape[i+1:]))
- Simple formula, easily generated in MLIR
- No need to require CustomOp to provide strides

**Alternative considered:** CustomOp computes and provides strides
- ❌ Rejected: More burden on CustomOp, more error-prone

### Decision 4: No Shape Validation in inference_compute

**Rationale:**
- CustomOp already validated shapes (ONNX Runtime requirement)
- MIOpen will fail with clear error if shapes are incompatible
- Extra validation adds overhead without benefit

**Alternative considered:** Assert shapes match expected ranges
- ❌ Rejected: Redundant validation, performance cost

---

## Implementation Checklist

### ONNX→HIP Pass

- [ ] Generate `@main` with rank-specific memref struct types
- [ ] Add function attributes: `inference.num_inputs`, `inference.num_outputs`, `inference.input_ranks`, `inference.output_ranks`
- [ ] Handle multiple inputs/outputs with correct ranks

### HIP→LLVM Pass

- [ ] Lower `@main` memrefs to struct-by-value (not unpacked)
- [ ] Generate wrapper functions that extract dimension values from memref structs
- [ ] Wrapper functions pass dimension values to MIOpen/hipBLAS

### GenerateInterfacePass

- [ ] Parse `@main` attributes to discover number of inputs/outputs and their ranks
- [ ] Generate `inference_compute` that:
  - [ ] Parses `span_t` to get `tensor_t` array
  - [ ] For each tensor, loads dimensions from `tensor_t.shape`
  - [ ] Computes strides from dimensions (row-major)
  - [ ] Builds memref struct with runtime dimensions
  - [ ] Calls `@main` with memref structs
- [ ] Handle multiple inputs/outputs

### Testing

- [ ] Test with static shapes (verify LLVM optimizes constants)
- [ ] Test with dynamic batch size (verify correctness)
- [ ] Test with fully dynamic shapes (verify correctness)
- [ ] Benchmark overhead (should be <1% of total inference time)

---

## Future Enhancements

### Optimization: Shape Caching

If the same shape is used repeatedly, cache the MIOpen descriptors in state:

```c
struct InferenceState {
  void* hip_stream;
  void* miopen_handle;
  void* hipblas_handle;
  void* weight_pointers[N];

  // NEW: Shape cache
  struct {
    int64_t last_input_shape[4];
    miopenTensorDescriptor_t input_desc;
    miopenTensorDescriptor_t output_desc;
  } descriptor_cache;
};
```

Wrapper function checks cache:
```mlir
// Check if shape changed
%same = compare %input_n with %cached_n, %input_c with %cached_c, ...
if %same:
  %xDesc = load cached descriptor
else:
  create new descriptor
  update cache
```

**Benefit:** Amortizes descriptor creation cost for static or slowly-changing shapes.

### Optimization: Compile-Time Shape Specialization

For common shapes (e.g., batch=1), generate specialized `@main_batch1` function:

```mlir
// Generic (dynamic batch)
llvm.func @main(%ctx, %input, %output) -> i32

// Specialized (batch=1, fully static)
llvm.func @main_batch1(%ctx, %input, %output) -> i32
  // All dimensions hardcoded, LLVM can fully optimize
```

Runtime dispatcher chooses specialized version when shapes match:
```mlir
llvm.func @inference_compute(...) {
  %batch = extract batch from tensor_t
  if %batch == 1:
    call @main_batch1(...)
  else:
    call @main(...)
}
```

**Benefit:** Best-of-both-worlds - static performance for common cases, dynamic flexibility for others.

---

## Summary

**Dynamic shape support is achieved through:**
1. ✅ C interface provides runtime shapes (`tensor_t.shape`)
2. ✅ `inference_compute` loads dimensions from shape pointer
3. ✅ Memref structs built with runtime dimensions
4. ✅ Wrapper functions extract dimensions and pass to MIOpen
5. ✅ MIOpen handles runtime dimensions natively

**No interface changes needed** - the design was already dynamic-ready from the start.

**Performance impact:** Negligible (<1% overhead for dynamic shapes vs static shapes)

**Implementation complexity:** Low - just load from pointer instead of using constants.

---

**Related Documents:**
- [ARCHITECTURE.md](ARCHITECTURE.md) - C interface design
- [MLIR-COMPILATION-DESIGN.md](MLIR-COMPILATION-DESIGN.md) - Wrapper functions, two-layer architecture
- [RUNTIME-ARCHITECTURE.md](RUNTIME-ARCHITECTURE.md) - Runtime state struct layout
