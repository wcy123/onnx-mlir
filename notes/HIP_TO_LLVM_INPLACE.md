# HIP→LLVM Lowering for In-Place Operations

**Date:** 2026-02-09
**Status:** Implemented (awaiting build verification)

---

## Summary

Updated `lib/HipDialect/HipToLLVM.cpp` to lower `hip.conv` operation (in-place semantics) to LLVM dialect with MIOpen runtime call.

---

## What Was Implemented

### ConvOpLowering Pattern

Added new lowering pattern for `hip.conv` operation that generates LLVM IR calling a runtime function `miopenConvolutionForward`.

**Input HIP dialect:**
```mlir
%output = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
hip.conv(%ctx, %input, %weights, %bias, %output)
  {kernel_shape = [3, 3], strides = [1, 1],
   pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
  : (!hip.context, memref<1x3x224x224xf32, 1>,
     memref<64x3x3x3xf32, 1>, memref<64xf32, 1>,
     memref<1x64x224x224xf32, 1>)
```

**Output LLVM dialect:**
```mlir
// Allocation lowering (already handled by AllocOpLowering)
%ptr = llvm.call @hipMalloc(%size) : (i64) -> !llvm.ptr
%output_desc = [create memref descriptor]

// Convolution lowering (NEW - ConvOpLowering)
%input_ptr = [extract aligned ptr from input memref descriptor]
%weights_ptr = [extract aligned ptr from weights memref descriptor]
%bias_ptr = [extract aligned ptr from bias memref descriptor]
%output_ptr = [extract aligned ptr from output memref descriptor]

%result = llvm.call @miopenConvolutionForward(
    %ctx,              // handle
    %input_ptr,        // input data pointer
    %weights_ptr,      // weights data pointer
    %bias_ptr,         // bias data pointer (or null)
    %output_ptr,       // output data pointer (in-place!)
    %kernel_h,         // kernel height (i64)
    %kernel_w,         // kernel width (i64)
    %stride_h,         // stride height (i64)
    %stride_w,         // stride width (i64)
    %pad_top,          // padding top (i64)
    %pad_left,         // padding left (i64)
    %pad_bottom,       // padding bottom (i64)
    %pad_right,        // padding right (i64)
    %dilation_h,       // dilation height (i64)
    %dilation_w,       // dilation width (i64)
    %group             // number of groups (i64)
) : (...) -> i32
```

---

## Runtime Function Signature

The generated LLVM IR declares and calls the following runtime function:

```c
int miopenConvolutionForward(
    void* handle,           // miopenHandle from state
    void* input,            // input tensor data pointer
    void* weights,          // weights tensor data pointer
    void* bias,             // bias tensor data pointer (nullable)
    void* output,           // output tensor data pointer (in-place!)
    int64_t kernel_h,       // kernel height
    int64_t kernel_w,       // kernel width
    int64_t stride_h,       // stride height
    int64_t stride_w,       // stride width
    int64_t pad_top,        // padding top
    int64_t pad_left,       // padding left
    int64_t pad_bottom,     // padding bottom
    int64_t pad_right,      // padding right
    int64_t dilation_h,     // dilation height
    int64_t dilation_w,     // dilation width
    int64_t group           // number of groups
);
```

**Return value:** 0 on success, non-zero on error.

---

## Implementation Details

### Pattern: ConvOpLowering

**File:** `lib/HipDialect/HipToLLVM.cpp`

**Key steps:**

1. **Extract memref pointers**: Use `MemRefDescriptor::alignedPtr()` to get data pointers from memref descriptors
2. **Handle address space**: Cast GPU address space (1) to generic pointer (address space 0) for C ABI compatibility
3. **Handle optional bias**: Pass null pointer if bias is not provided
4. **Extract attributes**: Convert MLIR attributes to LLVM i64 constants
5. **Build function signature**: Declare `miopenConvolutionForward` with correct parameter types
6. **Call runtime function**: Generate LLVM call operation
7. **Erase HIP operation**: Remove the original `hip.conv` (in-place, no results)

**Code location:** Lines 163-283 in `lib/HipDialect/HipToLLVM.cpp`

---

## Phase 1 Design: Simplified Runtime Wrapper

**Current approach:** Generate a single runtime function call that handles:
- Descriptor creation (miopenTensorDescriptor, miopenConvolutionDescriptor)
- Workspace allocation and management
- Algorithm selection (miopenFindConvolutionForwardAlgorithm)
- Actual convolution (miopenConvolutionForward)
- Cleanup

**Rationale:**
- ✅ Simpler LLVM IR (single call vs 10+ MIOpen API calls)
- ✅ Easier to debug (all MIOpen logic in C++ runtime)
- ✅ Flexible (can change MIOpen API usage without recompiling MLIR)
- ✅ Gets the pipeline working end-to-end

**Phase 2 TODO:** Inline descriptor creation and workspace management into LLVM IR for better performance.

---

## Next Steps

### Immediate (High Priority)

1. **Implement runtime wrapper** (`miopenConvolutionForward` in C++)
   - Create `lib/HipRuntime/miopenConvolutionForward.cpp`
   - Implement full MIOpen convolution logic
   - Build as shared library linked with compiled DLL

2. **Test ONNX→HIP→LLVM pipeline**
   - Input: `test_conv_inplace.mlir`
   - Run: `hip-opt --convert-onnx-to-hip --convert-hip-to-llvm`
   - Verify: Clean LLVM IR with correct miopenConvolutionForward call

3. **Add more operation lowerings**
   - GemmOpLowering (hip.gemm → miopenGEMM or hipblasLtMatmul)
   - MaxPoolOpLowering (hip.maxpool → miopenPoolingForward)
   - AvgPoolOpLowering (hip.avgpool → miopenPoolingForward)

### Phase 2 (Performance Optimization)

4. **Inline descriptor creation**
   - Generate LLVM IR for miopenCreateTensorDescriptor, miopenSetTensorDescriptor
   - Avoid runtime overhead of creating descriptors per-call
   - **Expected benefit:** 5-10% speedup

5. **Descriptor caching in state**
   - Create descriptors once in `inference_init()`
   - Reuse in `inference_compute()`
   - **Expected benefit:** 10-20% speedup

6. **Direct MIOpen API calls**
   - Remove runtime wrapper, generate direct MIOpen API calls in LLVM IR
   - Maximum performance, but more complex IR
   - **Expected benefit:** 5-15% additional speedup

---

## Testing Strategy

### Unit Test

**File:** `test/test_conv_lowering.mlir`

```mlir
// RUN: hip-opt --convert-hip-to-llvm %s | FileCheck %s

func.func @test_conv(%ctx: !hip.context,
                      %input: memref<1x3x224x224xf32, 1>,
                      %weights: memref<64x3x3x3xf32, 1>,
                      %bias: memref<64xf32, 1>) {
  %output = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
  hip.conv(%ctx, %input, %weights, %bias, %output)
    {kernel_shape = [3, 3], strides = [1, 1],
     pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
    : (!hip.context, memref<1x3x224x224xf32, 1>,
       memref<64x3x3x3xf32, 1>, memref<64xf32, 1>,
       memref<1x64x224x224xf32, 1>)
  return
}

// CHECK: llvm.func @miopenConvolutionForward
// CHECK: llvm.call @miopenConvolutionForward
```

### End-to-End Test

**File:** `test/test_onnx_to_llvm.mlir`

Full ONNX Conv → HIP Conv → LLVM IR pipeline test.

---

## Design Rationale

### Why Runtime Wrapper Instead of Direct MIOpen Calls?

**Advantages of runtime wrapper:**
1. **Simpler IR**: One call vs 10+ calls
2. **Easier debugging**: MIOpen logic in readable C++ code
3. **Flexibility**: Can change MIOpen API usage without recompiling MLIR
4. **Workspace management**: Runtime can allocate/free workspace automatically
5. **Error handling**: Runtime can provide detailed error messages

**Disadvantages:**
1. **Function call overhead**: ~5-10ns per call (negligible for compute-bound ops)
2. **Descriptor creation overhead**: ~1-5μs (can be optimized in Phase 2)

**Conclusion:** Runtime wrapper is the right choice for Phase 1 to get the pipeline working. We can inline critical paths in Phase 2 for maximum performance.

---

## File Changes

### Modified Files

- `lib/HipDialect/HipToLLVM.cpp`: Added ConvOpLowering pattern
  - Lines 30-34: Added `kMiopenConvolutionForward` constant
  - Lines 163-283: Implemented ConvOpLowering pattern
  - Line 324: Registered ConvOpLowering in pattern list

### Files to Create (Next Steps)

- `lib/HipRuntime/miopenConvolutionForward.cpp`: Runtime wrapper implementation
- `lib/HipRuntime/CMakeLists.txt`: Build configuration for runtime library
- `test/test_conv_lowering.mlir`: Unit test for ConvOpLowering

---

## Build Verification

**Status:** Awaiting CMake configuration fix to build and test.

**Known issue:** onnx-mlir CMake configuration needs LLVM_DIR adjustment.

**Workaround for testing:** Use pre-built hip-opt if available, or fix CMake configuration.

---

## References

- **Implementation:** `lib/HipDialect/HipToLLVM.cpp` lines 163-283
- **HIP Dialect Definition:** `lib/HipDialect/HipOps.td` lines 89-134
- **Design Document:** `doc/MEMORY-MANAGEMENT.md` (In-Place Semantics section)
- **ONNX→HIP Conversion:** `notes/ONNX_TO_HIP_CONVERSION_WORKING.md`

---

**Status:** ✅ ConvOpLowering implemented for in-place hip.conv
**Next:** Implement miopenConvolutionForward runtime wrapper
