# HIP Dialect Design

**Related:** [../MLIR-COMPILATION-OVERVIEW.md](../MLIR-COMPILATION-OVERVIEW.md)

---

## Overview

This document describes the HIP MLIR dialect design, including:
1. HIP context type (!hip.context)
2. Wrapper function generation for MIOpen/hipBLAS calls
3. Dynamic shape support in wrappers

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

### Design Rationale

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

### Usage in IR

```mlir
// Context is first parameter, outputs as arguments (destination-passing)
func @main_graph(%ctx: !hip.context, %input: memref<...>, %output_arg: memref<...>) -> i32 {
  // HIP operations use in-place semantics (output as argument)
  %output = hip.alloc(%ctx) : memref<...>
  hip.conv(%ctx, %input, %weights, %bias, %output) {...}

  %result = hip.alloc(%ctx) : memref<...>
  hip.gemm(%ctx, %output, %matrix, %result) {...}

  // Destination-passing: copy to output argument
  memref.copy %result, %output_arg : memref<...> to memref<...>

  // Return success status
  %c0_i32 = arith.constant 0 : i32
  return %c0_i32 : i32
}
```

### Lowering to LLVM

```mlir
// HIP dialect (before lowering) - in-place semantics
%output = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
hip.conv(%ctx, %input, %weights, %bias, %output) {kernel_shape = [3, 3], ...}
  : (!hip.context, memref<...>, memref<...>, memref<...>, memref<...>)

// LLVM dialect (after lowering)
// %ctx is now !llvm.ptr, used to extract handles
%miopen_ptr = llvm.getelementptr %ctx[0, 1] : (!llvm.ptr) -> !llvm.ptr
%miopen = llvm.load %miopen_ptr : !llvm.ptr
llvm.call @hip_conv_wrapper(%ctx, %input, %weights, %bias, %output, ...)
```

**Alternative Designs Considered (and rejected):**

- **Structured type** `!hip.context<{stream, miopen, hipblas, weights}>`: Too rigid, exposes implementation
- **Multiple parameters** `func(..., %stream, %miopen, %hipblas, ...)`: Doesn't scale, verbose
- **Global state**: Not thread-safe, harder to reason about

---

## Wrapper Function Design

### Motivation: Why Wrappers?

MIOpen and hipBLAS have complex C APIs with many parameters. For example, `miopenConvolutionForward` requires 13+ parameters including descriptors, scalars, and workspace.

**Problems with direct calls:**
1. **Verbose**: 30+ lines of LLVM IR per convolution
2. **Repetitive**: Same pattern repeated for every operation
3. **Hard to generate**: Complex codegen in lowering patterns
4. **Hard to read**: Generated IR is cluttered

**Solution:** Generate wrapper functions that encapsulate MIOpen complexity.

### Wrapper Function Structure

Each HIP operation gets a corresponding wrapper function. See [LOWERING-PIPELINE.md](LOWERING-PIPELINE.md) for complete example.

**Key steps in wrapper:**
1. Extract data pointers from memref structs
2. **Extract runtime dimensions from memref structs** (supports dynamic shapes!)
3. Get library handle from context
4. Create descriptors using runtime dimensions
5. Call MIOpen/hipBLAS
6. Cleanup descriptors
7. Return status

**Critical for dynamic shapes:** Wrappers extract dimension values from memref structs at runtime:

```mlir
// Extract runtime dimensions (not compile-time constants!)
%input_n = llvm.extractvalue %input[3, 0] : !llvm.struct<...> -> i64  // Batch (runtime!)
%input_c = llvm.extractvalue %input[3, 1] : !llvm.struct<...> -> i64  // Channels (runtime!)
%input_h = llvm.extractvalue %input[3, 2] : !llvm.struct<...> -> i64  // Height (runtime!)
%input_w = llvm.extractvalue %input[3, 3] : !llvm.struct<...> -> i64  // Width (runtime!)

// Pass runtime dimensions to MIOpen
llvm.call @miopenSet4dTensorDescriptor(%xDesc, %dataType,
                                        %input_n, %input_c, %input_h, %input_w)
```

This is how dynamic shapes work end-to-end:
1. User provides tensor with shape [2, 3, 256, 256]
2. inference_compute loads dimensions from tensor_t.shape
3. Builds memref struct with sizes = [2, 3, 256, 256] (runtime values!)
4. @main passes memref to wrapper
5. Wrapper extracts dimensions and passes to MIOpen

### Wrapper Generation in HipToLLVM Pass

**Implementation strategy:**

```cpp
// lib/HipDialect/HipToLLVM.cpp

// Get or create wrapper function (called on-demand)
static LLVM::LLVMFuncOp getOrCreateConvWrapper(ModuleOp module, ...) {
  // Check if wrapper already exists
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>("hip_conv_wrapper"))
    return func;  // Reuse

  // Create wrapper function with memref struct parameters
  // Build wrapper body (extract, call MIOpen, cleanup)
  return func;
}

// Pattern: Lower hip.conv to wrapper call
struct ConvOpLowering : public ConvertOpToLLVMPattern<ConvOp> {
  LogicalResult matchAndRewrite(ConvOp op, ...) {
    // Get or create wrapper (generated once, reused many times)
    auto wrapper = getOrCreateConvWrapper(module, ...);

    // Replace hip.conv with call to wrapper
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, wrapper, args);
    return success();
  }
};
```

### Design Benefits

1. **Clean separation**: Wrapper generation separate from lowering patterns
2. **Reusable**: One wrapper per operation type, reused across module
3. **Maintainable**: MIOpen complexity encapsulated
4. **Testable**: Can test wrapper generation independently
5. **Readable IR**: @main is concise, wrappers are isolated
6. **Dynamic shape ready**: Dimensions extracted at runtime

### Wrapper Functions for Other Operations

Similar wrappers for:
- `@hip_gemm_wrapper` - hipblasLtMatmul
- `@hip_maxpool_wrapper` - miopenPoolingForward
- `@hip_batchnorm_wrapper` - miopenBatchNormalizationForward
- `@hip_relu_wrapper` - miopenActivationForward

All follow same pattern: memref structs → extract runtime dimensions → call library → return status

### Future Optimizations (Phase 2)

1. **Descriptor caching**: Create once in inference_init, reuse
2. **Workspace pre-allocation**: Allocate in init, reuse across operations
3. **Algorithm caching**: Find best algorithm once, cache in state

These optimizations don't change wrapper signatures - backward compatible.

---

## Related Documents

- [MODULE-STRUCTURE.md](MODULE-STRUCTURE.md) - Where wrappers fit in module structure
- [LOWERING-PIPELINE.md](LOWERING-PIPELINE.md) - Complete wrapper implementation examples
- [../DYNAMIC-SHAPE-DESIGN.md](../DYNAMIC-SHAPE-DESIGN.md) - How dynamic shapes flow through wrappers
- [../STATE-AND-CONTEXT.md](../STATE-AND-CONTEXT.md) - Context struct layout
