<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# OnnxToHip Pass

**Location:** `lib/HipDialect/OnnxToHip.cpp` (to be implemented)
**Input:** ONNX-MLIR module (from MorphiZen)
**Output:** HIP dialect module

---

## Overview

The OnnxToHip pass transforms ONNX operations into HIP dialect operations. This is the first major lowering step in the compilation pipeline.

**Key transformations:**
1. Extract constants to `llvm.mlir.global`
2. Convert ONNX operations to HIP operations
3. Change @main signature (add context, output arg, i32 return)
4. Generate constant management helper functions
5. Add module I/O metadata attributes

---

## Input Format (from MorphiZen)

```mlir
module {
  func.func @main(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x64x224x224xf32> {
    %weights = "onnx.Constant"() {value = dense<[...]> : tensor<64x3x3x3xf32>}
      : () -> tensor<64x3x3x3xf32>
    %bias = "onnx.Constant"() {value = dense<[...]> : tensor<64xf32>}
      : () -> tensor<64xf32>

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

**Characteristics:**
- High-level ONNX operations
- Tensor types
- Constants as `onnx.Constant` operations
- Return value semantics

---

## Output Format (HIP Dialect)

```mlir
// Module with I/O metadata
module attributes {
  hipdnn.input_count = 1 : i64,
  hipdnn.input_ranks = dense<[4]> : tensor<1xi64>,
  hipdnn.output_count = 1 : i64,
  hipdnn.output_ranks = dense<[2]> : tensor<1xi64>
} {
  // Constants extracted to globals
  llvm.mlir.global constant @constant_0(
    dense<[[[[1.0, ...]]]]> : tensor<64x3x3x3xf32>
  ) : !llvm.array<1728 x f32>

  llvm.mlir.global constant @constant_1(
    dense<[0.5, ...]> : tensor<64xf32>
  ) : !llvm.array<64 x f32>

  // Main function - signature changed
  func.func @main(%ctx: !hip.context,
                   %input: memref<1x3x224x224xf32>,
                   %output: memref<1x64x224x224xf32>) -> i32 {

    // Retrieve constants from GPU state
    %weights = hip.get_constant(%ctx, 0) : (!hip.context, i64) -> memref<64x3x3x3xf32, 1>
    %bias = hip.get_constant(%ctx, 1) : (!hip.context, i64) -> memref<64xf32, 1>

    // ONNX operations → HIP operations (inline, in-place)
    %temp = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
    hip.conv(%ctx, %input, %weights, %bias, %temp) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 1
    } : (!hip.context, memref<...>, memref<...>, memref<...>, memref<...>)

    // Destination-passing: copy result to output argument
    memref.copy %temp, %output : memref<1x64x224x224xf32, 1> to memref<1x64x224x224xf32>

    // Return success status
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }

  // Helper functions generated
  llvm.func @get_constant_count() -> i64 {
    %c2 = llvm.mlir.constant(2 : i64) : i64  // 2 constants in this model
    llvm.return %c2 : i64
  }

  func.func @initialize_constants(%ctx: !hip.context) -> i32 {
    // Upload constant_0
    %addr_0 = llvm.mlir.addressof @constant_0 : !llvm.ptr
    %size_0 = llvm.mlir.constant(6912 : i64) : i64  // 1728 × 4 bytes
    %gpu_ptr_0 = [allocate and copy to GPU]
    [store in ctx.gpu_constants[0]]

    // Upload constant_1
    %addr_1 = llvm.mlir.addressof @constant_1 : !llvm.ptr
    %size_1 = llvm.mlir.constant(256 : i64) : i64  // 64 × 4 bytes
    %gpu_ptr_1 = [allocate and copy to GPU]
    [store in ctx.gpu_constants[1]]

    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }

  func.func @release_constants(%ctx: !hip.context) -> i32 {
    // Free GPU memory for all constants
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
```

---

## Key Transformations

### 1. Add Module Metadata Attributes

**NEW: Critical for GenerateInterfacePass**

```mlir
module attributes {
  hipdnn.input_count = 1 : i64,
  hipdnn.input_ranks = dense<[4]> : tensor<1xi64>,    // Input 0 is rank 4
  hipdnn.output_count = 1 : i64,
  hipdnn.output_ranks = dense<[2]> : tensor<1xi64>    // Output 0 is rank 2
}
```

**Why?** GenerateInterfacePass needs to know:
- How many inputs/outputs to expect
- What rank each tensor has
- This drives validation and memref struct generation

**How to compute:**
- Count arguments to original @main
- Inspect tensor types to get ranks
- Store as module attributes

### 2. Extract Constants to Globals

**Before:**
```mlir
%weights = "onnx.Constant"() {value = dense<[...]> : tensor<64x3x3x3xf32>}
  : () -> tensor<64x3x3x3xf32>
```

**After:**
```mlir
llvm.mlir.global constant @constant_0(dense<[...]> : tensor<64x3x3x3xf32>)
  : !llvm.array<1728 x f32>
```

**Why?**
- Constants embedded in DLL `.data` section
- Avoid runtime parsing overhead
- Enable efficient CPU → GPU transfer

### 3. Change @main Signature

**Before:**
```mlir
func.func @main(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x64x224x224xf32>
```

**After:**
```mlir
func.func @main(%ctx: !hip.context,
                %input: memref<1x3x224x224xf32>,
                %output: memref<1x64x224x224xf32>) -> i32
```

**Changes:**
1. **Add context parameter** - `%ctx: !hip.context` (first parameter)
2. **tensor → memref** - `tensor<...>` → `memref<...>`
3. **Add output argument** - Destination-passing style
4. **Return i32** - Status code instead of tensor value

### 4. Convert Operations

**Pattern-based conversion:**

| ONNX Operation | HIP Operation |
|----------------|---------------|
| `onnx.Conv` | `hip.conv` |
| `onnx.Relu` | `hip.relu` |
| `onnx.MaxPool` | `hip.maxpool` |
| `onnx.Gemm` | `hip.gemm` |
| `onnx.BatchNormalization` | `hip.batchnorm` |
| `onnx.Constant` | (extracted to global) |

**Example conversion:**
```mlir
// Before (ONNX)
%0 = "onnx.Conv"(%input, %weights, %bias) {kernel_shape = [3, 3], ...}
  : (tensor<...>, tensor<...>, tensor<...>) -> tensor<...>

// After (HIP)
%temp = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
hip.conv(%ctx, %input, %weights, %bias, %temp) {kernel_shape = [3, 3], ...}
  : (!hip.context, memref<...>, memref<...>, memref<...>, memref<...>)
```

**Key differences:**
- HIP ops take context as first parameter
- HIP ops use in-place semantics (output as last argument)
- HIP ops don't return values (modify output in-place)

### 5. Generate Constant Helpers

**Three functions generated:**

1. **get_constant_count()**
   - Returns number of constants (compile-time constant)
   - Used by inference_init to allocate gpu_constants array

2. **initialize_constants(ctx)**
   - Uploads all constants to GPU
   - Stores GPU pointers in ctx.gpu_constants array
   - Called by inference_init

3. **release_constants(ctx)**
   - Frees all GPU constant memory
   - Called by inference_cleanup

See [../CONSTANT-MANAGEMENT.md](../CONSTANT-MANAGEMENT.md) for details.

---

## Implementation Strategy

### Pattern-Based Conversion

```cpp
// lib/HipDialect/OnnxToHip.cpp

class OnnxToHipPass : public PassWrapper<OnnxToHipPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // 1. Extract constants to globals
    extractConstantsToGlobals(module);

    // 2. Add module metadata
    addModuleMetadata(module);

    // 3. Convert operations using patterns
    ConversionTarget target(*context);
    target.addLegalDialect<HipDialect, func::FuncDialect, memref::MemRefDialect>();
    target.addIllegalDialect<ONNXDialect>();

    RewritePatternSet patterns(context);
    populateOnnxToHipPatterns(patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();

    // 4. Generate constant helper functions
    generateConstantHelpers(module);
  }
};

// Conversion patterns
void populateOnnxToHipPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvOpConversion>(patterns.getContext());
  patterns.add<ReluOpConversion>(patterns.getContext());
  patterns.add<MaxPoolOpConversion>(patterns.getContext());
  // ... more patterns
}

struct ConvOpConversion : public OpConversionPattern<ONNXConvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ONNXConvOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Get context from function arguments
    auto funcOp = op->getParentOfType<func::FuncOp>();
    Value ctx = funcOp.getArgument(0);  // First arg is context

    // Allocate output buffer
    auto outputType = convertToMemRef(op.getType());
    Value output = rewriter.create<hip::AllocOp>(
        op.getLoc(), outputType, ctx);

    // Create hip.conv
    rewriter.create<hip::ConvOp>(
        op.getLoc(),
        ctx,
        adaptor.getInput(),
        adaptor.getWeights(),
        adaptor.getBias(),
        output,
        op->getAttrs());

    rewriter.replaceOp(op, output);
    return success();
  }
};
```

---

## Prerequisites Met

This pass satisfies prerequisites for [GenerateInterfacePass.md](GenerateInterfacePass.md):

✅ **Prerequisite 2:** Adds module metadata (hipdnn.input_count, hipdnn.input_ranks, etc.) - see [GenerateInterfacePass.md - Prerequisite 2](GenerateInterfacePass.md#prerequisite-2-module-metadata-attributes)
✅ **Prerequisite 3:** Generates constant management helpers (get_constant_count, initialize_constants, release_constants) - see [GenerateInterfacePass.md - Prerequisite 3](GenerateInterfacePass.md#prerequisite-3-constant-management-functions)
✅ Generates @main with signature: `(context, input, output) -> i32`
✅ Uses memref types (ready for struct-by-value in later passes)

**Note:** This pass generates single-input, single-output @main. Multi-I/O support will be added in Phase 2.

For complete interface design, see [../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md).

---

## Related Documents

- [HipToLLVM.md](HipToLLVM.md) - Next pass in pipeline
- [../CONSTANT-MANAGEMENT.md](../CONSTANT-MANAGEMENT.md) - Constant helper details
- [../HIP-DIALECT-DESIGN.md](../HIP-DIALECT-DESIGN.md) - HIP dialect overview
- [../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md) - Prerequisites this pass must satisfy
