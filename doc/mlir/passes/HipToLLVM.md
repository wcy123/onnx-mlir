# HipToLLVM Pass

**Location:** `lib/HipDialect/HipToLLVM.cpp`
**Input:** HIP dialect module
**Output:** LLVM dialect module with wrapper functions

---

## Overview

The HipToLLVM pass lowers HIP dialect operations to LLVM dialect, generating wrapper functions that encapsulate MIOpen/hipBLAS calls.

**Key transformations:**
1. Lower !hip.context → !llvm.ptr
2. Generate wrapper functions for HIP operations
3. Convert HIP ops to wrapper calls
4. Transform @main to use memref struct arrays
5. Lower constant helpers to LLVM

---

## Input Format (HIP Dialect)

```mlir
module attributes {
  hipdnn.input_count = 1 : i64,
  hipdnn.input_ranks = dense<[4]> : tensor<1xi64>,
  hipdnn.output_count = 1 : i64,
  hipdnn.output_ranks = dense<[2]> : tensor<1xi64>
} {
  func.func @main(%ctx: !hip.context,
                   %input: memref<1x3x224x224xf32>,
                   %output: memref<1x64x224x224xf32>) -> i32 {

    %weights = hip.get_constant(%ctx, 0) : (!hip.context, i64) -> memref<64x3x3x3xf32, 1>
    %bias = hip.get_constant(%ctx, 1) : (!hip.context, i64) -> memref<64xf32, 1>

    %temp = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
    hip.conv(%ctx, %input, %weights, %bias, %temp) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 1
    } : (!hip.context, memref<...>, memref<...>, memref<...>, memref<...>)

    memref.copy %temp, %output
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
```

---

## Output Format (LLVM Dialect)

```mlir
module attributes {
  hipdnn.input_count = 1 : i64,
  hipdnn.input_ranks = dense<[4]> : tensor<1xi64>,
  hipdnn.output_count = 1 : i64,
  hipdnn.output_ranks = dense<[2]> : tensor<1xi64>
} {
  // Wrapper function generated for hip.conv
  llvm.func @hip_conv_wrapper(
      %ctx: !llvm.ptr,
      %input: !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>,
      %weights: !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>,
      %bias: !llvm.struct<(ptr<1>, ptr<1>, i64, array<1xi64>, array<1xi64>)>,
      %output: !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>,
      %kernel_h: i64, %kernel_w: i64,
      %stride_h: i64, %stride_w: i64,
      %pad_top: i64, %pad_left: i64, %pad_bottom: i64, %pad_right: i64,
      %dilation_h: i64, %dilation_w: i64,
      %group: i64) -> i32 {

    // Extract data pointers from memref structs
    %input_ptr = llvm.extractvalue %input[1] : !llvm.struct<...>
    %weights_ptr = llvm.extractvalue %weights[1] : !llvm.struct<...>
    %bias_ptr = llvm.extractvalue %bias[1] : !llvm.struct<...>
    %output_ptr = llvm.extractvalue %output[1] : !llvm.struct<...>

    // CRITICAL: Extract RUNTIME dimensions from memref structs
    %input_n = llvm.extractvalue %input[3, 0] : !llvm.struct<...> -> i64  // Runtime!
    %input_c = llvm.extractvalue %input[3, 1] : !llvm.struct<...> -> i64  // Runtime!
    %input_h = llvm.extractvalue %input[3, 2] : !llvm.struct<...> -> i64  // Runtime!
    %input_w = llvm.extractvalue %input[3, 3] : !llvm.struct<...> -> i64  // Runtime!
    // ... extract output dimensions

    // Get MIOpen handle from context
    %miopen_ptr = llvm.getelementptr %ctx[0, 1] : (!llvm.ptr) -> !llvm.ptr
    %miopen = llvm.load %miopen_ptr : !llvm.ptr

    // Create MIOpen descriptors using RUNTIME dimensions
    %xDesc = [create tensor descriptor with runtime dims]
    llvm.call @miopenSet4dTensorDescriptor(%xDesc, %dataType,
                                            %input_n, %input_c, %input_h, %input_w)

    // Call MIOpen
    %ret = llvm.call @miopenConvolutionForward(
      %miopen, %alpha_ptr,
      %xDesc, %input_ptr,
      %wDesc, %weights_ptr,
      %convDesc, %algo,
      %beta_ptr,
      %yDesc, %output_ptr,
      %workspace, %workspace_size
    ) : (...) -> i32

    // Cleanup descriptors
    llvm.call @miopenDestroyTensorDescriptor(%xDesc)
    // ...

    llvm.return %ret : i32
  }

  // Main function - transformed to use memref struct arrays
  llvm.func @main(%context: !llvm.ptr,
                  %inputs: !llvm.ptr,   // Array of memref structs
                  %outputs: !llvm.ptr)  // Array of memref structs
                  -> i32 {

    // Load input memref struct from array
    %input_0_ptr = llvm.getelementptr %inputs[0] : (!llvm.ptr) -> !llvm.ptr
    %input_0 = llvm.load %input_0_ptr : !llvm.ptr
      -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>

    // Load output memref struct from array
    %output_0_ptr = llvm.getelementptr %outputs[0] : (!llvm.ptr) -> !llvm.ptr
    %output_0 = llvm.load %output_0_ptr : !llvm.ptr
      -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>

    // Get constants
    %weights_gpu = llvm.call @hip_get_constant(%context, %c0) : (!llvm.ptr, i64) -> !llvm.ptr
    %weights_struct = [build memref struct]

    %bias_gpu = llvm.call @hip_get_constant(%context, %c1) : (!llvm.ptr, i64) -> !llvm.ptr
    %bias_struct = [build memref struct]

    // Call wrapper
    %ret = llvm.call @hip_conv_wrapper(
      %context, %input_0, %weights_struct, %bias_struct, %temp_struct,
      3, 3,        // kernel_h, kernel_w
      1, 1,        // stride_h, stride_w
      1, 1, 1, 1,  // pads
      1, 1,        // dilations
      1            // group
    ) : (...) -> i32

    // Copy to output
    [memref.copy lowering]

    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %c0_i32 : i32
  }

  // Constant helpers lowered to LLVM
  llvm.func @get_constant_count() -> i64 {
    %c2 = llvm.mlir.constant(2 : i64) : i64
    llvm.return %c2 : i64
  }

  llvm.func @initialize_constants(%context: !llvm.ptr) -> i32 {
    // Get constant addresses
    %addr_0 = llvm.mlir.addressof @constant_0 : !llvm.ptr
    %size_0 = llvm.mlir.constant(6912 : i64) : i64

    // Allocate GPU memory
    %gpu_ptr_0_ptr = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr
    llvm.call @hipMalloc(%gpu_ptr_0_ptr, %size_0) : (!llvm.ptr, i64) -> i32
    %gpu_ptr_0 = llvm.load %gpu_ptr_0_ptr : !llvm.ptr

    // Copy to GPU
    %kind = llvm.mlir.constant(1 : i32) : i32  // HostToDevice
    llvm.call @hipMemcpy(%gpu_ptr_0, %addr_0, %size_0, %kind) : (...) -> i32

    // Store in context.gpu_constants[0]
    %gpu_constants_ptr = llvm.getelementptr %context[0, 3] : (!llvm.ptr) -> !llvm.ptr
    %gpu_constants = llvm.load %gpu_constants_ptr : !llvm.ptr
    %slot_0 = llvm.getelementptr %gpu_constants[0] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %gpu_ptr_0, %slot_0 : !llvm.ptr

    // Repeat for other constants...

    %c0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %c0 : i32
  }

  llvm.func @release_constants(%context: !llvm.ptr) -> i32 {
    %gpu_constants_ptr = llvm.getelementptr %context[0, 3] : (!llvm.ptr) -> !llvm.ptr
    %gpu_constants = llvm.load %gpu_constants_ptr : !llvm.ptr

    %slot_0 = llvm.getelementptr %gpu_constants[0] : (!llvm.ptr) -> !llvm.ptr
    %gpu_ptr_0 = llvm.load %slot_0 : !llvm.ptr
    llvm.call @hipFree(%gpu_ptr_0) : (!llvm.ptr) -> i32

    // Repeat for other constants...

    %c0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %c0 : i32
  }

  llvm.func @hip_get_constant(%context: !llvm.ptr, %index: i64) -> !llvm.ptr {
    %gpu_constants_ptr = llvm.getelementptr %context[0, 3] : (!llvm.ptr) -> !llvm.ptr
    %gpu_constants = llvm.load %gpu_constants_ptr : !llvm.ptr
    %slot = llvm.getelementptr %gpu_constants[%index] : (!llvm.ptr, i64) -> !llvm.ptr
    %gpu_ptr = llvm.load %slot : !llvm.ptr
    llvm.return %gpu_ptr : !llvm.ptr
  }
}
```

---

## Key Transformations

### 1. Generate Wrapper Functions

**Why?** MIOpen/hipBLAS have complex APIs (13+ parameters). Wrappers encapsulate this complexity.

**Strategy:** Generate wrapper on-demand (reuse if exists)

```cpp
static LLVM::LLVMFuncOp getOrCreateConvWrapper(ModuleOp module, OpBuilder &builder) {
  // Check if wrapper already exists
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>("hip_conv_wrapper"))
    return func;  // Reuse

  // Create wrapper function
  // ... (build signature, body)
  return func;
}
```

**Wrapper accepts memref structs:**
- Input, weights, bias, output as struct-by-value
- Extract dimensions at runtime (supports dynamic shapes!)
- Extract data pointers
- Call MIOpen with extracted values

See [../HIP-DIALECT-DESIGN.md](../HIP-DIALECT-DESIGN.md) for complete wrapper design.

### 2. Transform @main Signature

**Critical change for Prerequisite 1:**

**Before (HIP dialect):**
```mlir
func.func @main(%ctx: !hip.context,
                %input: memref<1x3x224x224xf32>,
                %output: memref<1x64x224x224xf32>) -> i32
```

**After (LLVM dialect):**
```mlir
llvm.func @main(%context: !llvm.ptr,
                %inputs: !llvm.ptr,   // Pointer to array of memref structs
                %outputs: !llvm.ptr)  // Pointer to array of memref structs
                -> i32
```

**Why this change?**
- Supports multiple inputs/outputs (scalable)
- Memref structs contain runtime dimensions (dynamic shapes!)
- Matches GenerateInterfacePass expectations (Prerequisite 1)

**How @main accesses tensors:**
```mlir
// Get input 0
%input_0_ptr = llvm.getelementptr %inputs[0] : (!llvm.ptr) -> !llvm.ptr
%input_0 = llvm.load %input_0_ptr : !llvm.ptr -> !llvm.struct<...>

// Extract runtime dimensions
%batch = llvm.extractvalue %input_0[3, 0] : !llvm.struct<...> -> i64  // Runtime!
```

### 3. Lower HIP Operations to Wrapper Calls

**Pattern:**
```cpp
struct ConvOpLowering : public ConvertOpToLLVMPattern<hip::ConvOp> {
  LogicalResult matchAndRewrite(hip::ConvOp op, ...) {
    // Get or create wrapper
    auto wrapper = getOrCreateConvWrapper(module, rewriter, loc);

    // Build arguments
    SmallVector<Value> args = {
      adaptor.getContext(),
      adaptor.getInput(),
      adaptor.getWeights(),
      adaptor.getBias(),
      adaptor.getOutput(),
      // ... kernel, stride, pad, dilation params from attributes
    };

    // Replace hip.conv with call to wrapper
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, wrapper, args);
    return success();
  }
};
```

### 4. Lower Constant Helpers

**initialize_constants:**
- Use `llvm.mlir.addressof` to get global constant addresses
- Call `hipMalloc` to allocate GPU memory
- Call `hipMemcpy` to copy CPU → GPU
- Store GPU pointers in context.gpu_constants array

**release_constants:**
- Load GPU pointers from context.gpu_constants
- Call `hipFree` for each

**hip_get_constant:**
- Load GPU pointer from context.gpu_constants[index]
- Return pointer

---

## Dynamic Shape Support

Wrappers extract dimensions at runtime from memref structs. For complete dynamic shape design and rationale, see [../../DYNAMIC-SHAPE-DESIGN.md](../../DYNAMIC-SHAPE-DESIGN.md).

Example:
```mlir
// In wrapper function - dimensions extracted at runtime
%input_n = llvm.extractvalue %input[3, 0] : !llvm.struct<...> -> i64  // RUNTIME!
%input_c = llvm.extractvalue %input[3, 1] : !llvm.struct<...> -> i64  // RUNTIME!
%input_h = llvm.extractvalue %input[3, 2] : !llvm.struct<...> -> i64  // RUNTIME!
%input_w = llvm.extractvalue %input[3, 3] : !llvm.struct<...> -> i64  // RUNTIME!

// Pass runtime dimensions to MIOpen
llvm.call @miopenSet4dTensorDescriptor(%xDesc, %dataType,
                                        %input_n, %input_c, %input_h, %input_w)
```

---

## Prerequisites Met

This pass satisfies Prerequisite 1 requirements from [../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md#prerequisite-1-main-function-signature):

✅ Transforms @main to signature: `(context, inputs, outputs) -> i32`
✅ Uses memref struct arrays (struct-by-value)
✅ Supports dynamic shapes (dimensions extracted at runtime)
✅ Generates wrapper functions that handle runtime dimensions
✅ Lowers constant helpers to LLVM

For detailed prerequisite specifications, see [../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md#generateinterfacepass-prerequisites).

---

## Implementation Notes

**Pattern registration:**
```cpp
void populateHipToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                RewritePatternSet &patterns) {
  patterns.add<ConvOpLowering>(typeConverter);
  patterns.add<AllocOpLowering>(typeConverter);
  patterns.add<GetConstantOpLowering>(typeConverter);
  // ... more patterns
}
```

**Type conversion:**
```cpp
typeConverter.addConversion([](hip::ContextType) -> Type {
  return LLVM::LLVMPointerType::get(context);
});

typeConverter.addConversion([](MemRefType type) -> Type {
  return convertMemRefToLLVMStruct(type);  // Struct-by-value!
});
```

---

## Related Documents

- [OnnxToHip.md](OnnxToHip.md) - Previous pass in pipeline
- [GenerateInterfacePass.md](GenerateInterfacePass.md) - Next pass in pipeline
- [../HIP-DIALECT-DESIGN.md](../HIP-DIALECT-DESIGN.md) - Wrapper function details
- [../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md) - Prerequisites satisfied
- [../../DYNAMIC-SHAPE-DESIGN.md](../../DYNAMIC-SHAPE-DESIGN.md) - Dynamic shape flow
