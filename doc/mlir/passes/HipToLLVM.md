<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
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

**Note:** This pass generates **simple calls to external C++ runtime functions**, not elaborate inline MIOpen wrappers. The documentation below shows the conceptual flow, but the actual implementation uses pre-existing C++ functions in `lib/Runtime/hipdnn_ep_runtime_miopen.cpp`.

```mlir
module attributes {
  hipdnn.input_count = 1 : i64,
  hipdnn.input_ranks = dense<[4]> : tensor<1xi64>,
  hipdnn.output_count = 1 : i64,
  hipdnn.output_ranks = dense<[2]> : tensor<1xi64>
} {
  // External declaration to C++ runtime function
  // Implementation in lib/Runtime/hipdnn_ep_runtime_miopen.cpp
  llvm.func @wrap_miopenConvolutionForward(
      !llvm.ptr,      // handle (miopenHandle + hipStream from RuntimeState)
      !llvm.ptr,      // stream
      !llvm.ptr,      // input data pointer
      !llvm.ptr,      // input_shape (int64_t[4])
      !llvm.ptr,      // weights data pointer
      !llvm.ptr,      // weights_shape (int64_t[4])
      !llvm.ptr,      // output data pointer
      !llvm.ptr,      // output_shape (int64_t[4])
      i64, i64,       // pad_h, pad_w
      i64, i64,       // stride_h, stride_w
      i64, i64        // dilation_h, dilation_w
  ) -> i32

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

    // Extract data pointers from memref structs
    %input_ptr = llvm.extractvalue %input_0[1] : !llvm.struct<...> -> !llvm.ptr<1>
    %output_ptr = llvm.extractvalue %output_0[1] : !llvm.struct<...> -> !llvm.ptr<1>

    // Extract shape arrays from memref structs
    %input_shape_arr = llvm.extractvalue %input_0[3] : !llvm.struct<...> -> !llvm.array<4xi64>
    %output_shape_arr = llvm.extractvalue %output_0[3] : !llvm.struct<...> -> !llvm.array<4xi64>

    // Get shape array pointers (for passing to C++ runtime)
    %input_shape_ptr = llvm.mlir.addressof @input_shape : !llvm.ptr
    %output_shape_ptr = llvm.mlir.addressof @output_shape : !llvm.ptr

    // Get constants (weights data pointer from RuntimeState)
    %weights_ptr = llvm.call @hipdnn_ep_get_constant(%context, %c0) : (!llvm.ptr, i64) -> !llvm.ptr

    // Get MIOpen handle and stream from RuntimeState
    %handle_ptr = llvm.getelementptr %context[0, 0] : (!llvm.ptr) -> !llvm.ptr
    %handle = llvm.load %handle_ptr : !llvm.ptr
    %stream_ptr = llvm.getelementptr %context[0, 2] : (!llvm.ptr) -> !llvm.ptr
    %stream = llvm.load %stream_ptr : !llvm.ptr

    // Weights shape (compile-time constant)
    %weights_shape_ptr = llvm.mlir.addressof @weights_shape_constant : !llvm.ptr

    // Call C++ runtime wrapper (NOT inline MIOpen calls!)
    %ret = llvm.call @wrap_miopenConvolutionForward(
      %handle, %stream,
      %input_ptr, %input_shape_ptr,
      %weights_ptr, %weights_shape_ptr,
      %output_ptr, %output_shape_ptr,
      1, 1,        // pad_h, pad_w
      1, 1,        // stride_h, stride_w
      1, 1         // dilation_h, dilation_w
    ) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
         !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64) -> i32

    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %c0_i32 : i32
  }
}
```

**Key Differences from Earlier Description:**

The HipToLLVM pass implements a **simplified Phase 1 architecture**:

1. **No inline MIOpen wrapper generation**: The pass generates **external function declarations** to pre-existing C++ runtime functions (see `lib/HipDialect/HipToLLVM.cpp:37-38`).

2. **All MIOpen complexity in C++ runtime**: Descriptor creation, algorithm finding, workspace allocation - all handled in `lib/Runtime/hipdnn_ep_runtime_miopen.cpp:34-115`.

3. **Pass only extracts data pointers**: The MLIR code extracts aligned pointers and shape arrays from memref structs, then passes them to the C++ wrapper.

4. **No runtime dimension extraction in MLIR**: Unlike the earlier description showing `llvm.extractvalue %input[3, 0]` for dimensions, the actual implementation passes entire shape arrays to C++, which extracts dimensions as needed.

**Why this architecture?** (See `lib/HipDialect/HipToLLVM.cpp:191-194`)
- Simplified initial implementation
- Easier debugging (MIOpen calls in familiar C++ code)
- Preparation for future Phase 2 (full MLIR lowering)

**What the C++ runtime does** (`lib/Runtime/hipdnn_ep_runtime_miopen.cpp`):
- Creates tensor descriptors with `miopenSet4dTensorDescriptor()`
- Finds optimal algorithm with `miopenFindConvolutionForwardAlgorithm()`
- Allocates workspace memory
- Calls `miopenConvolutionForward()`
- Cleans up descriptors and workspace

---

### Constant Registry (Unchanged)

```mlir
module {
  // Constant registry (no transformation needed - already in LLVM)
  llvm.mlir.global constant @constant_info_array() : !llvm.array<2 x !llvm.struct<(ptr, i64, i64, i64)>> {
    // Array of ConstantInfo structs
  }

  llvm.mlir.global constant @constant_registry() : !llvm.struct<(ptr, i64)> {
    // ConstantRegistry struct
  }

  llvm.func @get_constant_registry() -> !llvm.ptr {
    %registry_ptr = llvm.mlir.addressof @constant_registry : !llvm.ptr
    llvm.return %registry_ptr : !llvm.ptr
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

### 1. Transform @main Signature (Array-Based Interface)

**Critical transformation to satisfy GenerateInterfacePass Prerequisite 1.**

**Problem:** Standard MLIR memref-to-llvm conversion unpacks memrefs into scalar components. For a rank-4 tensor, this creates **11 parameters** (2 pointers + 1 offset + 4 sizes + 4 strides). With 1 input and 1 output, @main gets **23 parameters** - barely readable and doesn't scale.

**Solution:** Two-function wrapper architecture:

1. **@main** (3 parameters): Clean array-based interface
   - Signature: `(context: !llvm.ptr, inputs: !llvm.ptr, outputs: !llvm.ptr) -> i32`
   - Loads memref structs from arrays using GEP + load
   - Unpacks structs using extractvalue (11 extracts per rank-4 tensor)
   - Calls @main_internal with unpacked parameters
   - Private (not exported from DLL)

2. **@main_internal** (23+ parameters): Computation logic
   - Original unpacked signature from standard conversion
   - Contains actual computation (calls to MIOpen wrappers, etc.)
   - Private (not exported from DLL)

**Transformation Flow:**

```
Standard MLIR conversion
    ↓
@main with 23 unpacked params (allocated, aligned, offset, sizes[4], strides[4] for each tensor)
    ↓ transformMainFunction()
Rename to @main_internal (private)
    ↓
Create new @main (3 params)
    ↓
@main loads memref structs from arrays
    ↓
@main extracts 11 fields per tensor using llvm.extractvalue
    ↓
@main calls @main_internal with 23 unpacked params
```

**Code Example:**

```mlir
// NEW: Clean 3-parameter wrapper
llvm.func private @main(%ctx: !llvm.ptr, %inputs: !llvm.ptr, %outputs: !llvm.ptr) -> i32 {
  // Load input memref struct from inputs[0]
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %input_ptr = llvm.getelementptr %inputs[%c0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.ptr
  %input = llvm.load %input_ptr : !llvm.ptr
    -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>

  // Extract 11 fields
  %allocated = llvm.extractvalue %input[0] : !llvm.struct<...> -> !llvm.ptr<1>
  %aligned = llvm.extractvalue %input[1] : !llvm.struct<...> -> !llvm.ptr<1>
  %offset = llvm.extractvalue %input[2] : !llvm.struct<...> -> i64
  %size0 = llvm.extractvalue %input[3, 0] : !llvm.struct<...> -> i64  // Runtime dimension!
  // ... extract remaining sizes and strides

  // Load output struct (similar)
  // ...

  // Call computation function with 23 unpacked params
  %result = llvm.call @main_internal(%ctx, %allocated, %aligned, %offset,
                                      %size0, %size1, %size2, %size3,
                                      %stride0, %stride1, %stride2, %stride3,
                                      %out_allocated, %out_aligned, ...)
  llvm.return %result : i32
}

// Internal computation (original body, 23 parameters)
llvm.func private @main_internal(%ctx: !llvm.ptr, %arg1: !llvm.ptr<1>, ..., %arg22: i64) -> i32 {
  // Rebuild memref descriptors from unpacked params
  %0 = llvm.mlir.poison : !llvm.struct<...>
  %1 = llvm.insertvalue %arg12, %0[0] : ...
  // ... (computation logic)
}
```

**Metadata-Driven:** Uses module attributes to determine structure:
- `hipdnn.input_count` - number of input tensors
- `hipdnn.input_ranks` - rank of each input (e.g., [4] for one rank-4 tensor)
- `hipdnn.output_count` - number of output tensors
- `hipdnn.output_ranks` - rank of each output

For each tensor with rank R, unpacking extracts **2 + 1 + R + R** parameters.

**Dynamic Shape Support:**
- Rank is compile-time (from metadata)
- Dimension values are runtime (loaded from memref struct)
- Unpacking preserves runtime dimension values
- No special handling needed - works automatically!

**Benefits:**
- ✅ Readable: 3 parameters instead of 23+
- ✅ Scalable: Works for any number of inputs/outputs
- ✅ Type-safe: Memref structs preserve shape information
- ✅ Satisfies Prerequisite 1: Required by GenerateInterfacePass
- ✅ Dynamic shape ready: Runtime dimensions flow through unchanged

**Implementation:** `lib/HipDialect/HipToLLVM.cpp`, method `transformMainFunction()`

---

### 2. Generate Wrapper Functions

**Purpose:** Generate reusable wrapper functions that convert memref structs to GPU API calls.

**Why generate wrappers?** See [WHY-HIP-WRAPPERS.md](WHY-HIP-WRAPPERS.md) for detailed explanation.

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

### 4. Lower Constant Registry

**get_constant_registry:**
- Already in LLVM dialect (generated by OnnxToHip)
- No transformation needed
- Returns pointer to ConstantRegistry struct in .data section

**hip_get_constant:**
- Load GPU pointer from context.gpu_constants[index]
- Return pointer
- Used by @main to retrieve pre-uploaded constants

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

This pass satisfies **Prerequisite 1** for [GenerateInterfacePass.md](GenerateInterfacePass.md):

✅ **Prerequisite 1:** Transforms @main to signature: `(context, inputs, outputs) -> i32` with memref struct arrays - see [GenerateInterfacePass.md - Prerequisite 1](GenerateInterfacePass.md#prerequisite-1-main-function-signature-dynamic-shape-ready)
✅ Uses memref struct arrays (struct-by-value)
✅ Supports dynamic shapes (dimensions extracted at runtime)
✅ Generates wrapper functions that handle runtime dimensions
✅ Constant registry already in LLVM (no transformation needed)

For complete interface design, see [../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md).

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
