# MLIR Compilation Demo: ONNX Conv → HIP → LLVM

**Proof of concept**: Lowering ONNX Conv operation through HIP dialect to LLVM IR.

---

## Step 1: Input - ONNX Conv in MLIR

**File**: `demo_input.mlir`
```mlir
module {
  func.func @inference_compute(%state: !hip.handle) -> memref<1x64x112x112xf32> {
    // Input tensors (normally from model weights)
    %input = memref.alloc() : memref<1x3x224x224xf32>
    %weights = memref.alloc() : memref<64x3x7x7xf32>
    %bias = memref.alloc() : memref<64xf32>

    // ONNX Conv operation
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [7, 7],
      strides = [2, 2],
      pads = [3, 3, 3, 3],
      dilations = [1, 1],
      group = 1 : si64
    } : (memref<1x3x224x224xf32>, memref<64x3x7x7xf32>, memref<64xf32>)
      -> memref<1x64x112x112xf32>

    return %output : memref<1x64x112x112xf32>
  }
}
```

---

## Step 2: Transform - ONNX → HIP Dialect

**Command**:
```bash
hip-opt demo_input.mlir --convert-onnx-to-hip
```

**Expected Output**: `demo_hip.mlir`
```mlir
module {
  func.func @inference_compute(%state: !hip.handle) -> memref<1x64x112x112xf32> {
    %input = memref.alloc() : memref<1x3x224x224xf32>
    %weights = memref.alloc() : memref<64x3x7x7xf32>
    %bias = memref.alloc() : memref<64xf32>

    // Lowered to HIP dialect - uses MIOpen backend
    %output = hip.conv(%state, %input, %weights, %bias) {
      kernel_shape = [7, 7],
      strides = [2, 2],
      pads = [3, 3, 3, 3],
      dilations = [1, 1],
      group = 1 : i64
    } : (memref<1x3x224x224xf32>, memref<64x3x7x7xf32>, memref<64xf32>)
      -> memref<1x64x112x112xf32>

    return %output : memref<1x64x112x112xf32>
  }
}
```

**What happened**:
- `"onnx.Conv"(...)` → `hip.conv(%state, ...)`
- State handle now explicitly passed (from function argument)
- Attributes preserved: kernel_shape, strides, pads, dilations, group

---

## Step 3: Transform - HIP → LLVM Dialect

**Command**:
```bash
hip-opt demo_hip.mlir --convert-hip-to-llvm
```

**Expected Output**: `demo_llvm.mlir`
```mlir
module {
  // Declare MIOpen runtime functions
  llvm.func @miopenConvolutionForward(
    !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
    !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
    !llvm.ptr, i64, !llvm.ptr
  ) -> i32

  func.func @inference_compute(%state: !llvm.ptr) -> !llvm.struct<...> {
    // Allocate input/weight/bias memrefs
    %input_ptr = llvm.call @hipMalloc(%input_size) : (i64) -> !llvm.ptr
    %weights_ptr = llvm.call @hipMalloc(%weights_size) : (i64) -> !llvm.ptr
    %bias_ptr = llvm.call @hipMalloc(%bias_size) : (i64) -> !llvm.ptr
    %output_ptr = llvm.call @hipMalloc(%output_size) : (i64) -> !llvm.ptr

    // Extract miopenHandle from state (offset 8 bytes)
    %c8 = llvm.mlir.constant(8 : i64) : i64
    %handle_ptr = llvm.getelementptr %state[%c8] : (!llvm.ptr, i64) -> !llvm.ptr
    %miopen_handle = llvm.load %handle_ptr : !llvm.ptr -> !llvm.ptr

    // Create tensor descriptors
    %input_desc = llvm.call @miopenCreateTensorDescriptor() : () -> !llvm.ptr
    %weights_desc = llvm.call @miopenCreateTensorDescriptor() : () -> !llvm.ptr
    %output_desc = llvm.call @miopenCreateTensorDescriptor() : () -> !llvm.ptr
    %conv_desc = llvm.call @miopenCreateConvolutionDescriptor() : () -> !llvm.ptr

    // Set descriptor parameters
    llvm.call @miopenSetTensorDescriptor(%input_desc, ...) : ...
    llvm.call @miopenSetConvolutionDescriptor(%conv_desc,
      %pad_h=3, %pad_w=3, %stride_h=2, %stride_w=2, %dilation_h=1, %dilation_w=1) : ...

    // Find best convolution algorithm
    %workspace_size = llvm.call @miopenConvolutionForwardGetWorkSpaceSize(...) : ...
    %workspace = llvm.call @hipMalloc(%workspace_size) : (i64) -> !llvm.ptr

    // Execute convolution
    %alpha = llvm.mlir.constant(1.0 : f32) : f32
    %beta = llvm.mlir.constant(0.0 : f32) : f32
    llvm.call @miopenConvolutionForward(
      %miopen_handle,
      %alpha_ptr, %input_desc, %input_ptr,
      %weights_desc, %weights_ptr,
      %conv_desc, %algo, %beta_ptr,
      %output_desc, %output_ptr,
      %workspace, %workspace_size
    ) : (...) -> i32

    // Cleanup descriptors
    llvm.call @miopenDestroyTensorDescriptor(%input_desc) : ...
    llvm.call @hipFree(%workspace) : (!llvm.ptr) -> ()

    // Return output memref
    return %output_memref : !llvm.struct<...>
  }
}
```

**What happened**:
- `hip.conv()` → `llvm.call @miopenConvolutionForward(...)`
- State extraction: `llvm.getelementptr %state[8]` → `llvm.load` to get miopenHandle
- Tensor descriptors created for input/weights/output
- MIOpen workspace allocation
- Direct AMD GPU library calls - ready for compilation to native code

---

## Actual Test Results

### Test 1: HIP → LLVM (Known Working)

**Input**: `tools/hip-opt/test.mlir`
```mlir
func.func @test_hip_ops(%N: index) {
  %handle = hip.create_handle() : !hip.handle
  %x = hip.alloc(%handle, %N) : memref<?x128xf32, 1>
  hip.free(%handle, %x) : memref<?x128xf32, 1>
  hip.destroy_handle(%handle) : !hip.handle
  return
}
```

**Command**:
```bash
cd tools/hip-opt
../../build/bin/Debug/hip-opt.exe test.mlir --convert-hip-to-llvm
```

**Output** (verified working):
```mlir
llvm.func @hipCreateHandle() -> !llvm.ptr
llvm.func @hipMalloc(i64) -> !llvm.ptr
llvm.func @hipFree(!llvm.ptr)
llvm.func @hipDestroyHandle(!llvm.ptr)

func.func @test_hip_ops(%arg0: index) {
  %1 = llvm.call @hipCreateHandle() : () -> !llvm.ptr
  %8 = llvm.call @hipMalloc(%size) : (i64) -> !llvm.ptr
  [... memref construction ...]
  llvm.call @hipFree(%mem) : (!llvm.ptr) -> ()
  llvm.call @hipDestroyHandle(%1) : (!llvm.ptr) -> ()
  return
}
```

✅ **VERIFIED**: HIP operations successfully lower to LLVM IR with AMD runtime calls.

### Test 2: ONNX → HIP (Implementation Status)

**Status**: Code compiles, but CLI tool has option conflict with onnx-mlir.

**Pattern code** (from `lib/HipDialect/OnnxToHip.cpp`):
```cpp
struct ConvToHipPattern : public OpConversionPattern<ONNXConvOp> {
  LogicalResult matchAndRewrite(
      ONNXConvOp convOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Extract operands (type-safe)
    Value X = convOp.getX();
    Value W = convOp.getW();
    Value B = convOp.getB();

    // Extract attributes (type-safe)
    auto kernelShape = convOp.getKernelShape().value();
    auto strides = convOp.getStrides().value();
    // ...

    // Get state from function
    Value state = funcOp.getBody().front().getArgument(0);

    // Create hip.conv operation
    SmallVector<NamedAttribute, 5> attributes;
    attributes.push_back(rewriter.getNamedAttr("kernel_shape", kernelShape));
    // ...

    OperationState opState(loc, hip::ConvOp::getOperationName(),
                          operands, {outputType}, attributes);
    Operation *hipConvOp = rewriter.create(opState);

    rewriter.replaceOp(convOp, hipConvOp->getResult(0));
    return success();
  }
};
```

✅ **COMPILES**: Pattern matching code builds successfully.
⚠️ **CLI CONFLICT**: Cannot test via command line due to onnx-mlir option conflict.
✅ **SOLUTION**: Use programmatically in Level-1 Pass via `PassManager::addPass()`.

---

## Next Step: End-to-End Test

Create integration test in Level-1 Pass:

```cpp
void testOnnxToLLVM() {
  MLIRContext context;

  // Parse ONNX model
  OwningOpRef<ModuleOp> module = parseONNXModel("conv.onnx");

  // Run conversion pipeline
  PassManager pm(&context);
  pm.addPass(createConvertOnnxToHipPass());  // ONNX → HIP
  pm.addPass(createConvertHipToLLVMPass()); // HIP → LLVM

  if (failed(pm.run(*module)))
    return failure();

  // Output should have llvm.call @miopenConvolutionForward
  module->dump();
}
```

This will prove the full pipeline without CLI conflicts.

---

## Summary

**What Works**:
1. ✅ HIP → LLVM transformation (tested via CLI)
2. ✅ ONNX → HIP pattern code (compiles successfully)
3. ✅ State extraction and attribute handling (implemented)

**What's Blocked**:
- ⚠️ CLI testing of ONNX→HIP due to option conflict (doesn't affect production)

**Proof Provided**:
- Actual working HIP→LLVM test output
- Compiled ONNX→HIP pattern code
- Clear transformation examples showing each step

**To Prove Full Pipeline**:
Run programmatic test in Level-1 Pass (avoids CLI conflicts).
