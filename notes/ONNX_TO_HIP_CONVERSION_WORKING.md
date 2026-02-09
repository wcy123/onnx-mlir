# ONNX→HIP Conversion Implementation - Working!

**Date:** 2026-02-09
**Status:** ✅ Working - Destination-passing style with in-place operations

---

## Summary

Successfully implemented ONNX to HIP dialect conversion with proper type conversion and destination-passing function semantics. The conversion correctly transforms ONNX Conv operations to HIP Conv operations with GPU memory allocation and in-place execution.

---

## What Works

### Type Conversion
- ✅ `tensor<1x3x224x224xf32>` → `memref<1x3x224x224xf32, 1>` (GPU address space)
- ✅ Function signature conversion (inputs converted, outputs as arguments)
- ✅ Block argument type updates
- ✅ Return value changed to i32 status code

### Function Signature Transformation
- ✅ Add `!hip.context` as first parameter
- ✅ Convert input tensor types to memref types
- ✅ **Add output arguments** (destination-passing style)
- ✅ Change return type to i32 (status code)

### Operation Conversion
- ✅ ONNX Conv → HIP Conv with in-place semantics
- ✅ Context parameter insertion (!hip.context as first arg)
- ✅ Attribute extraction (kernel_shape, strides, pads, dilations, group)
- ✅ Output buffer allocation using hip.alloc
- ✅ memref.copy to write results to output arguments

### Example

**Input:**
```mlir
func.func @main(%input: tensor<1x3x224x224xf32>,
                %weights: tensor<64x3x3x3xf32>,
                %bias: tensor<64xf32>) -> tensor<1x64x224x224xf32> {
  %output = "onnx.Conv"(%input, %weights, %bias) {
    kernel_shape = [3, 3],
    strides = [1, 1],
    pads = [1, 1, 1, 1],
    dilations = [1, 1],
    group = 1 : si64
  } : (tensor<1x3x224x224xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>)
      -> tensor<1x64x224x224xf32>
  return %output : tensor<1x64x224x224xf32>
}
```

**Output:**
```mlir
func.func @main(%arg0: !hip.context,
                %arg1: memref<1x3x224x224xf32, 1>,
                %arg2: memref<64x3x3x3xf32, 1>,
                %arg3: memref<64xf32, 1>,
                %arg4: memref<1x64x224x224xf32, 1>) -> i32 {
  // Allocate intermediate buffer (Phase 1)
  %0 = hip.alloc(%arg0) : memref<1x64x224x224xf32, 1>

  // Execute convolution in-place (writes to %0)
  hip.conv(%arg0, %arg1, %arg2, %arg3, %0)
    {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
     pads = [1, 1, 1, 1], strides = [1, 1]}
    : (!hip.context, memref<1x3x224x224xf32, 1>,
       memref<64x3x3x3xf32, 1>, memref<64xf32, 1>,
       memref<1x64x224x224xf32, 1>)

  // Copy result to output argument (destination-passing)
  memref.copy %0, %arg4 : memref<1x64x224x224xf32, 1> to memref<1x64x224x224xf32, 1>

  // Return success status
  %c0_i32 = arith.constant 0 : i32
  return %c0_i32 : i32
}
```

---

## Implementation Components

### 1. OnnxToHipTypeConverter

**Purpose:** Systematic type conversion for dialect lowering

**Conversion Rules:**
```cpp
// Rule 1: RankedTensorType → MemRefType (GPU address space 1)
addConversion([](RankedTensorType type) -> Type {
  auto memSpace = IntegerAttr::get(IntegerType::get(ctx, 64), 1);
  return MemRefType::get(type.getShape(), type.getElementType(),
                         AffineMap(), memSpace);
});

// Rule 2: Keep MemRefType unchanged
addConversion([](MemRefType type) -> Type { return type; });

// Rule 3: Keep HIP types unchanged
addConversion([](hip::ContextType type) -> Type { return type; });

// Rule 4: Identity for other types, fail for unhandled tensors
addConversion([](Type type) -> std::optional<Type> {
  if (isa<TensorType>(type)) return std::nullopt;
  return type;
});
```

### 2. ConvToHipPattern

**Purpose:** Convert ONNX Conv to HIP Conv with in-place semantics

**Steps:**
1. Get type-converted operands from OpAdaptor (X, W, B)
2. Extract ONNX attributes (kernel_shape, strides, pads, dilations, group)
3. Convert output type: tensor → memref<..., 1>
4. Get !hip.context from first function argument
5. Allocate output buffer: `hip.alloc(%ctx) : memref<...>`
6. Create in-place hip.conv: `hip.conv(%ctx, %X, %W, %B, %output)`
7. Replace ONNX Conv result with allocated buffer

**Key Design:**
- Operations use in-place semantics (output as argument, no return)
- Phase 1: Allocate inline with hip.alloc
- Phase 2 TODO: Use pre-allocated buffers from state

### 3. Function Signature Transformation

**Purpose:** Transform function to destination-passing style

**Implementation (in runOnOperation):**
```cpp
// Add context parameter
entryBlock.insertArgument(0u, contextType, func.getLoc());

// Convert input types
for (Type inputType : funcType.getInputs()) {
  Type convertedType = typeConverter.convertType(inputType);
  newInputs.push_back(convertedType);
}

// Add output arguments (destination-passing!)
for (Type resultType : funcType.getResults()) {
  Type convertedType = typeConverter.convertType(resultType);
  newInputs.push_back(convertedType);
  entryBlock.addArgument(convertedType, func.getLoc());
}

// Return type is i32 (status code)
newResults.push_back(builder.getI32Type());
```

### 4. ReturnOpConversion

**Purpose:** Convert func.return to destination-passing style

**Implementation:**
```cpp
struct ReturnOpConversion : public OpConversionPattern<func::ReturnOp> {
  LogicalResult matchAndRewrite(
      func::ReturnOp returnOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto funcOp = returnOp->getParentOfType<func::FuncOp>();
    auto &entryBlock = funcOp.getBody().front();
    unsigned numResults = returnOp.getNumOperands();
    unsigned numArgs = entryBlock.getNumArguments();

    // Copy return values to output arguments
    for (unsigned i = 0; i < numResults; ++i) {
      Value returnValue = adaptor.getOperands()[i];
      Value outputArg = entryBlock.getArgument(numArgs - numResults + i);
      rewriter.create<memref::CopyOp>(loc, returnValue, outputArg);
    }

    // Return success status (i32 0)
    Value successStatus = rewriter.create<arith::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp, successStatus);
    return success();
  }
};
```

### 5. ConvertOnnxToHipPass

**Purpose:** Orchestrate the conversion

**Steps:**
1. Insert !hip.context as first function parameter
2. Convert function signature types (inputs converted, outputs as arguments)
3. Change return type to i32
4. Update block argument types
5. Set up conversion target:
   - Mark HIP dialect as legal
   - Mark func.return as dynamically legal (only if operands are legal types)
   - Mark ONNX Conv as illegal (must be lowered)
6. Apply conversion patterns

**Legality Rules:**
```cpp
target.addLegalDialect<hip::HipDialect>();
target.addLegalDialect<func::FuncDialect>();
target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
  return llvm::all_of(op.getOperandTypes(), [&](Type type) {
    return typeConverter.isLegal(type);
  });
});
target.addIllegalOp<ONNXConvOp>();
```

---

## Design: Destination-Passing at All Levels

**ONNX dialect** (input):
- Value semantics: functions return tensor results
- Example: `func.func @main(...) -> tensor<...>`

**HIP dialect** (after conversion):
- Destination-passing: outputs as arguments, return i32 status
- Example: `func.func @main(%ctx: !hip.context, %input: memref<...>, %output: memref<...>) -> i32`
- Operations: in-place semantics (output buffer as argument)

**LLVM dialect** (future):
- Destination-passing: same concept with LLVM types
- Example: `func.func @inference_compute(%state: !llvm.ptr, %inputs: !llvm.ptr, %outputs: !llvm.ptr) -> i32`

**C interface** (final):
- Destination-passing: outputs via span_t
- Example: `int inference_compute(void* state, span_t inputs, span_t outputs)`

---

## Testing

**Test file:** `tools/hip-opt/test_conv_inplace.mlir`

**Command:**
```bash
hip-opt --convert-onnx-to-hip test_conv_inplace.mlir
```

**Result:** Clean conversion with destination-passing function signature, no errors.

---

## Next Steps

### Immediate (High Priority)

1. **Update HipToLLVM.cpp** for destination-passing functions
   - Handle function signature (already takes %inputs, %outputs)
   - No changes needed for operations (already in-place)

2. **Add more operation patterns**
   - GemmToHipPattern (for matrix multiplication)
   - MaxPoolToHipPattern (for max pooling)
   - AvgPoolToHipPattern (for average pooling)
   - ReLUToHipPattern (for activation)

### Phase 2 (Critical Performance)

3. **Implement allocation hoisting**
   - Track all hip.alloc operations during conversion
   - Generate inference_init() that pre-allocates all buffers
   - Replace hip.alloc in inference_compute() with loads from state
   - **Expected: 4-12x speedup**

### Future Optimizations

4. **Memory pooling** (Phase 3)
   - Liveness analysis for intermediate tensors
   - Memory layout optimization
   - **Expected: 60-70% memory savings**

5. **Descriptor caching**
   - Create MIOpen descriptors in inference_init()
   - Reuse in inference_compute()
   - **Expected: 10-20% speedup**

---

## Key Learnings

### Destination-Passing Transformation
- Outputs become function arguments (not return values)
- Return type changes to i32 (status code)
- Operations still use in-place semantics (unchanged)
- memref.copy writes intermediate results to output arguments

### TypeConverter Rule Ordering
- Specific conversions (RankedTensorType) must come BEFORE generic fallback
- Otherwise the fallback rule matches everything and prevents conversion

### OpAdaptor Provides Converted Operands
- `adaptor.getX()` returns operands AFTER type conversion
- Don't manually convert operands that OpAdaptor already handles

### Materialization is Required
- TypeConverter needs source/target materialization hooks
- Use UnrealizedConversionCastOp for bridging type mismatches

### Dynamic Legality for Partial Conversion
- Use `addDynamicallyLegalOp` when an operation is legal under certain conditions
- Example: func.return is legal only if operands are already converted types

---

## References

- **Implementation:** `lib/HipDialect/OnnxToHip.cpp`
- **Test:** `tools/hip-opt/test_conv_inplace.mlir`
- **Design:** `doc/MEMORY-MANAGEMENT.md` (Destination-Passing Design section)
- **MLIR Dialect Conversion:** https://mlir.llvm.org/docs/DialectConversion/

---

**Status:** ✅ ONNX→HIP conversion working with destination-passing style
**Next:** Update test files and documentation to match new design
