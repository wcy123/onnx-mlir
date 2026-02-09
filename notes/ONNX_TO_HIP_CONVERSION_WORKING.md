# ONNX→HIP Conversion Implementation - Working!

**Date:** 2026-02-09
**Status:** ✅ Working - Basic Conv operation conversion complete

---

## Summary

Successfully implemented ONNX to HIP dialect conversion with proper type conversion and in-place semantics. The conversion correctly transforms ONNX Conv operations to HIP Conv operations with GPU memory allocation.

---

## What Works

### Type Conversion
- ✅ `tensor<1x3x224x224xf32>` → `memref<1x3x224x224xf32, 1>` (GPU address space)
- ✅ Function signature conversion (inputs and results)
- ✅ Block argument type updates
- ✅ Return value type conversion

### Operation Conversion
- ✅ ONNX Conv → HIP Conv with in-place semantics
- ✅ Context parameter insertion (!hip.context as first arg)
- ✅ Attribute extraction (kernel_shape, strides, pads, dilations, group)
- ✅ Output buffer allocation using hip.alloc

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
                %arg3: memref<64xf32, 1>) -> memref<1x64x224x224xf32, 1> {
  %0 = hip.alloc(%arg0) : memref<1x64x224x224xf32, 1>
  hip.conv(%arg0, %arg1, %arg2, %arg3, %0)
    {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
     pads = [1, 1, 1, 1], strides = [1, 1]}
    : (!hip.context, memref<1x3x224x224xf32, 1>,
       memref<64x3x3x3xf32, 1>, memref<64xf32, 1>,
       memref<1x64x224x224xf32, 1>)
  return %0 : memref<1x64x224x224xf32, 1>
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

**Materialization:**
- Uses `UnrealizedConversionCastOp` for source/target materialization
- Required by MLIR conversion framework
- Handles type mismatches during incremental conversion

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
- Operations use in-place semantics (output as argument)
- Functions use value semantics (return memref)
- Allocation happens inline (Phase 1 - naive)

### 3. ReturnOpConversion

**Purpose:** Convert func.return to use converted types

**Implementation:**
```cpp
struct ReturnOpConversion : public OpConversionPattern<func::ReturnOp> {
  LogicalResult matchAndRewrite(
      func::ReturnOp returnOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Replace with converted operands (tensor → memref)
    rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp, adaptor.getOperands());
    return success();
  }
};
```

### 4. ConvertOnnxToHipPass

**Purpose:** Orchestrate the conversion

**Steps:**
1. Insert !hip.context as first function parameter
2. Convert function signature types (inputs and results)
3. Update block argument types
4. Set up conversion target:
   - Mark HIP dialect as legal
   - Mark func.return as dynamically legal (only if operands are legal types)
   - Mark ONNX Conv as illegal (must be lowered)
5. Apply conversion patterns

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

## Testing

**Test file:** `tools/hip-opt/test_conv_inplace.mlir`

**Command:**
```bash
hip-opt --convert-onnx-to-hip test_conv_inplace.mlir
```

**Result:** Clean conversion, no errors, correct types and operations.

---

## Next Steps

### Immediate (High Priority)

1. **Update HipToLLVM.cpp** for in-place hip.conv
   - Current: Expects hip.conv to return result
   - Needed: Handle hip.conv with output argument, no result
   - Update miopenConvolutionForward call generation

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

### TypeConverter Rule Ordering Matters
- Specific conversions (RankedTensorType) must come BEFORE generic fallback
- Otherwise the fallback rule matches everything and prevents conversion

### OpAdaptor Provides Converted Operands
- `adaptor.getX()` returns operands AFTER type conversion
- Don't manually convert operands that OpAdaptor already handles

### Materialization is Required
- TypeConverter needs source/target materialization hooks
- Use UnrealizedConversionCastOp for bridging type mismatches
- The framework will resolve or error if casts remain after full conversion

### Dynamic Legality for Partial Conversion
- Use `addDynamicallyLegalOp` when an operation is legal under certain conditions
- Example: func.return is legal only if operands are already converted types

---

## References

- **Implementation:** `lib/HipDialect/OnnxToHip.cpp`
- **Test:** `tools/hip-opt/test_conv_inplace.mlir`
- **Design:** `doc/MEMORY-MANAGEMENT.md` (In-Place Semantics Design section)
- **MLIR Dialect Conversion:** https://mlir.llvm.org/docs/DialectConversion/

---

**Status:** ✅ ONNX→HIP conversion working for Conv operation
**Next:** Update HipToLLVM.cpp for in-place operations
