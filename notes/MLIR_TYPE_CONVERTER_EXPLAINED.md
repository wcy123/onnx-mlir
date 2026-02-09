# MLIR TypeConverter Explained

**Date:** 2026-02-09
**Context:** Understanding type conversion in ONNX→HIP dialect lowering

---

## The Problem TypeConverter Solves

When converting operations between dialects, you often need to change types. For example:

**Before conversion (ONNX):**
```mlir
func.func @main(%input: tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32> {
  %result = "onnx.Conv"(%input, ...) : (tensor<...>) -> tensor<1x64x112x112xf32>
  return %result : tensor<1x64x112x112xf32>
}
```

**After conversion (HIP):**
```mlir
func.func @main(%input: memref<1x3x224x224xf32, 1>) -> memref<1x64x112x112xf32, 1> {
  %result = hip.conv(%ctx, %input, ...) : (...) -> memref<1x64x112x112xf32, 1>
  return %result : memref<1x64x112x112xf32, 1>
}
```

Notice what changed:
- Function signature: `tensor<...>` → `memref<..., 1>`
- Operation result type: `tensor<...>` → `memref<..., 1>`
- Return type: `tensor<...>` → `memref<..., 1>`

**The Challenge:** How do you keep track of all these type changes consistently?

This is what `TypeConverter` does - it's basically a **type mapping table** plus **conversion rules**.

---

## How TypeConverter Works (Simple Explanation)

Think of `TypeConverter` as a dictionary:

```cpp
TypeConverter typeConverter;

// Register conversion rule: tensor<...> → memref<..., 1>
typeConverter.addConversion([](RankedTensorType tensorType) -> Type {
  // tensor<1x3x224x224xf32> → memref<1x3x224x224xf32, 1>
  return MemRefType::get(
    tensorType.getShape(),       // Keep shape [1, 3, 224, 224]
    tensorType.getElementType(), // Keep element type f32
    MemRefLayoutAttrInterface(), // Default layout
    IntegerAttr::get(..., 1)     // Address space 1 = GPU
  );
});
```

Now whenever MLIR sees a `tensor<...>` during conversion, it automatically converts to `memref<..., 1>`.

---

## Step-by-Step Example: How Conversion Patterns Use TypeConverter

### Step 1: Define the TypeConverter

```cpp
class TensorToMemRefConverter : public TypeConverter {
public:
  TensorToMemRefConverter() {
    // Rule 1: Keep non-tensor types unchanged
    addConversion([](Type type) {
      return type;
    });

    // Rule 2: Convert tensor → memref (GPU address space)
    addConversion([](RankedTensorType tensorType) -> Type {
      auto shape = tensorType.getShape();
      auto elemType = tensorType.getElementType();

      // Create memref with address space 1 (GPU)
      return MemRefType::get(shape, elemType, {}, 1);
    });
  }
};
```

### Step 2: Use TypeConverter in Conversion Pattern

```cpp
struct ConvToHipPattern : public OpConversionPattern<ONNXConvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ONNXConvOp convOp,
      OpAdaptor adaptor,  // ← IMPORTANT: adaptor provides converted operands
      ConversionPatternRewriter &rewriter) const override {

    auto loc = convOp.getLoc();

    // ============================================================
    // KEY INSIGHT: adaptor.getX() returns CONVERTED operands
    // ============================================================
    // Before conversion: convOp.getX() returns Value with type tensor<...>
    // After conversion:  adaptor.getX() returns Value with type memref<..., 1>

    Value input = adaptor.getX();      // Already converted to memref!
    Value weights = adaptor.getW();    // Already converted to memref!
    Value bias = adaptor.getB();       // Already converted to memref!

    // ============================================================
    // Convert result type using TypeConverter
    // ============================================================
    // Original result type: tensor<1x64x112x112xf32>
    Type originalResultType = convOp.getType();

    // Converted result type: memref<1x64x112x112xf32, 1>
    Type convertedResultType = getTypeConverter()->convertType(originalResultType);
    auto memrefResultType = convertedResultType.cast<MemRefType>();

    // ============================================================
    // Allocate output buffer
    // ============================================================
    Value outputBuffer = rewriter.create<memref::AllocOp>(
      loc,
      memrefResultType  // memref<1x64x112x112xf32, 1>
    );

    // ============================================================
    // Get context (assume first function arg)
    // ============================================================
    auto funcOp = convOp->getParentOfType<func::FuncOp>();
    Value ctx = funcOp.getArgument(0);

    // ============================================================
    // Create HIP Conv operation
    // ============================================================
    // Now all types match: ctx, input, weights, bias, outputBuffer are all correct types
    rewriter.create<hip::ConvOp>(
      loc,
      ctx,           // !hip.context
      input,         // memref<..., 1>
      weights,       // memref<..., 1>
      bias,          // memref<..., 1>
      outputBuffer,  // memref<..., 1> - destination buffer
      convOp->getAttrs()  // Copy attributes (kernel_shape, strides, etc.)
    );

    // ============================================================
    // Replace original operation with the output buffer
    // ============================================================
    rewriter.replaceOp(convOp, outputBuffer);

    return success();
  }
};
```

### Step 3: Set Up Conversion with TypeConverter

```cpp
void ConvertOnnxToHipPass::runOnOperation() {
  auto func = getOperation();
  auto *context = &getContext();

  // Create type converter
  TensorToMemRefConverter typeConverter;

  // Set up conversion target
  ConversionTarget target(*context);
  target.addLegalDialect<hip::HipDialect>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addIllegalOp<ONNXConvOp>();

  // Add patterns with type converter
  RewritePatternSet patterns(context);
  patterns.add<ConvToHipPattern>(typeConverter, context);

  // Run conversion - TypeConverter automatically handles type changes!
  applyPartialConversion(func, target, std::move(patterns));
}
```

---

## What Happens During Conversion?

Here's the magic - MLIR's conversion framework does a lot automatically:

**Input IR:**
```mlir
func.func @main(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32> {
  %0 = "onnx.Conv"(%arg0, ...) : (tensor<...>) -> tensor<1x64x112x112xf32>
  return %0 : tensor<1x64x112x112xf32>
}
```

**During conversion, MLIR automatically:**
1. **Converts function signature** using TypeConverter rules:
   - `%arg0: tensor<...>` → `%arg0: memref<..., 1>`
   - Return type: `tensor<...>` → `memref<..., 1>`

2. **Calls your pattern** with converted operands:
   - `adaptor.getX()` gives you memref, not tensor

3. **Replaces operations** while maintaining SSA form:
   - All uses of `%0` now see the memref value

**Output IR:**
```mlir
func.func @main(%arg0: memref<1x3x224x224xf32, 1>) -> memref<1x64x112x112xf32, 1> {
  %alloc = memref.alloc() : memref<1x64x112x112xf32, 1>
  hip.conv(%ctx, %arg0, %weights, %bias, %alloc) {...}
  return %alloc : memref<1x64x112x112xf32, 1>
}
```

---

## Why This Is Better Than Manual Conversion

**Without TypeConverter (manual):**
```cpp
// You'd have to:
1. Manually track all SSA value type changes
2. Manually update function signatures
3. Manually update all uses of converted values
4. Manually handle type mismatches
// Hundreds of lines of error-prone code!
```

**With TypeConverter:**
```cpp
// You just:
1. Define type conversion rules (5 lines)
2. Write conversion patterns (focus on operation logic)
3. MLIR handles all the bookkeeping automatically
```

---

## FAQ

### Q1: Can memref represent GPU memory?

**Yes!** Memref has an **address space** parameter:

```mlir
// CPU memory (default address space 0)
memref<1x3x224x224xf32>

// GPU memory (address space 1 - common convention)
memref<1x3x224x224xf32, 1>
```

During HIP→LLVM lowering, this address space information tells the compiler:
- Use GPU memory instructions
- Generate correct pointer types for HIP API calls
- Handle address space casts when needed

### Q2: Can onnx.Conv accept memref as argument?

**No.** ONNX operations are defined to work with **tensor types only**.

Verification (actual error when attempting):
```
error: 'onnx.Conv' op operand #0 must be tensor of [...] values,
but got 'memref<1x3x224x224xf32>'
```

This is why we need ONNX→HIP conversion - we can't just bufferize ONNX operations in place.

### Q3: What is OpAdaptor?

`OpAdaptor` is a helper class that provides **type-converted** operands during pattern matching.

```cpp
LogicalResult matchAndRewrite(
    ONNXConvOp convOp,      // Original operation (tensors)
    OpAdaptor adaptor,      // Converted operands (memrefs)
    ConversionPatternRewriter &rewriter) const override {

  // convOp.getX() returns tensor<...>
  // adaptor.getX() returns memref<..., 1> (already converted!)

  Value input = adaptor.getX();  // Use adaptor, not convOp!
}
```

Without OpAdaptor, you'd have to manually track which values have been converted.

---

## Application to Our Project

**Current Issue:**
```
error: 'hip.conv' op operand #1 must be memref of any type values,
but got 'tensor<1x3x224x224xf32>'
```

**Root Cause:**
We're creating `hip.conv` with tensor operands from ONNX, but HIP operations expect memrefs.

**Solution:**
Add TypeConverter to ONNX→HIP pass:

1. Define `TensorToMemRefConverter` with rule: `tensor<...>` → `memref<..., 1>`
2. Update `ConvToHipPattern` to:
   - Use `adaptor.getX()` (memrefs) instead of `convOp.getX()` (tensors)
   - Allocate output buffer with `memref.alloc`
   - Create `hip.conv` with all memref operands
3. MLIR automatically converts function signatures

**Pipeline:**
```
ONNX (tensor) → [ONNX→HIP with TypeConverter] → HIP (memref, GPU)
```

**Benefits:**
- ✅ Single pass handles everything
- ✅ Automatic function signature conversion
- ✅ Standard MLIR pattern
- ✅ Type-safe throughout conversion

---

## References

- MLIR Dialect Conversion: https://mlir.llvm.org/docs/DialectConversion/
- MLIR TypeConverter: https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/#type-system
- Bufferization in MLIR: https://mlir.llvm.org/docs/Bufferization/

---

## Next Steps

1. Implement `TensorToMemRefConverter` in `lib/HipDialect/OnnxToHip.cpp`
2. Update `ConvToHipPattern` to use `OpAdaptor` and allocate output buffers
3. Test with `tools/hip-opt/demo_input.mlir`
4. Verify full pipeline: ONNX→HIP→LLVM
