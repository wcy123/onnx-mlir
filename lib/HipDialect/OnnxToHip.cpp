/**
 ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 ** Licensed under the MIT License.
 **/

//===----------------------------------------------------------------------===//
// ONNX to HIP Dialect Conversion
//===----------------------------------------------------------------------===//
// This file implements conversion patterns from ONNX dialect operations
// (provided by onnx-mlir) to HIP dialect operations (using MIOpen).
//===----------------------------------------------------------------------===//

#include "HipDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

// Include ONNX dialect operations from onnx-mlir
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// ONNX Conv → HIP Conv Conversion Pattern (In-Place Semantics)
//===----------------------------------------------------------------------===//

struct ConvToHipPattern : public OpConversionPattern<ONNXConvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ONNXConvOp convOp,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Get location for error reporting
    auto loc = convOp.getLoc();

    // Get operands from adaptor
    // OpAdaptor provides operands after type conversion (tensor → memref)
    Value X = adaptor.getX();
    Value W = adaptor.getW();
    Value B = adaptor.getB();

    // ✅ Type-safe attribute access (compile-time checked)
    auto kernelShape = convOp.getKernelShape();
    auto strides = convOp.getStrides();
    auto pads = convOp.getPads();
    auto dilations = convOp.getDilations();
    auto group = convOp.getGroup();

    // Extract attribute values
    if (!kernelShape || !strides || !pads || !dilations) {
      return rewriter.notifyMatchFailure(
          convOp, "Conv operation missing required attributes");
    }

    // Convert ArrayAttr to I64ArrayAttr for HIP dialect
    auto kernelShapeAttr = kernelShape;
    auto stridesAttr = strides;
    auto padsAttr = pads;
    auto dilationsAttr = dilations;
    auto groupAttr = rewriter.getI64IntegerAttr(group);

    // Get output type from ONNX operation (tensor type)
    auto onnxOutputType = convOp.getResult().getType();

    // Convert output type: tensor<...> → memref<..., 1> (GPU address space)
    auto outputMemRefType = getTypeConverter()->convertType(onnxOutputType);
    if (!outputMemRefType) {
      return rewriter.notifyMatchFailure(
          convOp, "Failed to convert output tensor type to memref");
    }

    // Verify the converted type is actually a MemRefType
    if (!isa<MemRefType>(outputMemRefType)) {
      return rewriter.notifyMatchFailure(
          convOp, "Converted output type is not a MemRefType");
    }

    // Get state from function argument
    // The compiled function signature is:
    //   func @inference_compute(%state: !hip.context, %inputs: !llvm.ptr, %outputs: !llvm.ptr) -> i32
    //
    // Phase 1 Design:
    // - We use !hip.context type for state parameter (simple, type-safe at HIP dialect level)
    // - The !hip.context actually points to the State struct (documented semantic)
    // - Handle extraction (to get miopenHandle/hipblasHandle) happens in HIP→LLVM lowering
    //
    // State struct layout (used by HipToLLVM.cpp):
    //   struct State {
    //     hipStream_t stream;              // offset 0 (8 bytes)
    //     miopenHandle_t miopenHandle;     // offset 8 (8 bytes)  ← used by hip.conv
    //     hipblasLtHandle_t hipblasHandle; // offset 16 (8 bytes) ← used by hip.gemm
    //     void** gpu_weights;              // offset 24 (8 bytes)
    //   };
    //
    // Phase 2 TODO: Define high-level state type: !hip.state<...> for better type safety
    auto funcOp = convOp->getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return rewriter.notifyMatchFailure(convOp, "Not inside a function");
    }

    // First argument should be the state (typed as !hip.context for now)
    auto &entryBlock = funcOp.getBody().front();
    if (entryBlock.getNumArguments() == 0) {
      return rewriter.notifyMatchFailure(convOp, "Function has no arguments (expected state as first arg)");
    }

    Value state = entryBlock.getArgument(0);

    // Verify it's a handle type (in Phase 1, state is represented as !hip.context)
    if (!isa<hip::ContextType>(state.getType())) {
      return rewriter.notifyMatchFailure(convOp, "First function argument is not a !hip.context (expected state)");
    }

    // Pass state directly to hip.conv
    // The hip.conv operation will use this state to access miopenHandle during HIP→LLVM lowering
    Value handle = state;

    // ⭐ IN-PLACE SEMANTICS (Phase 1: Naive inline allocation)
    // Allocate output buffer on GPU using hip.alloc
    // Phase 2 TODO: Hoist this allocation to inference_init() for 4-12x speedup
    // Phase 3 TODO: Use memory pooling to reduce memory footprint by 60-70%

    // Extract dynamic sizes if the output memref has dynamic dimensions
    SmallVector<Value> dynamicSizes;
    auto memRefType = cast<MemRefType>(outputMemRefType);
    for (int64_t i = 0; i < memRefType.getRank(); ++i) {
      if (memRefType.isDynamicDim(i)) {
        // Get dimension size from input (assumes ONNX shape inference succeeded)
        Value dimSize = rewriter.create<memref::DimOp>(loc, X, i);
        dynamicSizes.push_back(dimSize);
      }
    }

    // Allocate GPU memory for output
    auto outputBuffer = rewriter.create<hip::AllocOp>(
        loc, outputMemRefType, handle, dynamicSizes);

    // Create HIP Conv operation (in-place: writes to pre-allocated output buffer)
    // Signature: hip.conv(%handle, %input, %weights, %bias?, %output)
    SmallVector<Value, 5> operands = {handle, X, W};
    if (B) {
      operands.push_back(B);
    }
    operands.push_back(outputBuffer.getResult());  // ⭐ Output buffer as argument

    // Prepare attributes (unwrap optional values)
    SmallVector<NamedAttribute, 5> attributes;
    attributes.push_back(rewriter.getNamedAttr("kernel_shape", kernelShapeAttr.value()));
    attributes.push_back(rewriter.getNamedAttr("strides", stridesAttr.value()));
    attributes.push_back(rewriter.getNamedAttr("pads", padsAttr.value()));
    attributes.push_back(rewriter.getNamedAttr("dilations", dilationsAttr.value()));
    attributes.push_back(rewriter.getNamedAttr("group", groupAttr));

    // Build the in-place operation (no results!)
    OperationState opState(loc, hip::ConvOp::getOperationName(),
                          operands, {}, attributes);  // ⭐ Empty result types

    rewriter.create(opState);

    // ⭐ Replace ONNX Conv result with the allocated output buffer
    // Users of the original ONNX result will now use the output buffer
    rewriter.replaceOp(convOp, outputBuffer.getResult());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Func Return Conversion Pattern
//===----------------------------------------------------------------------===//
// Convert func.return to use converted operand types (memref instead of tensor)

struct ReturnOpConversion : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::ReturnOp returnOp,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Replace return with converted operands
    // OpAdaptor provides operands after type conversion (tensor → memref)
    rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Type Converter: Tensor → MemRef (GPU Address Space)
//===----------------------------------------------------------------------===//
//
// TypeConverter provides systematic type conversion for dialect lowering.
// It converts ONNX tensor types to HIP memref types with GPU address space.
//
// Example conversion:
//   tensor<1x3x224x224xf32> → memref<1x3x224x224xf32, 1>
//                                                     ↑
//                                        Address space 1 = GPU memory
//
// Why address space 1?
// - Address space 0: CPU memory (default for memref)
// - Address space 1: GPU memory (AMD ROCm convention)
// - This ensures correct memory allocation (hipMalloc vs malloc)
//
class OnnxToHipTypeConverter : public TypeConverter {
public:
  OnnxToHipTypeConverter() {
    // Rule 1: Convert RankedTensorType to MemRefType with GPU address space
    // This rule MUST be added first before the identity conversion
    addConversion([](RankedTensorType type) -> Type {
      // Extract tensor properties
      auto shape = type.getShape();
      auto elementType = type.getElementType();

      // Create memref type with address space 1 (GPU memory)
      // Use default (identity) layout and GPU memory space
      auto memSpace = IntegerAttr::get(
          IntegerType::get(type.getContext(), 64), 1);
      return MemRefType::get(shape, elementType,
                            AffineMap(),  // Default (identity) layout
                            memSpace);
    });

    // Rule 2: Keep MemRefType unchanged (already converted or GPU types)
    addConversion([](MemRefType type) -> Type {
      return type;
    });

    // Rule 3: Keep HIP types unchanged
    addConversion([](hip::ContextType type) -> Type {
      return type;
    });

    // Rule 4: Keep scalar types unchanged (i64, f32, etc.)
    // Only convert types not covered by specific rules above
    addConversion([](Type type) -> std::optional<Type> {
      // If it's a tensor type that wasn't handled by Rule 1, fail
      if (isa<TensorType>(type)) {
        return std::nullopt;  // Conversion failed
      }
      // For all other types, keep unchanged
      return type;
    });

    // Register materialization hooks (required by MLIR infrastructure)
    // These handle edge cases where type conversions need temporary values

    // Source materialization: Create a value of the original type from converted type
    // (e.g., when converting memref back to tensor for unconverted operations)
    addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return nullptr;
      // Create unrealized_conversion_cast to bridge type mismatch
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });

    // Target materialization: Create a value of the converted type from original type
    // (e.g., when an operation needs a converted type but gets unconverted input)
    addTargetMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return nullptr;
      // Create unrealized_conversion_cast to bridge type mismatch
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
  }
};

//===----------------------------------------------------------------------===//
// ONNX to HIP Conversion Pass
//===----------------------------------------------------------------------===//

class ConvertOnnxToHipPass
    : public PassWrapper<ConvertOnnxToHipPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertOnnxToHipPass)

  StringRef getArgument() const final { return "convert-onnx-to-hip"; }
  StringRef getDescription() const final {
    return "Convert ONNX dialect operations to HIP dialect operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hip::HipDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();  // Needed for memref.dim
  }

  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *context = &getContext();

    // Step 1: Set up TypeConverter (tensor → memref with GPU address space)
    OnnxToHipTypeConverter typeConverter;

    // Step 2: Add %ctx: !hip.context parameter to function if not present
    // Do this BEFORE conversion so patterns see the correct function signature
    auto &entryBlock = func.getBody().front();
    bool hasContext = false;
    if (entryBlock.getNumArguments() > 0) {
      // Check if first argument is already a context
      if (isa<hip::ContextType>(entryBlock.getArgument(0).getType())) {
        hasContext = true;
      }
    }

    if (!hasContext) {
      // Insert context parameter as first argument
      OpBuilder builder(context);
      auto contextType = hip::ContextType::get(context);

      // Insert block argument at position 0
      entryBlock.insertArgument(0u, contextType, func.getLoc());

      // Update function type to include new parameter
      auto funcType = func.getFunctionType();
      SmallVector<Type, 4> newInputs;
      newInputs.push_back(contextType);

      // Convert remaining input types through TypeConverter
      for (Type inputType : funcType.getInputs()) {
        Type convertedType = typeConverter.convertType(inputType);
        newInputs.push_back(convertedType ? convertedType : inputType);
      }

      // Convert result types through TypeConverter
      SmallVector<Type, 4> newResults;
      for (Type resultType : funcType.getResults()) {
        Type convertedType = typeConverter.convertType(resultType);
        newResults.push_back(convertedType ? convertedType : resultType);
      }

      auto newFuncType = builder.getFunctionType(newInputs, newResults);
      func.setFunctionType(newFuncType);

      // Update block argument types (except context which we just added)
      for (unsigned i = 1; i < entryBlock.getNumArguments(); ++i) {
        Type oldType = entryBlock.getArgument(i).getType();
        Type newType = typeConverter.convertType(oldType);
        if (newType && newType != oldType) {
          entryBlock.getArgument(i).setType(newType);
        }
      }
    }

    // Step 3: Set up conversion target
    ConversionTarget target(*context);

    // Mark HIP dialect as legal
    target.addLegalDialect<hip::HipDialect>();

    // Mark HIP dialect as legal
    target.addLegalDialect<hip::HipDialect>();

    // Mark Func dialect as legal EXCEPT func.return which we need to convert
    target.addLegalDialect<func::FuncDialect>();
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      // func.return is legal only if all operands are already converted types
      return llvm::all_of(op.getOperandTypes(), [&](Type type) {
        return typeConverter.isLegal(type);
      });
    });

    // Mark MemRef dialect as legal (we generate memref.dim for dynamic shapes)
    target.addLegalDialect<memref::MemRefDialect>();

    // Mark ONNX Conv as illegal (must be lowered)
    target.addIllegalOp<ONNXConvOp>();

    // All other ONNX ops are legal for now (only converting Conv)
    target.addLegalDialect<ONNXDialect>();

    // Step 4: Set up rewrite patterns (pass typeConverter to patterns)
    RewritePatternSet patterns(context);
    patterns.add<ConvToHipPattern>(typeConverter, context);
    patterns.add<ReturnOpConversion>(typeConverter, context);

    // Step 5: Apply conversion
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace hip {

std::unique_ptr<Pass> createConvertOnnxToHipPass() {
  return std::make_unique<ConvertOnnxToHipPass>();
}

} // namespace hip
} // namespace mlir
