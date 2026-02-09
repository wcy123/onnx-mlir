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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

// Include ONNX dialect operations from onnx-mlir
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// ONNX Conv → HIP Conv Conversion Pattern
//===----------------------------------------------------------------------===//

struct ConvToHipPattern : public OpConversionPattern<ONNXConvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ONNXConvOp convOp,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Get location for error reporting
    auto loc = convOp.getLoc();

    // ✅ Type-safe operand access (self-documenting)
    Value X = convOp.getX();      // Input tensor: [N, C_in, H, W]
    Value W = convOp.getW();      // Weight tensor: [C_out, C_in/group, Kh, Kw]
    Value B = convOp.getB();      // Bias tensor: [C_out] (optional)

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

    // Get output type (already computed by ONNX shape inference)
    auto outputType = convOp.getResult().getType();

    // Get state from function argument
    // The compiled function signature is:
    //   func @inference_compute(%state: !hip.handle, %inputs: !llvm.ptr, %outputs: !llvm.ptr) -> i32
    //
    // Phase 1 Design:
    // - We use !hip.handle type for state parameter (simple, type-safe at HIP dialect level)
    // - The !hip.handle actually points to the State struct (documented semantic)
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

    // First argument should be the state (typed as !hip.handle for now)
    auto &entryBlock = funcOp.getBody().front();
    if (entryBlock.getNumArguments() == 0) {
      return rewriter.notifyMatchFailure(convOp, "Function has no arguments (expected state as first arg)");
    }

    Value state = entryBlock.getArgument(0);

    // Verify it's a handle type (in Phase 1, state is represented as !hip.handle)
    if (!state.getType().isa<hip::HandleType>()) {
      return rewriter.notifyMatchFailure(convOp, "First function argument is not a !hip.handle (expected state)");
    }

    // Pass state directly to hip.conv
    // The hip.conv operation will use this state to access miopenHandle during HIP→LLVM lowering
    Value handle = state;

    // Create HIP Conv operation using builder
    auto hipConv = rewriter.create<hip::ConvOp>(
        loc,                   // Location
        outputType,            // Result type
        handle,                // HIP handle from function argument
        X,                     // Input
        W,                     // Weights
        B,                     // Bias (may be empty)
        kernelShapeAttr,       // kernel_shape
        stridesAttr,           // strides
        padsAttr,              // pads
        dilationsAttr,         // dilations
        groupAttr);            // group

    // Replace the ONNX Conv with HIP Conv result
    rewriter.replaceOp(convOp, hipConv.getResult());

    return success();
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

  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *context = &getContext();

    // Set up conversion target
    ConversionTarget target(*context);

    // Mark HIP dialect as legal
    target.addLegalDialect<hip::HipDialect>();

    // Mark Func dialect as legal (we don't convert function ops)
    target.addLegalDialect<func::FuncDialect>();

    // Mark ONNX Conv as illegal (must be lowered)
    target.addIllegalOp<ONNXConvOp>();

    // All other ONNX ops are legal for now (only converting Conv)
    target.addLegalDialect<ONNXDialect>();

    // Set up rewrite patterns
    RewritePatternSet patterns(context);
    patterns.add<ConvToHipPattern>(context);

    // Apply conversion
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
