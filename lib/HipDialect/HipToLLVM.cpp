//===- HipToLLVM.cpp - HIP to LLVM dialect conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HipDialect.h"
#include "HipPasses.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace hip {

namespace {

static constexpr const char *kHipCreateHandle = "hipCreateHandle";
static constexpr const char *kHipDestroyHandle = "hipDestroyHandle";
static constexpr const char *kHipMalloc = "hipMalloc";
static constexpr const char *kHipFree = "hipFree";
static constexpr const char *kMiopenConvolutionForward = "miopenConvolutionForward";

// --- CreateHandleOp: hip.create_handle() -> llvm.call @hipCreateHandle()
struct CreateHandleOpLowering : public ConvertOpToLLVMPattern<CreateHandleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CreateHandleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipCreateHandle, /*paramTypes=*/{}, ptrType);
    if (failed(funcOp))
      return failure();

    auto callOp = LLVM::CallOp::create(rewriter, loc, *funcOp, ValueRange());
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

// --- DestroyHandleOp: hip.destroy_handle(%h) -> llvm.call @hipDestroyHandle(%h)
struct DestroyHandleOpLowering
    : public ConvertOpToLLVMPattern<DestroyHandleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(DestroyHandleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipDestroyHandle, ptrType, voidType);
    if (failed(funcOp))
      return failure();

    LLVM::CallOp::create(rewriter, loc, *funcOp, adaptor.getHandle());
    rewriter.eraseOp(op);
    return success();
  }
};

// --- AllocOp: hip.alloc(%handle, %dyn...) -> hipMalloc(bytes) + memref descriptor
struct AllocOpLowering : public ConvertOpToLLVMPattern<AllocOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MemRefType memRefType = op.getMemref().getType();

    if (!isConvertibleAndHasIdentityMaps(memRefType))
      return rewriter.notifyMatchFailure(op, "incompatible memref type");

    // Declare hipMalloc(size: i64) -> ptr
    Type indexType = getIndexType();
    Type ptrType = getPtrType();
    FailureOr<LLVM::LLVMFuncOp> mallocFn = LLVM::lookupOrCreateFn(
        rewriter, module, kHipMalloc, indexType, ptrType);
    if (failed(mallocFn))
      return failure();

    // Compute sizes and sizeBytes (dynamic sizes are after the handle).
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value sizeBytes;
    getMemRefDescriptorSizes(loc, memRefType, adaptor.getDynamicSizes(),
                             rewriter, sizes, strides, sizeBytes, true);

    Value allocatedPtr =
        LLVM::CallOp::create(rewriter, loc, *mallocFn, sizeBytes).getResult();

    // Cast to memref address space if needed
    Type elementPtrType = getElementPtrType(memRefType);
    if (!elementPtrType)
      return rewriter.notifyMatchFailure(op, "could not compute element ptr type");
    FailureOr<unsigned> addrSpace = getTypeConverter()->getMemRefAddressSpace(memRefType);
    if (failed(addrSpace))
      return failure();
    if (cast<LLVM::LLVMPointerType>(allocatedPtr.getType()).getAddressSpace() != *addrSpace)
      allocatedPtr = rewriter.create<LLVM::AddrSpaceCastOp>(
          loc, LLVM::LLVMPointerType::get(rewriter.getContext(), *addrSpace),
          allocatedPtr);

    MemRefDescriptor desc = createMemRefDescriptor(
        loc, memRefType, allocatedPtr, allocatedPtr, sizes, strides, rewriter);
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

// --- FreeOp: hip.free(%handle, %memref) -> llvm.call @hipFree(allocated_ptr)
struct FreeOpLowering : public ConvertOpToLLVMPattern<FreeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(FreeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipFree, ptrType, voidType);
    if (failed(funcOp))
      return failure();

    Value memrefDesc = adaptor.getMemref();
    Value allocatedPtr =
        MemRefDescriptor(memrefDesc).allocatedPtr(rewriter, loc);
    // hipFree expects void*; if memref is in non-default address space, cast
    auto ptrTy = allocatedPtr.getType();
    if (cast<LLVM::LLVMPointerType>(ptrTy).getAddressSpace() != 0)
      allocatedPtr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, allocatedPtr);

    LLVM::CallOp::create(rewriter, loc, *funcOp, allocatedPtr);
    rewriter.eraseOp(op);
    return success();
  }
};

// --- ConvOp: hip.conv(%handle, %input, %weights, %bias, %output) ->
//             llvm.call @miopenConvolutionForward(...)
struct ConvOpLowering : public ConvertOpToLLVMPattern<ConvOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ConvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();

    // Phase 1: Simplified lowering
    // For now, we generate a call to a runtime function that will handle
    // the full MIOpen convolution setup (descriptors, workspace, etc.)
    //
    // Signature:
    // int miopenConvolutionForward(
    //     void* handle,           // miopenHandle from state
    //     void* input,            // input tensor data pointer
    //     void* weights,          // weights tensor data pointer
    //     void* bias,             // bias tensor data pointer (nullable)
    //     void* output,           // output tensor data pointer (in-place)
    //     int64_t kernel_h,       // kernel height
    //     int64_t kernel_w,       // kernel width
    //     int64_t stride_h,       // stride height
    //     int64_t stride_w,       // stride width
    //     int64_t pad_top,        // padding top
    //     int64_t pad_left,       // padding left
    //     int64_t pad_bottom,     // padding bottom
    //     int64_t pad_right,      // padding right
    //     int64_t dilation_h,     // dilation height
    //     int64_t dilation_w,     // dilation width
    //     int64_t group           // number of groups
    // );
    //
    // Returns: 0 on success, non-zero on error

    // Extract memref pointers (aligned pointers from descriptors)
    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      // Cast to void* (address space 0) if needed
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0) {
        ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, ptr);
      }
      return ptr;
    };

    Value handlePtr = adaptor.getHandle();
    Value inputPtr = getAlignedPtr(adaptor.getInput());
    Value weightsPtr = getAlignedPtr(adaptor.getWeights());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());

    // Handle optional bias
    Value biasPtr;
    if (adaptor.getBias()) {
      biasPtr = getAlignedPtr(adaptor.getBias());
    } else {
      // Pass null pointer if no bias
      biasPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrType);
    }

    // Extract attributes
    auto kernelShape = op.getKernelShape();
    auto strides = op.getStrides();
    auto pads = op.getPads();
    auto dilations = op.getDilations();
    auto group = op.getGroup();

    // Convert attributes to i64 constants
    Type i64Type = rewriter.getI64Type();
    auto createI64Const = [&](int64_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(value));
    };

    // Extract integer values from attributes
    auto getI64 = [](mlir::Attribute attr) -> int64_t {
      return cast<mlir::IntegerAttr>(attr).getInt();
    };

    Value kernelH = createI64Const(getI64(kernelShape[0]));
    Value kernelW = createI64Const(getI64(kernelShape[1]));
    Value strideH = createI64Const(getI64(strides[0]));
    Value strideW = createI64Const(getI64(strides[1]));
    Value padTop = createI64Const(getI64(pads[0]));
    Value padLeft = createI64Const(getI64(pads[1]));
    Value padBottom = createI64Const(getI64(pads[2]));
    Value padRight = createI64Const(getI64(pads[3]));
    Value dilationH = createI64Const(getI64(dilations[0]));
    Value dilationW = createI64Const(getI64(dilations[1]));
    Value groupVal = createI64Const(group);

    // Build function signature
    SmallVector<Type, 16> paramTypes = {
        ptrType,  // handle
        ptrType,  // input
        ptrType,  // weights
        ptrType,  // bias
        ptrType,  // output
        i64Type,  // kernel_h
        i64Type,  // kernel_w
        i64Type,  // stride_h
        i64Type,  // stride_w
        i64Type,  // pad_top
        i64Type,  // pad_left
        i64Type,  // pad_bottom
        i64Type,  // pad_right
        i64Type,  // dilation_h
        i64Type,  // dilation_w
        i64Type   // group
    };

    // Lookup or create the runtime function
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kMiopenConvolutionForward, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    // Build argument list
    SmallVector<Value, 16> args = {
        handlePtr, inputPtr, weightsPtr, biasPtr, outputPtr,
        kernelH, kernelW, strideH, strideW,
        padTop, padLeft, padBottom, padRight,
        dilationH, dilationW, groupVal};

    // Call the runtime function
    // Note: We're ignoring the return value for now (Phase 1 simplification)
    // Phase 2 TODO: Add error handling
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    // Erase the HIP conv operation (it's in-place, no results)
    rewriter.eraseOp(op);
    return success();
  }
};

// --- Pass
struct ConvertHipToLLVMPass
    : public PassWrapper<ConvertHipToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertHipToLLVMPass)

  StringRef getArgument() const final { return "convert-hip-to-llvm"; }
  StringRef getDescription() const final {
    return "Convert HIP dialect to LLVM dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    LowerToLLVMOptions options(ctx);
    LLVMTypeConverter typeConverter(ctx, options);

    // Convert !hip.context to !llvm.ptr
    typeConverter.addConversion(
        [ctx](ContextType type) -> Type {
          return LLVM::LLVMPointerType::get(ctx, 0);
        });

    RewritePatternSet patterns(ctx);
    patterns.add<CreateHandleOpLowering, DestroyHandleOpLowering,
                 AllocOpLowering, FreeOpLowering, ConvOpLowering>(typeConverter);

    LLVMConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<HipDialect>();
    target.addLegalOp<ModuleOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertHipToLLVMPass() {
  return std::make_unique<ConvertHipToLLVMPass>();
}

void registerHipPasses() {
  // Register all HIP-related conversion passes
  // Note: Pass registration uses the getArgument() string from each pass class

  // ConvertOnnxToHipPass (defined in OnnxToHip.cpp)
  // Registered via: --convert-onnx-to-hip
  registerPass([]() -> std::unique_ptr<Pass> {
    return createConvertOnnxToHipPass();
  });

  // ConvertHipToLLVMPass (defined in this file)
  // Registered via: --convert-hip-to-llvm
  PassRegistration<ConvertHipToLLVMPass>();
}

} // namespace hip
} // namespace mlir
