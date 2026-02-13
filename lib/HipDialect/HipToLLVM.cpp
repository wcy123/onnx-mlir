/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipDialect.h"
#include "HipPasses.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
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
static constexpr const char *kMiopenConvolutionForward =
    "wrap_miopenConvolutionForward";
static constexpr const char *kHipGetConstant = "hipdnn_ep_constant_get";

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

// --- DestroyHandleOp: hip.destroy_handle(%h) -> llvm.call
// @hipDestroyHandle(%h)
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

// --- AllocOp: hip.alloc(%handle, %dyn...) -> hipMalloc(bytes) + memref
// descriptor
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

    // Declare hipMalloc with CORRECT signature: (ptr, i64) -> i32
    // The real hipMalloc signature is: hipError_t hipMalloc(void **ptr, size_t size)
    Type indexType = getIndexType();
    Type ptrType = getPtrType();
    Type i32Type = IntegerType::get(getContext(), 32);
    FailureOr<LLVM::LLVMFuncOp> mallocFn = LLVM::lookupOrCreateFn(
        rewriter, module, kHipMalloc, {ptrType, indexType}, i32Type);
    if (failed(mallocFn))
      return failure();

    // Compute sizes and sizeBytes (dynamic sizes are after the handle).
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value sizeBytes;
    getMemRefDescriptorSizes(loc, memRefType, adaptor.getDynamicSizes(),
                             rewriter, sizes, strides, sizeBytes, true);

    // Allocate stack space for the returned pointer
    Value one = rewriter.create<LLVM::ConstantOp>(loc, indexType,
                                                    rewriter.getIndexAttr(1));
    Value ptrStorage = rewriter.create<LLVM::AllocaOp>(
        loc, ptrType, ptrType, one, /*alignment=*/8);

    // Call hipMalloc(&ptrStorage, sizeBytes)
    Value mallocResult =
        LLVM::CallOp::create(rewriter, loc, *mallocFn, {ptrStorage, sizeBytes})
            .getResult();

    // TODO: Check mallocResult for errors (hipSuccess == 0)
    // For now, assume success

    // Load the allocated pointer from ptrStorage
    Value allocatedPtr = rewriter.create<LLVM::LoadOp>(loc, ptrType, ptrStorage);

    // Cast to memref address space if needed
    Type elementPtrType = getElementPtrType(memRefType);
    if (!elementPtrType)
      return rewriter.notifyMatchFailure(op,
                                         "could not compute element ptr type");
    FailureOr<unsigned> addrSpace =
        getTypeConverter()->getMemRefAddressSpace(memRefType);
    if (failed(addrSpace))
      return failure();
    if (cast<LLVM::LLVMPointerType>(allocatedPtr.getType()).getAddressSpace() !=
        *addrSpace)
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

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kHipFree, ptrType, voidType);
    if (failed(funcOp))
      return failure();

    Value memrefDesc = adaptor.getMemref();
    Value allocatedPtr =
        MemRefDescriptor(memrefDesc).allocatedPtr(rewriter, loc);
    // hipFree expects void*; if memref is in non-default address space, cast
    auto ptrTy = allocatedPtr.getType();
    if (cast<LLVM::LLVMPointerType>(ptrTy).getAddressSpace() != 0)
      allocatedPtr =
          rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, allocatedPtr);

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

    // Generate call to runtime wrapper following opaque RuntimeState pattern.
    // The wrapper extracts handle/stream from state internally (no direct field access!).
    //
    // Signature:
    // int wrap_miopenConvolutionForward(
    //     RuntimeState* state,    // Opaque pointer - extracts handle/stream internally
    //     void* input,            // Input tensor data pointer
    //     int64_t input_n,        // Input batch size
    //     int64_t input_c,        // Input channels
    //     int64_t input_h,        // Input height
    //     int64_t input_w,        // Input width
    //     void* weights,          // Weights tensor data pointer
    //     int64_t weights_k,      // Output channels (number of filters)
    //     void* bias,             // Bias tensor data pointer (nullable)
    //     void* output,           // Output tensor data pointer (in-place)
    //     int64_t output_h,       // Output height
    //     int64_t output_w,       // Output width
    //     int64_t kernel_h,       // Kernel height
    //     int64_t kernel_w,       // Kernel width
    //     int64_t stride_h,       // Stride height
    //     int64_t stride_w,       // Stride width
    //     int64_t pad_top,        // Padding top
    //     int64_t pad_left,       // Padding left
    //     int64_t pad_bottom,     // Padding bottom
    //     int64_t pad_right,      // Padding right
    //     int64_t dilation_h,     // Dilation height
    //     int64_t dilation_w,     // Dilation width
    //     int64_t group           // Number of groups
    // );
    //
    // Returns: 0 on success, non-zero on error

    // Helper to create i64 constants
    Type i64Type = rewriter.getI64Type();
    auto createI64Const = [&](int64_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(value));
    };

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

    Value statePtr = adaptor.getHandle();  // RuntimeState* (opaque)
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

    // Extract shapes from memref types (static shapes known at compile time)
    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto weightsType = cast<MemRefType>(op.getWeights().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Input shape: [N, C, H, W]
    auto inputShape = inputType.getShape();
    if (inputShape.size() != 4) {
      return op.emitError("Input must be rank-4 tensor [N, C, H, W]");
    }
    Value inputN = createI64Const(inputShape[0]);
    Value inputC = createI64Const(inputShape[1]);
    Value inputH = createI64Const(inputShape[2]);
    Value inputW = createI64Const(inputShape[3]);

    // Weights shape: [K, C, R, S] where K=output channels
    auto weightsShape = weightsType.getShape();
    if (weightsShape.size() != 4) {
      return op.emitError("Weights must be rank-4 tensor [K, C, R, S]");
    }
    Value weightsK = createI64Const(weightsShape[0]);

    // Output shape: [N, K, H', W']
    auto outputShape = outputType.getShape();
    if (outputShape.size() != 4) {
      return op.emitError("Output must be rank-4 tensor [N, K, H', W']");
    }
    Value outputH = createI64Const(outputShape[2]);
    Value outputW = createI64Const(outputShape[3]);

    // Extract attributes
    auto kernelShape = op.getKernelShape();
    auto strides = op.getStrides();
    auto pads = op.getPads();
    auto dilations = op.getDilations();
    auto group = op.getGroup();

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
    SmallVector<Type, 24> paramTypes = {
        ptrType, // state
        ptrType, // input
        i64Type, // input_n
        i64Type, // input_c
        i64Type, // input_h
        i64Type, // input_w
        ptrType, // weights
        i64Type, // weights_k
        ptrType, // bias
        ptrType, // output
        i64Type, // output_h
        i64Type, // output_w
        i64Type, // kernel_h
        i64Type, // kernel_w
        i64Type, // stride_h
        i64Type, // stride_w
        i64Type, // pad_top
        i64Type, // pad_left
        i64Type, // pad_bottom
        i64Type, // pad_right
        i64Type, // dilation_h
        i64Type, // dilation_w
        i64Type  // group
    };

    // Lookup or create the runtime function
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kMiopenConvolutionForward, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    // Build argument list matching the signature
    SmallVector<Value, 24> args = {
        statePtr,   inputPtr, inputN,    inputC,   inputH,   inputW,
        weightsPtr, weightsK, biasPtr,   outputPtr, outputH,  outputW,
        kernelH,    kernelW,  strideH,   strideW,  padTop,   padLeft,
        padBottom,  padRight, dilationH, dilationW, groupVal};

    // Call the runtime function
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    // Erase the HIP conv operation (it's in-place, no results)
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Constant Management Operations Lowering
//===----------------------------------------------------------------------===//

// --- GetConstantOp: %mem = hip.get_constant(%ctx, %index) : memref<...>
//     -> %ptr = hip_get_constant(%ctx, %index) + build memref descriptor
struct GetConstantOpLowering : public ConvertOpToLLVMPattern<GetConstantOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GetConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = IntegerType::get(rewriter.getContext(), 64);
    MemRefType memRefType = op.getResult().getType();

    // Function signature: void* hip_get_constant(void* state, i64 index)
    SmallVector<Type, 2> paramTypes = {ptrType, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipGetConstant, paramTypes, ptrType);
    if (failed(funcOp))
      return failure();

    // Call runtime function to get GPU pointer
    SmallVector<Value, 2> args = {
        adaptor.getCtx(),  // state pointer
        adaptor.getIndex() // constant index
    };

    auto callOp = LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    Value gpuPtr = callOp.getResult();

    // Cast to GPU address space (address space 1)
    // hip_get_constant returns !llvm.ptr, but memref needs !llvm.ptr<1>
    Value gpuPtrWithAddrSpace = rewriter.create<LLVM::AddrSpaceCastOp>(
        loc, LLVM::LLVMPointerType::get(rewriter.getContext(), 1), gpuPtr);

    // Build sizes and strides for memref descriptor
    auto shape = memRefType.getShape();
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;

    // Calculate sizes (all static for constants)
    for (int64_t dim : shape) {
      Value size = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(dim));
      sizes.push_back(size);
    }

    // Calculate strides (row-major layout)
    int64_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      Value strideVal = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(stride));
      strides.insert(strides.begin(), strideVal);
      stride *= shape[i];
    }

    // Create memref descriptor from GPU pointer (with address space)
    MemRefDescriptor desc =
        createMemRefDescriptor(loc, memRefType, gpuPtrWithAddrSpace,
                               gpuPtrWithAddrSpace, sizes, strides, rewriter);

    // Replace with the constructed memref descriptor
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

// NOTE: Main function transformation is handled post-conversion in
// runOnOperation

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
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    LowerToLLVMOptions options(ctx);
    LLVMTypeConverter typeConverter(ctx, options);

    // Convert !hip.context to !llvm.ptr
    typeConverter.addConversion([ctx](ContextType type) -> Type {
      return LLVM::LLVMPointerType::get(ctx, 0);
    });

    RewritePatternSet patterns(ctx);

    // Add HIP-specific conversion patterns
    patterns
        .add<CreateHandleOpLowering, DestroyHandleOpLowering, AllocOpLowering,
             FreeOpLowering, ConvOpLowering, GetConstantOpLowering>(
            typeConverter);

    // Add standard MLIR→LLVM conversion patterns
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

    LLVMConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<HipDialect>();
    target.addIllegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addLegalOp<ModuleOp>();

    // FuncOp is legal only if types are converted
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();

    // Post-processing: Transform @main function signature
    // After standard conversion
    // (populateFinalizeMemRefToLLVMConversionPatterns),
    // @main has memrefs unpacked to scalar parameters (23 params for 1 input +
    // 1 output rank-4) We wrap it: @main (3 params, struct arrays) →
    // @main_internal (23 params, scalars)
    if (failed(transformMainFunction(module)))
      signalPassFailure();
  }

private:
  /// Returns LLVM struct type for memref: (ptr, ptr, i64, array<rank x i64>,
  /// array<rank x i64>)
  Type getMemRefStructType(OpBuilder &builder, int64_t rank,
                           unsigned addrSpace) {
    MLIRContext *ctx = builder.getContext();
    Type ptrType = LLVM::LLVMPointerType::get(ctx, addrSpace);
    Type i64Type = builder.getI64Type();
    Type sizeArrayType = LLVM::LLVMArrayType::get(i64Type, rank);
    Type strideArrayType = LLVM::LLVMArrayType::get(i64Type, rank);

    return LLVM::LLVMStructType::getLiteral(
        ctx, {ptrType, ptrType, i64Type, sizeArrayType, strideArrayType});
  }

  /// Unpacks memref struct into scalar values (2 + 1 + rank + rank)
  void unpackMemRefStruct(OpBuilder &builder, Location loc, Value memrefStruct,
                          int64_t rank, SmallVectorImpl<Value> &args) {
    // Extract allocated pointer (field 0)
    args.push_back(builder.create<LLVM::ExtractValueOp>(loc, memrefStruct,
                                                        ArrayRef<int64_t>{0}));

    // Extract aligned pointer (field 1)
    args.push_back(builder.create<LLVM::ExtractValueOp>(loc, memrefStruct,
                                                        ArrayRef<int64_t>{1}));

    // Extract offset (field 2)
    args.push_back(builder.create<LLVM::ExtractValueOp>(loc, memrefStruct,
                                                        ArrayRef<int64_t>{2}));

    // Extract sizes (field 3, array elements 0..rank-1)
    for (int64_t dim = 0; dim < rank; dim++) {
      args.push_back(builder.create<LLVM::ExtractValueOp>(
          loc, memrefStruct, ArrayRef<int64_t>{3, dim}));
    }

    // Extract strides (field 4, array elements 0..rank-1)
    for (int64_t dim = 0; dim < rank; dim++) {
      args.push_back(builder.create<LLVM::ExtractValueOp>(
          loc, memrefStruct, ArrayRef<int64_t>{4, dim}));
    }
  }

  /// Transform @main from unpacked memrefs to array-based interface
  LogicalResult transformMainFunction(ModuleOp module) {
    // Find @main function
    auto mainFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("main");
    if (!mainFunc) {
      return success(); // No main function - this is fine
    }

    // Read metadata
    auto inputCountAttr =
        module->getAttrOfType<IntegerAttr>("hipdnn.input_count");
    auto outputCountAttr =
        module->getAttrOfType<IntegerAttr>("hipdnn.output_count");
    auto inputRanksAttr =
        module->getAttrOfType<DenseI64ArrayAttr>("hipdnn.input_ranks");
    auto outputRanksAttr =
        module->getAttrOfType<DenseI64ArrayAttr>("hipdnn.output_ranks");

    if (!inputCountAttr || !outputCountAttr || !inputRanksAttr ||
        !outputRanksAttr) {
      llvm::errs() << "[HipToLLVM] Warning: No metadata found, skipping @main "
                      "transformation\n";
      return success(); // Graceful degradation
    }

    int64_t inputCount = inputCountAttr.getInt();
    int64_t outputCount = outputCountAttr.getInt();
    auto inputRanks = inputRanksAttr.asArrayRef();
    auto outputRanks = outputRanksAttr.asArrayRef();

    // Validate metadata
    if (inputRanks.size() != inputCount || outputRanks.size() != outputCount) {
      return module.emitError("Metadata mismatch: ranks array size != count");
    }

    // Calculate expected parameter count (1 context + unpacked memrefs)
    unsigned expectedParams = 1; // context
    for (int64_t rank : inputRanks) {
      expectedParams +=
          2 + 1 + rank + rank; // 2 ptrs + offset + sizes + strides
    }
    for (int64_t rank : outputRanks) {
      expectedParams += 2 + 1 + rank + rank;
    }

    unsigned actualParams = mainFunc.getFunctionType().getNumParams();
    if (actualParams != expectedParams) {
      return module.emitError()
             << "[HipToLLVM] Parameter count mismatch: expected "
             << expectedParams << ", got " << actualParams;
    }

    OpBuilder builder(module.getContext());
    Location loc = mainFunc.getLoc();

    // Phase 1: Rename @main → @main_internal (make private)
    mainFunc.setName("main_internal");
    mainFunc.setLinkage(LLVM::Linkage::Private);

    // Phase 2: Create new @main with array-based interface
    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    SmallVector<Type> newParamTypes = {ptrType, ptrType, ptrType};
    auto newFuncType = LLVM::LLVMFunctionType::get(i32Type, newParamTypes);

    builder.setInsertionPoint(mainFunc);
    auto newMainFunc =
        builder.create<LLVM::LLVMFuncOp>(loc, "main", newFuncType);
    newMainFunc.setLinkage(LLVM::Linkage::Private);

    Block *entryBlock = newMainFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value ctxArg = entryBlock->getArgument(0);     // %context
    Value inputsArg = entryBlock->getArgument(1);  // %inputs
    Value outputsArg = entryBlock->getArgument(2); // %outputs

    // Build arguments for @main_internal
    SmallVector<Value> mainInternalArgs;
    mainInternalArgs.push_back(ctxArg); // arg0: context

    // Phase 3: Unpack inputs
    for (int64_t i = 0; i < inputCount; i++) {
      int64_t rank = inputRanks[i];

      // GEP to get pointer to inputs[i]
      Value inputIdxVal = builder.create<LLVM::ConstantOp>(
          loc, i32Type, builder.getI32IntegerAttr(i));
      Value inputStructPtr = builder.create<LLVM::GEPOp>(
          loc, ptrType, ptrType, inputsArg, ValueRange{inputIdxVal});

      // Load memref struct from array
      Type memrefStructType =
          getMemRefStructType(builder, rank, 1); // addr space 1 (GPU)
      Value inputMemref =
          builder.create<LLVM::LoadOp>(loc, memrefStructType, inputStructPtr);

      // Extract fields (for rank-4: 2 ptrs + offset + 4 sizes + 4 strides = 11)
      unpackMemRefStruct(builder, loc, inputMemref, rank, mainInternalArgs);
    }

    // Phase 4: Unpack outputs
    for (int64_t i = 0; i < outputCount; i++) {
      int64_t rank = outputRanks[i];

      Value outputIdxVal = builder.create<LLVM::ConstantOp>(
          loc, i32Type, builder.getI32IntegerAttr(i));
      Value outputStructPtr = builder.create<LLVM::GEPOp>(
          loc, ptrType, ptrType, outputsArg, ValueRange{outputIdxVal});

      Type memrefStructType = getMemRefStructType(builder, rank, 1);
      Value outputMemref =
          builder.create<LLVM::LoadOp>(loc, memrefStructType, outputStructPtr);

      unpackMemRefStruct(builder, loc, outputMemref, rank, mainInternalArgs);
    }

    // Phase 5: Call @main_internal with unpacked arguments
    auto callOp = builder.create<LLVM::CallOp>(loc, mainFunc, mainInternalArgs);
    Value result = callOp.getResult();

    // Return the result
    builder.create<LLVM::ReturnOp>(loc, result);

    llvm::errs() << "[HipToLLVM] Transformed @main signature: " << actualParams
                 << " params → 3 params\n";
    return success();
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
  registerPass(
      []() -> std::unique_ptr<Pass> { return createConvertOnnxToHipPass(); });

  // ConvertHipToLLVMPass (defined in this file)
  // Registered via: --convert-hip-to-llvm
  PassRegistration<ConvertHipToLLVMPass>();

  // GenerateInterfacePass (defined in GenerateInterfacePass.cpp)
  // Registered via: --generate-interface
  registerPass(
      []() -> std::unique_ptr<Pass> { return createGenerateInterfacePass(); });
}

} // namespace hip
} // namespace mlir
