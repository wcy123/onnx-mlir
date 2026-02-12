/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Generate Interface Pass - Create C-compatible interface functions
//===----------------------------------------------------------------------===//
// This pass generates three C-ABI compatible functions that wrap the internal
// @main function:
// - inference_init: Allocate context, create handles, upload constants
// - inference_compute: Parse inputs/outputs, call @main
// - inference_cleanup: Free resources
//===----------------------------------------------------------------------===//

#include "HipDialect.h"
#include "HipPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

class GenerateInterfacePass
    : public PassWrapper<GenerateInterfacePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateInterfacePass)

  StringRef getArgument() const final { return "generate-interface"; }
  StringRef getDescription() const final {
    return "Generate C interface wrapper functions (inference_init, "
           "inference_compute, inference_cleanup)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Verify prerequisites
    if (failed(verifyPrerequisites(module))) {
      signalPassFailure();
      return;
    }

    // Read metadata
    auto inputCount = module->getAttrOfType<IntegerAttr>("hipdnn.input_count");
    auto inputRanks =
        module->getAttrOfType<DenseI64ArrayAttr>("hipdnn.input_ranks");
    auto outputCount =
        module->getAttrOfType<IntegerAttr>("hipdnn.output_count");
    auto outputRanks =
        module->getAttrOfType<DenseI64ArrayAttr>("hipdnn.output_ranks");

    // Declare malloc and free at module level (before generating functions)
    declareMallocFree(module);

    // Declare all runtime library functions
    declareRuntimeFunctions(module);

    // Generate interface functions
    generateInferenceInit(module);
    generateInferenceCompute(module, inputCount, inputRanks, outputCount,
                             outputRanks);
    generateInferenceCleanup(module);

    llvm::errs() << "[GenerateInterface] Generated 3 interface functions\n";
  }

private:
  /// Returns LLVM struct type for memref: (ptr, ptr, i64, array<rank x i64>, array<rank x i64>)
  Type getMemRefStructType(OpBuilder &builder, int64_t rank, unsigned addrSpace) {
    MLIRContext *ctx = builder.getContext();
    Type ptrType = LLVM::LLVMPointerType::get(ctx, addrSpace);
    Type i64Type = builder.getI64Type();
    Type sizeArrayType = LLVM::LLVMArrayType::get(i64Type, rank);
    Type strideArrayType = LLVM::LLVMArrayType::get(i64Type, rank);

    return LLVM::LLVMStructType::getLiteral(
        ctx, {ptrType, ptrType, i64Type, sizeArrayType, strideArrayType});
  }

  /// Declare malloc and free functions at module level
  void declareMallocFree(ModuleOp module) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();
    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);

    // Find insertion point at start of module body (before first operation)
    auto &firstOp = module.getBody()->front();
    builder.setInsertionPoint(&firstOp);

    // Declare malloc if not already present
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("malloc")) {
      auto mallocFuncType =
          LLVM::LLVMFunctionType::get(ptrType, {builder.getI64Type()});
      auto mallocFunc =
          builder.create<LLVM::LLVMFuncOp>(loc, "malloc", mallocFuncType);
      mallocFunc.setLinkage(LLVM::Linkage::External);
    }

    // Declare free if not already present
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("free")) {
      auto freeFuncType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(builder.getContext()), {ptrType});
      auto freeFunc =
          builder.create<LLVM::LLVMFuncOp>(loc, "free", freeFuncType);
      freeFunc.setLinkage(LLVM::Linkage::External);
    }
  }

  /// Declare runtime library functions for GPU operations
  void declareRuntimeFunctions(ModuleOp module) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();
    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    Type i64Type = builder.getI64Type();
    Type voidType = LLVM::LLVMVoidType::get(builder.getContext());

    // Find insertion point
    auto &firstOp = module.getBody()->front();
    builder.setInsertionPoint(&firstOp);

    // Declare HIP functions
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hipStreamCreate")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "hipStreamCreate", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hipStreamDestroy")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "hipStreamDestroy", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hipStreamSynchronize")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(loc, "hipStreamSynchronize",
                                                   funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    // Declare MIOpen functions
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("miopenCreate")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "miopenCreate", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("miopenSetStream")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType, ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "miopenSetStream", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("miopenDestroy")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "miopenDestroy", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    // Declare hipBLASLt functions
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hipblasLtCreate")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "hipblasLtCreate", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hipblasLtDestroy")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "hipblasLtDestroy", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    // Declare high-level runtime state management functions
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("runtime_state_init")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "runtime_state_init", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("runtime_state_cleanup")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(loc, "runtime_state_cleanup",
                                                   funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    // Declare runtime wrapper functions
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hip_malloc_wrapper")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType, i64Type});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "hip_malloc_wrapper", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hip_free_wrapper")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "hip_free_wrapper", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hip_memcpy_h2d_async")) {
      auto funcType = LLVM::LLVMFunctionType::get(
          i32Type, {ptrType, ptrType, i64Type, ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(loc, "hip_memcpy_h2d_async",
                                                   funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("hip_memcpy_d2h_async")) {
      auto funcType = LLVM::LLVMFunctionType::get(
          i32Type, {ptrType, ptrType, i64Type, ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(loc, "hip_memcpy_d2h_async",
                                                   funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
            "hip_stream_synchronize_wrapper")) {
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "hip_stream_synchronize_wrapper", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    // Declare runtime_get_stream accessor
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("runtime_get_stream")) {
      // void* runtime_get_stream(RuntimeState* state)
      auto funcType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
      auto func =
          builder.create<LLVM::LLVMFuncOp>(loc, "runtime_get_stream", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }
  }

  /// Verify that module has all required prerequisites
  LogicalResult verifyPrerequisites(ModuleOp module) {
    MLIRContext *ctx = module.getContext();
    Type ptrType = LLVM::LLVMPointerType::get(ctx, 0);
    Type i32Type = IntegerType::get(ctx, 32);
    Type i64Type = IntegerType::get(ctx, 64);

    // 0. Check pass hasn't run before (idempotency check)
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("inference_init") ||
        module.lookupSymbol<LLVM::LLVMFuncOp>("inference_compute") ||
        module.lookupSymbol<LLVM::LLVMFuncOp>("inference_cleanup")) {
      llvm::errs() << "[GenerateInterface] Interface functions already exist. "
                   << "Pass already ran.\n";
      return failure();
    }

    // 1. Check @main exists as llvm.func with correct signature: (ptr,ptr,ptr)->i32
    auto mainFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("main");
    if (!mainFunc) {
      // Give helpful error if it's func.func
      if (module.lookupSymbol<func::FuncOp>("main")) {
        llvm::errs() << "[GenerateInterface] @main is func.func, needs llvm.func.\n"
                     << "Run --convert-hip-to-llvm first.\n";
        return failure();
      }
      llvm::errs() << "[GenerateInterface] @main (llvm.func) not found\n";
      return failure();
    }

    // Verify @main signature
    auto mainType = mainFunc.getFunctionType();
    if (mainType.getNumParams() != 3 ||
        mainType.getParamType(0) != ptrType ||
        mainType.getParamType(1) != ptrType ||
        mainType.getParamType(2) != ptrType ||
        mainType.getReturnType() != i32Type) {
      llvm::errs() << "[GenerateInterface] @main has wrong signature.\n"
                   << "Expected: (ptr, ptr, ptr) -> i32\n";
      return failure();
    }

    // 2. Check get_constant_count: () -> i64
    auto getCountFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("get_constant_count");
    if (!getCountFunc) {
      llvm::errs() << "[GenerateInterface] get_constant_count (llvm.func) not found\n";
      return failure();
    }
    auto getCountType = getCountFunc.getFunctionType();
    if (getCountType.getNumParams() != 0 ||
        getCountType.getReturnType() != i64Type) {
      llvm::errs() << "[GenerateInterface] get_constant_count has wrong signature.\n"
                   << "Expected: () -> i64\n";
      return failure();
    }

    // 3. Check initialize_constants: (ptr) -> i32
    auto initFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("initialize_constants");
    if (!initFunc) {
      llvm::errs() << "[GenerateInterface] initialize_constants (llvm.func) not found\n";
      return failure();
    }
    auto initType = initFunc.getFunctionType();
    if (initType.getNumParams() != 1 ||
        initType.getParamType(0) != ptrType ||
        initType.getReturnType() != i32Type) {
      llvm::errs() << "[GenerateInterface] initialize_constants has wrong signature.\n"
                   << "Expected: (ptr) -> i32\n";
      return failure();
    }

    // 4. Check release_constants: (ptr) -> i32
    auto releaseFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("release_constants");
    if (!releaseFunc) {
      llvm::errs() << "[GenerateInterface] release_constants (llvm.func) not found\n";
      return failure();
    }
    auto releaseType = releaseFunc.getFunctionType();
    if (releaseType.getNumParams() != 1 ||
        releaseType.getParamType(0) != ptrType ||
        releaseType.getReturnType() != i32Type) {
      llvm::errs() << "[GenerateInterface] release_constants has wrong signature.\n"
                   << "Expected: (ptr) -> i32\n";
      return failure();
    }

    // 5. Check all 4 metadata attributes exist
    if (!module->getAttr("hipdnn.input_count")) {
      llvm::errs() << "[GenerateInterface] hipdnn.input_count attribute missing\n";
      return failure();
    }
    if (!module->getAttr("hipdnn.input_ranks")) {
      llvm::errs() << "[GenerateInterface] hipdnn.input_ranks attribute missing\n";
      return failure();
    }
    if (!module->getAttr("hipdnn.output_count")) {
      llvm::errs() << "[GenerateInterface] hipdnn.output_count attribute missing\n";
      return failure();
    }
    if (!module->getAttr("hipdnn.output_ranks")) {
      llvm::errs() << "[GenerateInterface] hipdnn.output_ranks attribute missing\n";
      return failure();
    }

    return success();
  }

  /// Generate inference_init function - simplified to call runtime_state_init()
  /// Signature: int inference_init(void** out_state);
  ///
  /// This function is now a simple wrapper that delegates to
  /// runtime_state_init() in the runtime library. All the complex
  /// initialization logic (creating handles, error handling, LIFO cleanup) is
  /// in C++ code instead of LLVM IR generation.
  void generateInferenceInit(ModuleOp module) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();

    // Set insertion point at end of module
    builder.setInsertionPointToEnd(module.getBody());

    // Create function type: (ptr) -> i32
    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    SmallVector<Type> paramTypes = {ptrType};
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, paramTypes);

    // Create function with C ABI attributes
    auto funcOp =
        builder.create<LLVM::LLVMFuncOp>(loc, "inference_init", funcType);
    funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

    // Create function body
    Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value outStatePtr = entryBlock->getArgument(0);

    // Call runtime_state_init(out_state)
    auto runtimeInitFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("runtime_state_init");
    auto call = builder.create<LLVM::CallOp>(loc, runtimeInitFunc,
                                             ValueRange{outStatePtr});

    // Return the result from runtime_state_init
    builder.create<LLVM::ReturnOp>(loc, call.getResult());
  }

  /// Generate inference_compute function - direct span_t/tensor_t parsing
  /// Signature: int inference_compute(void* state, span_t* inputs, span_t*
  /// outputs);
  ///
  /// Phase 1: Parse span_t and tensor_t structures to extract input/output metadata
  /// - Get stream using runtime_get_stream()
  /// - Parse inputsSpanPtr to extract tensor array
  /// - Get first input tensor and extract data, shape, rank
  /// - Parse outputsSpanPtr to extract tensor array
  /// - Get first output tensor and extract data, shape, rank
  ///
  /// Structure layouts:
  /// - span_t: {tensor_t* data, size_t count}
  /// - tensor_t: {void* data, int64_t* shape, size_t rank}
  void generateInferenceCompute(ModuleOp module, IntegerAttr inputCount,
                                DenseI64ArrayAttr inputRanks,
                                IntegerAttr outputCount,
                                DenseI64ArrayAttr outputRanks) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();

    builder.setInsertionPointToEnd(module.getBody());

    // Create function type: (ptr, ptr, ptr) -> i32
    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    Type i64Type = builder.getI64Type();
    SmallVector<Type> paramTypes = {ptrType, ptrType, ptrType};
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, paramTypes);

    // Create function with C ABI attributes
    auto funcOp =
        builder.create<LLVM::LLVMFuncOp>(loc, "inference_compute", funcType);
    funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

    // Create function body
    Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value state = entryBlock->getArgument(0);
    Value inputsSpanPtr = entryBlock->getArgument(1);
    Value outputsSpanPtr = entryBlock->getArgument(2);

    // Constants
    Value c0_i32 = builder.create<LLVM::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(0));
    Value c0_i64 = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(0));
    Value c1_i64 = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(1));
    Value c2_i64 = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(2));

    // Get stream from runtime state
    auto getStreamFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("runtime_get_stream");
    Value stream = builder.create<LLVM::CallOp>(
        loc, getStreamFunc, ValueRange{state}).getResult();

    // ========================================================================
    // Parse input span_t structure
    // ========================================================================
    // span_t structure: { tensor_t* data, size_t count }
    // Field 0: tensor_t* data
    // Field 1: size_t count

    // Get pointer to inputs.data field (field 0 of span_t)
    Value inputsDataFieldPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, ptrType, inputsSpanPtr,
        ArrayRef<LLVM::GEPArg>{0, 0});

    // Load inputs.data (tensor_t* array)
    Value inputTensorsArray =
        builder.create<LLVM::LoadOp>(loc, ptrType, inputsDataFieldPtr);

    // Get first input tensor (index 0)
    Value firstInputTensorPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, ptrType, inputTensorsArray,
        ArrayRef<LLVM::GEPArg>{0});

    // Parse first input tensor_t structure
    // tensor_t structure: { void* data, int64_t* shape, size_t rank }
    // Field 0: void* data (host data pointer)
    // Field 1: int64_t* shape (pointer to shape array)
    // Field 2: size_t rank (number of dimensions)

    // Get pointer to input_tensor.data field (field 0)
    Value inputDataFieldPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, ptrType, firstInputTensorPtr,
        ArrayRef<LLVM::GEPArg>{0, 0});

    // Load input data pointer (host memory)
    Value inputHostDataPtr =
        builder.create<LLVM::LoadOp>(loc, ptrType, inputDataFieldPtr);

    // Get pointer to input_tensor.shape field (field 1)
    Value inputShapeFieldPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, ptrType, firstInputTensorPtr,
        ArrayRef<LLVM::GEPArg>{0, 1});

    // Load input shape pointer
    Value inputShapePtr =
        builder.create<LLVM::LoadOp>(loc, ptrType, inputShapeFieldPtr);

    // Get pointer to input_tensor.rank field (field 2)
    Value inputRankFieldPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, ptrType, firstInputTensorPtr,
        ArrayRef<LLVM::GEPArg>{0, 2});

    // Load input rank (size_t, treat as i64)
    Value inputRank =
        builder.create<LLVM::LoadOp>(loc, i64Type, inputRankFieldPtr);

    // ========================================================================
    // Parse output span_t structure
    // ========================================================================
    // span_t structure: { tensor_t* data, size_t count }

    // Get pointer to outputs.data field (field 0 of span_t)
    Value outputsDataFieldPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, ptrType, outputsSpanPtr,
        ArrayRef<LLVM::GEPArg>{0, 0});

    // Load outputs.data (tensor_t* array)
    Value outputTensorsArray =
        builder.create<LLVM::LoadOp>(loc, ptrType, outputsDataFieldPtr);

    // Get first output tensor (index 0)
    Value firstOutputTensorPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, ptrType, outputTensorsArray,
        ArrayRef<LLVM::GEPArg>{0});

    // Parse first output tensor_t structure
    // tensor_t structure: { void* data, int64_t* shape, size_t rank }

    // Get pointer to output_tensor.data field (field 0)
    Value outputDataFieldPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, ptrType, firstOutputTensorPtr,
        ArrayRef<LLVM::GEPArg>{0, 0});

    // Load output data pointer (host memory)
    Value outputHostDataPtr =
        builder.create<LLVM::LoadOp>(loc, ptrType, outputDataFieldPtr);

    // Get pointer to output_tensor.shape field (field 1)
    Value outputShapeFieldPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, ptrType, firstOutputTensorPtr,
        ArrayRef<LLVM::GEPArg>{0, 1});

    // Load output shape pointer
    Value outputShapePtr =
        builder.create<LLVM::LoadOp>(loc, ptrType, outputShapeFieldPtr);

    // Get pointer to output_tensor.rank field (field 2)
    Value outputRankFieldPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, ptrType, firstOutputTensorPtr,
        ArrayRef<LLVM::GEPArg>{0, 2});

    // Load output rank (size_t, treat as i64)
    Value outputRank =
        builder.create<LLVM::LoadOp>(loc, i64Type, outputRankFieldPtr);

    // ========================================================================
    // Section A: Calculate buffer sizes
    // ========================================================================
    // For simplicity in first iteration, use inputRanks metadata
    // Future: Use inputRank runtime value for generic implementation

    // Calculate input buffer size: elements = shape[0] * shape[1] * ... * shape[rank-1]
    // For first input, use rank from metadata
    auto inputRanksArray = inputRanks.asArrayRef();
    int64_t firstInputRank = inputRanksArray[0];

    // Load all dimensions for first input
    Value inputElementCount = c1_i64;
    for (int64_t i = 0; i < firstInputRank; i++) {
      Value dimIndexVal = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(i));
      Value dimPtr = builder.create<LLVM::GEPOp>(
          loc, ptrType, ptrType, inputShapePtr,
          ArrayRef<LLVM::GEPArg>{dimIndexVal});
      Value dimValue = builder.create<LLVM::LoadOp>(loc, i64Type, dimPtr);
      inputElementCount = builder.create<LLVM::MulOp>(loc, inputElementCount, dimValue);
    }

    // Multiply by sizeof(float) = 4 to get size in bytes
    Value c4_i64 = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(4));
    Value inputSizeBytes = builder.create<LLVM::MulOp>(loc, inputElementCount, c4_i64);

    // Calculate output buffer size similarly
    auto outputRanksArray = outputRanks.asArrayRef();
    int64_t firstOutputRank = outputRanksArray[0];

    Value outputElementCount = c1_i64;
    for (int64_t i = 0; i < firstOutputRank; i++) {
      Value dimIndexVal = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(i));
      Value dimPtr = builder.create<LLVM::GEPOp>(
          loc, ptrType, ptrType, outputShapePtr,
          ArrayRef<LLVM::GEPArg>{dimIndexVal});
      Value dimValue = builder.create<LLVM::LoadOp>(loc, i64Type, dimPtr);
      outputElementCount = builder.create<LLVM::MulOp>(loc, outputElementCount, dimValue);
    }

    Value outputSizeBytes = builder.create<LLVM::MulOp>(loc, outputElementCount, c4_i64);

    // ========================================================================
    // Section B: Allocate GPU buffers
    // ========================================================================
    // Create blocks for control flow
    Block *allocInputBlock = funcOp.addBlock();
    Block *allocOutputBlock = funcOp.addBlock();
    Block *h2dCopyBlock = funcOp.addBlock();
    Block *errorAllocInputBlock = funcOp.addBlock();
    Block *errorAllocOutputBlock = funcOp.addBlock();

    // Unconditional branch to start allocation
    builder.create<LLVM::BrOp>(loc, allocInputBlock);

    // Allocate input GPU buffer
    builder.setInsertionPointToEnd(allocInputBlock);

    // Use LLVM::AllocaOp to create stack storage for GPU pointer (void**)
    Value inputGpuPtrStorage = builder.create<LLVM::AllocaOp>(
        loc, ptrType, ptrType, c1_i64, 0);

    // Call hip_malloc_wrapper(ptrStorage, sizeBytes)
    auto hipMallocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("hip_malloc_wrapper");
    Value inputMallocRet = builder.create<LLVM::CallOp>(
        loc, hipMallocFunc, ValueRange{inputGpuPtrStorage, inputSizeBytes}).getResult();

    // Check return value (0 = success)
    Value inputMallocFailed = builder.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::ne, inputMallocRet, c0_i32);

    // If malloc fails, return error code 3
    builder.create<LLVM::CondBrOp>(loc, inputMallocFailed,
                                   errorAllocInputBlock, allocOutputBlock);

    // Error handler for input allocation failure
    builder.setInsertionPointToEnd(errorAllocInputBlock);
    Value c3_i32 = builder.create<LLVM::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(3));
    builder.create<LLVM::ReturnOp>(loc, c3_i32);

    // Allocate output GPU buffer
    builder.setInsertionPointToEnd(allocOutputBlock);

    // Load the allocated input GPU pointer
    Value inputGpuPtr = builder.create<LLVM::LoadOp>(loc, ptrType, inputGpuPtrStorage);

    // Use LLVM::AllocaOp to create stack storage for output GPU pointer
    Value outputGpuPtrStorage = builder.create<LLVM::AllocaOp>(
        loc, ptrType, ptrType, c1_i64, 0);

    // Call hip_malloc_wrapper(ptrStorage, sizeBytes)
    Value outputMallocRet = builder.create<LLVM::CallOp>(
        loc, hipMallocFunc, ValueRange{outputGpuPtrStorage, outputSizeBytes}).getResult();

    // Check return value (0 = success)
    Value outputMallocFailed = builder.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::ne, outputMallocRet, c0_i32);

    // If malloc fails, free input buffer and return error code 3
    builder.create<LLVM::CondBrOp>(loc, outputMallocFailed,
                                   errorAllocOutputBlock, h2dCopyBlock);

    // Error handler for output allocation failure - free input GPU buffer first
    builder.setInsertionPointToEnd(errorAllocOutputBlock);
    auto hipFreeFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("hip_free_wrapper");
    builder.create<LLVM::CallOp>(loc, hipFreeFunc, ValueRange{inputGpuPtr});
    builder.create<LLVM::ReturnOp>(loc, c3_i32);

    // ========================================================================
    // Section C: H2D copy for input
    // ========================================================================
    builder.setInsertionPointToEnd(h2dCopyBlock);

    // Load the allocated output GPU pointer
    Value outputGpuPtr = builder.create<LLVM::LoadOp>(loc, ptrType, outputGpuPtrStorage);

    // Call hip_memcpy_h2d_async(gpuPtr, hostDataPtr, sizeBytes, stream)
    auto h2dFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("hip_memcpy_h2d_async");
    Value h2dRet = builder.create<LLVM::CallOp>(
        loc, h2dFunc,
        ValueRange{inputGpuPtr, inputHostDataPtr, inputSizeBytes, stream}).getResult();

    // Check return value
    Value h2dFailed = builder.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::ne, h2dRet, c0_i32);

    // Create blocks for control flow
    Block *mainCallBlock = funcOp.addBlock();
    Block *errorH2DBlock = funcOp.addBlock();

    // If copy fails, free allocated GPU memory and return error code 4
    builder.create<LLVM::CondBrOp>(loc, h2dFailed, errorH2DBlock, mainCallBlock);

    // Error handler for H2D copy failure
    builder.setInsertionPointToEnd(errorH2DBlock);
    builder.create<LLVM::CallOp>(loc, hipFreeFunc, ValueRange{inputGpuPtr});
    builder.create<LLVM::CallOp>(loc, hipFreeFunc, ValueRange{outputGpuPtr});
    Value c4_i32 = builder.create<LLVM::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(4));
    builder.create<LLVM::ReturnOp>(loc, c4_i32);

    // ========================================================================
    // Phase 3: Call @main with GPU buffers
    // ========================================================================
    builder.setInsertionPointToEnd(mainCallBlock);

    // Create blocks for D2H and cleanup
    Block *d2hCopyBlock = funcOp.addBlock();
    Block *cleanupBlock = funcOp.addBlock();
    Block *errorMainBlock = funcOp.addBlock();

    // Lookup @main function
    auto mainFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("main");
    if (!mainFunc) {
      // No @main function - this shouldn't happen after verification, but handle gracefully
      llvm::errs() << "[GenerateInterface] Warning: @main not found, skipping computation\n";
      builder.create<LLVM::BrOp>(loc, d2hCopyBlock);
    } else {
      // Build memref structs for input and output
      // Memref struct: {ptr allocated, ptr aligned, i64 offset, array<rank x i64> sizes, array<rank x i64> strides}

      // Get metadata for ranks
      auto inputRanksArray = inputRanks.asArrayRef();
      auto outputRanksArray = outputRanks.asArrayRef();
      int64_t firstInputRank = inputRanksArray[0];
      int64_t firstOutputRank = outputRanksArray[0];

      // Build input memref struct type (GPU address space = 1)
      Type inputMemrefType = getMemRefStructType(builder, firstInputRank, 1);
      Type outputMemrefType = getMemRefStructType(builder, firstOutputRank, 1);

      // Allocate stack space for input array (1 element for simplified case)
      Value c1_i32 = builder.create<LLVM::ConstantOp>(
          loc, i32Type, builder.getI32IntegerAttr(1));
      Value inputArrayPtr = builder.create<LLVM::AllocaOp>(
          loc, ptrType, inputMemrefType, c1_i32, 0);

      // Build input memref struct
      Value inputMemref = builder.create<LLVM::UndefOp>(loc, inputMemrefType);

      // Set allocated pointer (field 0)
      inputMemref = builder.create<LLVM::InsertValueOp>(
          loc, inputMemref, inputGpuPtr, ArrayRef<int64_t>{0});

      // Set aligned pointer (field 1) - same as allocated
      inputMemref = builder.create<LLVM::InsertValueOp>(
          loc, inputMemref, inputGpuPtr, ArrayRef<int64_t>{1});

      // Set offset (field 2) - always 0
      inputMemref = builder.create<LLVM::InsertValueOp>(
          loc, inputMemref, c0_i64, ArrayRef<int64_t>{2});

      // Build sizes array (field 3) - load from inputShapePtr
      Value sizesArray = builder.create<LLVM::UndefOp>(
          loc, LLVM::LLVMArrayType::get(i64Type, firstInputRank));
      for (int64_t i = 0; i < firstInputRank; i++) {
        Value dimIndexVal = builder.create<LLVM::ConstantOp>(
            loc, i64Type, builder.getI64IntegerAttr(i));
        Value dimPtr = builder.create<LLVM::GEPOp>(
            loc, ptrType, ptrType, inputShapePtr,
            ArrayRef<LLVM::GEPArg>{dimIndexVal});
        Value dimValue = builder.create<LLVM::LoadOp>(loc, i64Type, dimPtr);
        sizesArray = builder.create<LLVM::InsertValueOp>(
            loc, sizesArray, dimValue, ArrayRef<int64_t>{i});
      }
      inputMemref = builder.create<LLVM::InsertValueOp>(
          loc, inputMemref, sizesArray, ArrayRef<int64_t>{3});

      // Build strides array (field 4) - compute row-major strides
      Value stridesArray = builder.create<LLVM::UndefOp>(
          loc, LLVM::LLVMArrayType::get(i64Type, firstInputRank));
      Value strideAccum = c1_i64;
      // Compute strides in reverse order (innermost dimension has stride 1)
      for (int64_t i = firstInputRank - 1; i >= 0; i--) {
        stridesArray = builder.create<LLVM::InsertValueOp>(
            loc, stridesArray, strideAccum, ArrayRef<int64_t>{i});
        if (i > 0) {
          // Get dimension size for this level
          Value dimIndexVal = builder.create<LLVM::ConstantOp>(
              loc, i64Type, builder.getI64IntegerAttr(i));
          Value dimPtr = builder.create<LLVM::GEPOp>(
              loc, ptrType, ptrType, inputShapePtr,
              ArrayRef<LLVM::GEPArg>{dimIndexVal});
          Value dimValue = builder.create<LLVM::LoadOp>(loc, i64Type, dimPtr);
          strideAccum = builder.create<LLVM::MulOp>(loc, strideAccum, dimValue);
        }
      }
      inputMemref = builder.create<LLVM::InsertValueOp>(
          loc, inputMemref, stridesArray, ArrayRef<int64_t>{4});

      // Store input memref to array
      builder.create<LLVM::StoreOp>(loc, inputMemref, inputArrayPtr);

      // Allocate stack space for output array
      Value outputArrayPtr = builder.create<LLVM::AllocaOp>(
          loc, ptrType, outputMemrefType, c1_i32, 0);

      // Build output memref struct (similar to input)
      Value outputMemref = builder.create<LLVM::UndefOp>(loc, outputMemrefType);

      outputMemref = builder.create<LLVM::InsertValueOp>(
          loc, outputMemref, outputGpuPtr, ArrayRef<int64_t>{0});
      outputMemref = builder.create<LLVM::InsertValueOp>(
          loc, outputMemref, outputGpuPtr, ArrayRef<int64_t>{1});
      outputMemref = builder.create<LLVM::InsertValueOp>(
          loc, outputMemref, c0_i64, ArrayRef<int64_t>{2});

      // Build output sizes array
      Value outputSizesArray = builder.create<LLVM::UndefOp>(
          loc, LLVM::LLVMArrayType::get(i64Type, firstOutputRank));
      for (int64_t i = 0; i < firstOutputRank; i++) {
        Value dimIndexVal = builder.create<LLVM::ConstantOp>(
            loc, i64Type, builder.getI64IntegerAttr(i));
        Value dimPtr = builder.create<LLVM::GEPOp>(
            loc, ptrType, ptrType, outputShapePtr,
            ArrayRef<LLVM::GEPArg>{dimIndexVal});
        Value dimValue = builder.create<LLVM::LoadOp>(loc, i64Type, dimPtr);
        outputSizesArray = builder.create<LLVM::InsertValueOp>(
            loc, outputSizesArray, dimValue, ArrayRef<int64_t>{i});
      }
      outputMemref = builder.create<LLVM::InsertValueOp>(
          loc, outputMemref, outputSizesArray, ArrayRef<int64_t>{3});

      // Build output strides array
      Value outputStridesArray = builder.create<LLVM::UndefOp>(
          loc, LLVM::LLVMArrayType::get(i64Type, firstOutputRank));
      strideAccum = c1_i64;
      for (int64_t i = firstOutputRank - 1; i >= 0; i--) {
        outputStridesArray = builder.create<LLVM::InsertValueOp>(
            loc, outputStridesArray, strideAccum, ArrayRef<int64_t>{i});
        if (i > 0) {
          Value dimIndexVal = builder.create<LLVM::ConstantOp>(
              loc, i64Type, builder.getI64IntegerAttr(i));
          Value dimPtr = builder.create<LLVM::GEPOp>(
              loc, ptrType, ptrType, outputShapePtr,
              ArrayRef<LLVM::GEPArg>{dimIndexVal});
          Value dimValue = builder.create<LLVM::LoadOp>(loc, i64Type, dimPtr);
          strideAccum = builder.create<LLVM::MulOp>(loc, strideAccum, dimValue);
        }
      }
      outputMemref = builder.create<LLVM::InsertValueOp>(
          loc, outputMemref, outputStridesArray, ArrayRef<int64_t>{4});

      // Store output memref to array
      builder.create<LLVM::StoreOp>(loc, outputMemref, outputArrayPtr);

      // Call @main(state, inputArrayPtr, outputArrayPtr)
      auto mainCallOp = builder.create<LLVM::CallOp>(
          loc, mainFunc, ValueRange{state, inputArrayPtr, outputArrayPtr});
      Value mainResult = mainCallOp.getResult();

      // Check return value (0 = success)
      Value mainFailed = builder.create<LLVM::ICmpOp>(
          loc, LLVM::ICmpPredicate::ne, mainResult, c0_i32);

      // If @main fails, cleanup and return error
      builder.create<LLVM::CondBrOp>(loc, mainFailed, errorMainBlock, d2hCopyBlock);

      // Error handler for @main failure
      builder.setInsertionPointToEnd(errorMainBlock);
      builder.create<LLVM::CallOp>(loc, hipFreeFunc, ValueRange{inputGpuPtr});
      builder.create<LLVM::CallOp>(loc, hipFreeFunc, ValueRange{outputGpuPtr});
      Value c5_i32 = builder.create<LLVM::ConstantOp>(
          loc, i32Type, builder.getI32IntegerAttr(5));
      builder.create<LLVM::ReturnOp>(loc, c5_i32);
    }

    // ========================================================================
    // Section D: D2H copy for output
    // ========================================================================
    builder.setInsertionPointToEnd(d2hCopyBlock);

    // Call hip_memcpy_d2h_async(hostDataPtr, gpuPtr, sizeBytes, stream)
    auto d2hFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("hip_memcpy_d2h_async");
    Value d2hRet = builder.create<LLVM::CallOp>(
        loc, d2hFunc,
        ValueRange{outputHostDataPtr, outputGpuPtr, outputSizeBytes, stream}).getResult();

    // Check return value (for now, ignore errors and continue to sync)
    // Future: Add proper error handling

    // Add stream synchronization: hip_stream_synchronize_wrapper(stream)
    auto streamSyncFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("hip_stream_synchronize_wrapper");
    builder.create<LLVM::CallOp>(loc, streamSyncFunc, ValueRange{stream});

    // Continue to cleanup regardless of sync result (for now)
    builder.create<LLVM::BrOp>(loc, cleanupBlock);

    // ========================================================================
    // Section E: Free GPU buffers
    // ========================================================================
    builder.setInsertionPointToEnd(cleanupBlock);

    // Call hip_free_wrapper(inputGpuPtr)
    builder.create<LLVM::CallOp>(loc, hipFreeFunc, ValueRange{inputGpuPtr});

    // Call hip_free_wrapper(outputGpuPtr)
    builder.create<LLVM::CallOp>(loc, hipFreeFunc, ValueRange{outputGpuPtr});

    // Return success
    builder.create<LLVM::ReturnOp>(loc, c0_i32);
  }

  /// Generate inference_cleanup function with resource cleanup
  /// Signature: int inference_cleanup(void* state);
  /// Destroys GPU resources in reverse order of creation (LIFO)
  /// Uses best-effort cleanup: continues even if some operations fail
  /// Generate inference_cleanup function - simplified to call
  /// runtime_state_cleanup() Signature: int inference_cleanup(void* state);
  ///
  /// This function is now a simple wrapper that delegates to
  /// runtime_state_cleanup() in the runtime library. All the cleanup logic
  /// (synchronization, handle destruction, LIFO order) is in C++ code instead
  /// of LLVM IR generation.
  void generateInferenceCleanup(ModuleOp module) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();

    builder.setInsertionPointToEnd(module.getBody());

    // Create function type: (ptr) -> i32
    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    SmallVector<Type> paramTypes = {ptrType};
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, paramTypes);

    // Create function with C ABI attributes
    auto funcOp =
        builder.create<LLVM::LLVMFuncOp>(loc, "inference_cleanup", funcType);
    funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

    // Create function body
    Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value state = entryBlock->getArgument(0);

    // Call runtime_state_cleanup(state)
    auto runtimeCleanupFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("runtime_state_cleanup");
    auto call = builder.create<LLVM::CallOp>(loc, runtimeCleanupFunc,
                                             ValueRange{state});

    // Return the result from runtime_state_cleanup (always 0)
    builder.create<LLVM::ReturnOp>(loc, call.getResult());
  }
};

} // namespace

namespace mlir {
namespace hip {

std::unique_ptr<Pass> createGenerateInterfacePass() {
  return std::make_unique<GenerateInterfacePass>();
}

} // namespace hip
} // namespace mlir
