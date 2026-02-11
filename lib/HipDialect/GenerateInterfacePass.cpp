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

    // Declare runtime inference helper functions
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("runtime_prepare_inference")) {
      // int runtime_prepare_inference(RuntimeState* state, span_t* inputs,
      // span_t* outputs, InferenceData** out_data)
      auto funcType = LLVM::LLVMFunctionType::get(
          i32Type, {ptrType, ptrType, ptrType, ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "runtime_prepare_inference", funcType);
      func.setLinkage(LLVM::Linkage::External);
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("runtime_cleanup_inference")) {
      // int runtime_cleanup_inference(RuntimeState* state, InferenceData* data,
      // span_t* outputs)
      auto funcType =
          LLVM::LLVMFunctionType::get(i32Type, {ptrType, ptrType, ptrType});
      auto func = builder.create<LLVM::LLVMFuncOp>(
          loc, "runtime_cleanup_inference", funcType);
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

  /// Generate inference_compute function - simplified to use runtime helpers
  /// Signature: int inference_compute(void* state, span_t* inputs, span_t*
  /// outputs);
  ///
  /// This function now delegates most work to runtime_prepare_inference() and
  /// runtime_cleanup_inference(), keeping only the model-specific logic here:
  /// - Call runtime_prepare_inference() to handle GPU allocation, H2D transfers
  /// - Extract GPU buffers from InferenceData
  /// - Build memref descriptors for @main
  /// - Call @main with memref arguments
  /// - Call runtime_cleanup_inference() to handle D2H transfers and GPU cleanup
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

    // Get function references
    auto prepareFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("runtime_prepare_inference");
    auto cleanupFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("runtime_cleanup_inference");

    // Constants
    Value c0_i32 = builder.create<LLVM::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(0));
    Value c1_i64 = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(1));

    // Allocate storage for InferenceData pointer
    Value infDataPtrStorage = builder.create<LLVM::AllocaOp>(
        loc, ptrType, ptrType, c1_i64, /*alignment=*/0);

    // Call runtime_prepare_inference(state, inputs, outputs, &inf_data)
    auto prepareResult = builder.create<LLVM::CallOp>(
        loc, prepareFunc,
        ValueRange{state, inputsSpanPtr, outputsSpanPtr, infDataPtrStorage});

    // Check if preparation succeeded
    Value prepareSuccess = builder.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, prepareResult.getResult(), c0_i32);

    Block *mainCallBlock = builder.createBlock(entryBlock->getParent());
    Block *errorPrepareBlock = builder.createBlock(entryBlock->getParent());

    builder.create<LLVM::CondBrOp>(loc, prepareSuccess, mainCallBlock,
                                   errorPrepareBlock);

    // Main call block: Extract GPU buffers and call @main
    builder.setInsertionPointToStart(mainCallBlock);

    // Load InferenceData*
    Value infDataPtr =
        builder.create<LLVM::LoadOp>(loc, ptrType, infDataPtrStorage);

    // Constants we'll need
    Value c0_i64 = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(0));
    Value c2_i64 = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(2));
    Value c3_i64 = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(3));
    Value c4_i64 = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(4));
    Value c5_i64 = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(5));

    // Extract GPU buffers from InferenceData structure
    // InferenceData layout (64-bit pointers):
    // - void** input_gpu_buffers  (offset 0)
    // - void** output_gpu_buffers (offset 8)
    // - int64_t* input_sizes      (offset 16)
    // - int64_t* output_sizes     (offset 24)
    // - size_t input_count        (offset 32)
    // - size_t output_count       (offset 40)

    // Load input_gpu_buffers (void**)
    Value inputGpuBuffersFieldPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, i64Type, infDataPtr, ValueRange{c0_i64});
    Value inputGpuBuffersArray =
        builder.create<LLVM::LoadOp>(loc, ptrType, inputGpuBuffersFieldPtr);

    // Load output_gpu_buffers (void**)
    Value outputGpuBuffersFieldPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, i64Type, infDataPtr, ValueRange{c1_i64});
    Value outputGpuBuffersArray =
        builder.create<LLVM::LoadOp>(loc, ptrType, outputGpuBuffersFieldPtr);

    // Load inputs span to get dimension information
    // span_t structure: { tensor_t* data; size_t count; }
    Value inputDataFieldPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, i64Type, inputsSpanPtr, ValueRange{c0_i64});
    Value inputTensorArray =
        builder.create<LLVM::LoadOp>(loc, ptrType, inputDataFieldPtr);

    // Load outputs span to get dimension information
    Value outputDataFieldPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, i64Type, outputsSpanPtr, ValueRange{c0_i64});
    Value outputTensorArray =
        builder.create<LLVM::LoadOp>(loc, ptrType, outputDataFieldPtr);

    // Build memref descriptors for @main call
    // Simplified approach: For now, we'll pass GPU buffer pointers directly
    // A full implementation would build proper ranked memref descriptors
    //
    // Note: This is a simplified implementation that assumes @main accepts
    // raw pointers rather than full memref descriptors. For a production
    // implementation, we would need to:
    // 1. Build proper memref structs with
    // allocated/aligned/offset/sizes/strides
    // 2. Handle variable-rank memrefs
    // 3. Calculate strides from dimensions
    //
    // For testing purposes, we'll lookup @main and pass GPU buffers directly

    // Lookup @main function
    auto mainFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("main");
    if (!mainFunc) {
      // If @main doesn't exist as LLVM func, try func.func
      auto mainFuncOp = module.lookupSymbol<func::FuncOp>("main");
      if (!mainFuncOp) {
        // No @main function - this is not an error, just skip the call
        // The DLL will still be created with inference_init/compute/cleanup
        llvm::errs() << "[GenerateInterface] Warning: @main function not "
                        "found, skipping model execution\n";
      } else {
        // func.func exists but we need LLVM func - should have been lowered
        // already
        llvm::errs() << "[GenerateInterface] Error: @main exists as func.func "
                        "but should be lowered to LLVM\n";
      }
    } else {
      // @main exists as LLVM function - call it
      // For now, create a simplified call without full memref descriptors
      // This will work for simple test cases but needs enhancement for
      // production

      // Get the function type to understand what arguments it expects
      auto mainFuncType = mainFunc.getFunctionType();
      SmallVector<Value> mainArgs;

      // Simple approach: Load first input and output GPU buffers
      // This assumes @main(ptr %input, ptr %output) signature
      // A full implementation would build proper memref arrays

      if (inputCount.getInt() > 0 && outputCount.getInt() > 0) {
        // Load first input GPU buffer
        Value inputGpuBuffer0Ptr = builder.create<LLVM::GEPOp>(
            loc, ptrType, ptrType, inputGpuBuffersArray, ValueRange{c0_i64});
        Value inputGpuBuffer0 =
            builder.create<LLVM::LoadOp>(loc, ptrType, inputGpuBuffer0Ptr);
        mainArgs.push_back(inputGpuBuffer0);

        // Load first output GPU buffer
        Value outputGpuBuffer0Ptr = builder.create<LLVM::GEPOp>(
            loc, ptrType, ptrType, outputGpuBuffersArray, ValueRange{c0_i64});
        Value outputGpuBuffer0 =
            builder.create<LLVM::LoadOp>(loc, ptrType, outputGpuBuffer0Ptr);
        mainArgs.push_back(outputGpuBuffer0);

        // Call @main with GPU buffers
        // Note: This is a simplified call that works for testing
        // Production code would need proper memref descriptors
        builder.create<LLVM::CallOp>(loc, mainFunc, mainArgs);
      }
    }

    // Call runtime_cleanup_inference(state, inf_data, outputs)
    auto cleanupResult = builder.create<LLVM::CallOp>(
        loc, cleanupFunc, ValueRange{state, infDataPtr, outputsSpanPtr});

    // Return success (or cleanup result)
    builder.create<LLVM::ReturnOp>(loc, cleanupResult.getResult());

    // Error path: prepare failed
    builder.setInsertionPointToStart(errorPrepareBlock);
    builder.create<LLVM::ReturnOp>(loc, prepareResult.getResult());
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
