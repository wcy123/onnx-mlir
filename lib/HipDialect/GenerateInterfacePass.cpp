/**
 ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 ** Licensed under the MIT License.
 **/

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
    return "Generate C interface wrapper functions (inference_init, inference_compute, inference_cleanup)";
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
    auto inputRanks = module->getAttrOfType<DenseI64ArrayAttr>("hipdnn.input_ranks");
    auto outputCount = module->getAttrOfType<IntegerAttr>("hipdnn.output_count");
    auto outputRanks = module->getAttrOfType<DenseI64ArrayAttr>("hipdnn.output_ranks");

    // Declare malloc and free at module level (before generating functions)
    declareMallocFree(module);

    // Generate interface functions
    generateInferenceInit(module);
    generateInferenceCompute(module, inputCount, inputRanks, outputCount, outputRanks);
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
      auto mallocFuncType = LLVM::LLVMFunctionType::get(ptrType, {builder.getI64Type()});
      auto mallocFunc = builder.create<LLVM::LLVMFuncOp>(loc, "malloc", mallocFuncType);
      mallocFunc.setLinkage(LLVM::Linkage::External);
    }

    // Declare free if not already present
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("free")) {
      auto freeFuncType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(builder.getContext()), {ptrType});
      auto freeFunc = builder.create<LLVM::LLVMFuncOp>(loc, "free", freeFuncType);
      freeFunc.setLinkage(LLVM::Linkage::External);
    }
  }

  /// Verify that module has all required prerequisites
  LogicalResult verifyPrerequisites(ModuleOp module) {
    // Check @main exists (can be func.func or llvm.func)
    auto mainFunc = module.lookupSymbol<func::FuncOp>("main");
    auto mainLLVMFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("main");
    if (!mainFunc && !mainLLVMFunc) {
      llvm::errs() << "[GenerateInterface] Error: @main function not found\n";
      return failure();
    }

    // Check metadata exists
    if (!module->getAttr("hipdnn.input_count")) {
      llvm::errs() << "[GenerateInterface] Error: hipdnn.input_count attribute missing\n";
      return failure();
    }
    if (!module->getAttr("hipdnn.input_ranks")) {
      llvm::errs() << "[GenerateInterface] Error: hipdnn.input_ranks attribute missing\n";
      return failure();
    }
    if (!module->getAttr("hipdnn.output_count")) {
      llvm::errs() << "[GenerateInterface] Error: hipdnn.output_count attribute missing\n";
      return failure();
    }
    if (!module->getAttr("hipdnn.output_ranks")) {
      llvm::errs() << "[GenerateInterface] Error: hipdnn.output_ranks attribute missing\n";
      return failure();
    }

    // Check constant helpers exist
    if (!module.lookupSymbol("get_constant_count")) {
      llvm::errs() << "[GenerateInterface] Error: get_constant_count function not found\n";
      return failure();
    }
    if (!module.lookupSymbol("initialize_constants")) {
      llvm::errs() << "[GenerateInterface] Error: initialize_constants function not found\n";
      return failure();
    }
    if (!module.lookupSymbol("release_constants")) {
      llvm::errs() << "[GenerateInterface] Error: release_constants function not found\n";
      return failure();
    }

    return success();
  }

  /// Generate inference_init function with GPU resource initialization
  /// Signature: int inference_init(void** out_state);
  void generateInferenceInit(ModuleOp module) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();

    // Set insertion point at end of module
    builder.setInsertionPointToEnd(module.getBody());

    // Create function type: (ptr<ptr>) -> i32
    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    Type i64Type = builder.getI64Type();
    SmallVector<Type> paramTypes = {ptrType};
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, paramTypes);

    // Create function with C ABI attributes
    auto funcOp = builder.create<LLVM::LLVMFuncOp>(loc, "inference_init", funcType);
    funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

    // Create function body with error handling blocks
    Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value outStatePtr = entryBlock->getArgument(0);

    // Constants
    Value c0_i32 = builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0));
    Value c1_i32 = builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(1));
    Value c32_i64 = builder.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(32));
    Value nullPtr = builder.create<LLVM::ZeroOp>(loc, ptrType);

    // 1. Allocate context struct (32 bytes)
    auto mallocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("malloc");
    auto mallocCall = builder.create<LLVM::CallOp>(loc, mallocFunc, ValueRange{c32_i64});
    Value context = mallocCall.getResult();

    // Check if allocation succeeded
    Value allocFailed = builder.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, context, nullPtr);

    Block *allocSuccessBlock = builder.createBlock(entryBlock->getParent());
    Block *errorAllocBlock = builder.createBlock(entryBlock->getParent());

    builder.setInsertionPointToEnd(entryBlock);
    builder.create<LLVM::CondBrOp>(loc, allocFailed, errorAllocBlock, allocSuccessBlock);

    // 2. Success path: Store context and call initialize_constants
    builder.setInsertionPointToStart(allocSuccessBlock);

    // Store context pointer in output parameter
    builder.create<LLVM::StoreOp>(loc, context, outStatePtr);

    // Get initialize_constants function (if it exists, it's func.func not llvm.func)
    auto initConstantsFunc = module.lookupSymbol<func::FuncOp>("initialize_constants");
    if (initConstantsFunc) {
      // Call initialize_constants(context)
      // Note: This is a simplified call - proper implementation would convert func.func to llvm.func
      // For now, we'll skip this call in the LLVM dialect version
      // TODO: Convert initialize_constants to llvm.func or handle cross-dialect calls
    }

    // Return success
    builder.create<LLVM::ReturnOp>(loc, c0_i32);

    // 3. Error path: Return error code
    builder.setInsertionPointToStart(errorAllocBlock);
    builder.create<LLVM::ReturnOp>(loc, c1_i32);  // Error code 1: allocation failed
  }

  /// Generate inference_compute function with input validation
  /// Signature: int inference_compute(void* state, span_t* inputs, span_t* outputs);
  void generateInferenceCompute(ModuleOp module,
                                  IntegerAttr inputCount,
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
    auto funcOp = builder.create<LLVM::LLVMFuncOp>(loc, "inference_compute", funcType);
    funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

    // Create function body
    Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value state = entryBlock->getArgument(0);
    Value inputsPtr = entryBlock->getArgument(1);   // span_t*
    Value outputsPtr = entryBlock->getArgument(2);  // span_t*

    // Constants
    Value c0_i32 = builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0));
    Value c5_i32 = builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(5)); // Error: invalid input
    Value expectedInputCount = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(inputCount.getInt()));
    Value expectedOutputCount = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(outputCount.getInt()));

    // span_t structure: { tensor_t* data; size_t count; }
    // We need to:
    // 1. Load inputs->count (offset 1, assuming data is first field)
    // 2. Validate count matches metadata
    // 3. Load tensor_t array pointer
    // 4. For each tensor, validate rank and build memref descriptor

    // Create validation blocks
    Block *validateInputsBlock = builder.createBlock(entryBlock->getParent());
    Block *validateOutputsBlock = builder.createBlock(entryBlock->getParent());
    Block *buildMemrefsBlock = builder.createBlock(entryBlock->getParent());
    Block *callMainBlock = builder.createBlock(entryBlock->getParent());
    Block *errorBlock = builder.createBlock(entryBlock->getParent());

    // Entry: Jump to validation
    builder.setInsertionPointToEnd(entryBlock);
    builder.create<LLVM::BrOp>(loc, validateInputsBlock);

    // 1. Validate input count
    builder.setInsertionPointToStart(validateInputsBlock);

    // Get pointer to inputs->count (field at offset 8 bytes, assuming pointer is 8 bytes)
    Value c1_i32_idx = builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(1));
    Value inputCountPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, i64Type, inputsPtr, ValueRange{c1_i32_idx});
    Value actualInputCount = builder.create<LLVM::LoadOp>(loc, i64Type, inputCountPtr);

    // Compare counts
    Value inputCountMatch = builder.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, actualInputCount, expectedInputCount);

    builder.create<LLVM::CondBrOp>(loc, inputCountMatch, validateOutputsBlock, errorBlock);

    // 2. Validate output count
    builder.setInsertionPointToStart(validateOutputsBlock);

    Value outputCountPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, i64Type, outputsPtr, ValueRange{c1_i32_idx});
    Value actualOutputCount = builder.create<LLVM::LoadOp>(loc, i64Type, outputCountPtr);

    Value outputCountMatch = builder.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, actualOutputCount, expectedOutputCount);

    builder.create<LLVM::CondBrOp>(loc, outputCountMatch, buildMemrefsBlock, errorBlock);

    // 3. Build memref descriptors (simplified - just placeholder)
    builder.setInsertionPointToStart(buildMemrefsBlock);
    // TODO: Load tensor_t arrays, validate ranks, build memref structs
    // TODO: Allocate memref struct arrays
    builder.create<LLVM::BrOp>(loc, callMainBlock);

    // 4. Call @main (simplified - skip for now since @main signature is complex)
    builder.setInsertionPointToStart(callMainBlock);
    // TODO: Call @main with built memref arrays
    builder.create<LLVM::ReturnOp>(loc, c0_i32);

    // Error path
    builder.setInsertionPointToStart(errorBlock);
    builder.create<LLVM::ReturnOp>(loc, c5_i32);
  }

  /// Generate inference_cleanup function with resource cleanup
  /// Signature: int inference_cleanup(void* state);
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
    auto funcOp = builder.create<LLVM::LLVMFuncOp>(loc, "inference_cleanup", funcType);
    funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    funcOp->setAttr("sym_visibility", builder.getStringAttr("public"));

    // Create function body
    Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value state = entryBlock->getArgument(0);
    Value c0_i32 = builder.create<LLVM::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(0));

    // Get release_constants function (if exists)
    auto releaseConstantsFunc = module.lookupSymbol<func::FuncOp>("release_constants");
    if (releaseConstantsFunc) {
      // TODO: Call release_constants(state)
      // Skipping for now as it's a func.func not llvm.func
    }

    // TODO: Destroy GPU handles in reverse order:
    // 1. Load hipblasHandle from state[offset 16] and call hipblasLtDestroy
    // 2. Load miopenHandle from state[offset 8] and call miopenDestroy
    // 3. Load stream from state[offset 0] and call hipStreamDestroy
    // 4. Load gpu_constants from state[offset 24] and free

    // For now: just free the state itself
    auto freeFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("free");
    builder.create<LLVM::CallOp>(loc, freeFunc, ValueRange{state});

    // Return success
    builder.create<LLVM::ReturnOp>(loc, c0_i32);
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
