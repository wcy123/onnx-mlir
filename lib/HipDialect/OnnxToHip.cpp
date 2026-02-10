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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

// Include ONNX dialect operations from onnx-mlir
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Constant Information Storage
//===----------------------------------------------------------------------===//

struct ConstantInfo {
  int64_t globalIndex;        // Sequential index (0, 1, 2, ...)
  ElementsAttr value;         // Constant data (from onnx.Constant)
  Type elementType;           // Element type (f32, i64, etc.)
  SmallVector<int64_t, 4> shape;  // Tensor shape (owned storage)
  size_t sizeInBytes;         // Total size in bytes
  std::string name;           // Debug name (from operation location)

  // Default constructor (required by DenseMap)
  ConstantInfo() : globalIndex(-1), sizeInBytes(0) {}

  ConstantInfo(int64_t idx, ElementsAttr val, Type elemType,
               ArrayRef<int64_t> shp, size_t size, StringRef debugName)
      : globalIndex(idx), value(val), elementType(elemType),
        shape(shp.begin(), shp.end()), sizeInBytes(size),
        name(debugName.str()) {}
};

namespace {

//===----------------------------------------------------------------------===//
// ONNX Constant → HIP Get Constant Conversion Pattern
//===----------------------------------------------------------------------===//

/// Convert onnx.Constant to hip.get_constant that retrieves pre-uploaded constant from state
struct ConstantToHipPattern : public OpConversionPattern<ONNXConstantOp> {
  const DenseMap<Value, ConstantInfo> &constantRegistry;

  ConstantToHipPattern(TypeConverter &typeConverter, MLIRContext *context,
                       const DenseMap<Value, ConstantInfo> &registry)
      : OpConversionPattern(typeConverter, context), constantRegistry(registry) {}

  LogicalResult matchAndRewrite(
      ONNXConstantOp constantOp,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = constantOp.getLoc();

    // Look up this constant in the registry
    auto it = constantRegistry.find(constantOp.getResult());
    if (it == constantRegistry.end()) {
      return rewriter.notifyMatchFailure(
          constantOp, "Constant not found in registry (not discovered during Phase 2)");
    }

    const auto &info = it->second;

    // Get context from parent function's first argument
    auto funcOp = constantOp->getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return rewriter.notifyMatchFailure(constantOp, "Not inside a function");
    }

    auto &entryBlock = funcOp.getBody().front();
    if (entryBlock.getNumArguments() == 0) {
      return rewriter.notifyMatchFailure(
          constantOp, "Function has no arguments (expected context as first arg)");
    }

    Value context = entryBlock.getArgument(0);
    if (!isa<hip::ContextType>(context.getType())) {
      return rewriter.notifyMatchFailure(
          constantOp, "First function argument is not a !hip.context");
    }

    // Create index constant
    Value index = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(info.globalIndex));

    // Convert output type: tensor<...> → memref<..., 1> (GPU address space)
    auto tensorType = cast<TensorType>(constantOp.getResult().getType());
    auto memrefType = getTypeConverter()->convertType(tensorType);
    if (!memrefType) {
      return rewriter.notifyMatchFailure(
          constantOp, "Failed to convert constant tensor type to memref");
    }

    // Create hip.get_constant operation to retrieve pre-uploaded constant
    auto getConstOp = rewriter.create<hip::GetConstantOp>(
        loc, memrefType, context, index);

    // Replace the onnx.Constant with the retrieved constant
    rewriter.replaceOp(constantOp, getConstOp.getResult());

    return success();
  }
};

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
      return rewriter.notifyMatchFailure(convOp, "Function has no arguments (expected context as first arg)");
    }

    Value context = entryBlock.getArgument(0);

    // Verify it's a context type
    if (!isa<hip::ContextType>(context.getType())) {
      return rewriter.notifyMatchFailure(convOp, "First function argument is not a !hip.context");
    }

    // Pass context directly to hip.conv
    // The hip.conv operation will use this context to access miopenHandle during HIP→LLVM lowering
    Value handle = context;

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
    // Phase 1: Allocate intermediate buffer inline (this buffer will be copied
    // to the function's output argument by ReturnOpConversion)
    // Phase 2 TODO: Use pre-allocated buffers from state instead
    rewriter.replaceOp(convOp, outputBuffer.getResult());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Func Return Conversion Pattern
//===----------------------------------------------------------------------===//
// Convert func.return to return i32 status code (destination-passing style)
//
// Destination-passing design:
// - Original ONNX: return %result : tensor<...>
// - After conversion: Write to output argument, return i32 status
//
// The pattern finds where return values should be written (output arguments)
// and generates stores, then returns status code 0 (success).

struct ReturnOpConversion : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::ReturnOp returnOp,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = returnOp.getLoc();

    // Destination-passing: outputs are already written to output arguments
    // by the operations (hip.conv, etc. use in-place semantics)
    //
    // For each return value, we need to copy it to the corresponding output argument.
    // The output arguments are the last N arguments of the function, where N is
    // the number of return values.

    auto funcOp = returnOp->getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return rewriter.notifyMatchFailure(returnOp, "Not inside a function");
    }

    auto &entryBlock = funcOp.getBody().front();
    unsigned numResults = returnOp.getNumOperands();
    unsigned numArgs = entryBlock.getNumArguments();

    // Output arguments are the last numResults arguments
    // (context + inputs + outputs)
    if (numArgs < numResults) {
      return rewriter.notifyMatchFailure(
          returnOp, "Function has fewer arguments than return values");
    }

    // Copy return values to output arguments
    for (unsigned i = 0; i < numResults; ++i) {
      Value returnValue = adaptor.getOperands()[i];
      Value outputArg = entryBlock.getArgument(numArgs - numResults + i);

      // Generate memref.copy to write result to output argument
      rewriter.create<memref::CopyOp>(loc, returnValue, outputArg);
    }

    // Return success status (i32 0)
    auto i32Type = rewriter.getI32Type();
    Value successStatus = rewriter.create<arith::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(0));

    rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp, successStatus);
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
// ONNX Function Identification Helper
//===----------------------------------------------------------------------===//

/// Check if a function is an ONNX function (has tensor types + ONNX operations)
/// This allows the pass to coexist with other MLIR passes and skip non-ONNX functions.
/// Also makes the pass idempotent: already-transformed functions won't match.
static bool isOnnxFunction(func::FuncOp funcOp) {
  auto funcType = funcOp.getFunctionType();

  // Quick filter: ONNX functions use tensor types
  bool hasTensorTypes = llvm::any_of(funcType.getInputs(), [](Type t) {
    return isa<TensorType>(t);
  }) || llvm::any_of(funcType.getResults(), [](Type t) {
    return isa<TensorType>(t);
  });

  if (!hasTensorTypes)
    return false;

  // Confirm: must have ONNX dialect operations
  bool hasOnnxOps = false;
  funcOp.walk([&](Operation *op) {
    if (auto *dialect = op->getDialect()) {
      if (isa<ONNXDialect>(dialect)) {
        hasOnnxOps = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  return hasOnnxOps;
}

//===----------------------------------------------------------------------===//
// ONNX to HIP Conversion Pass (Module-Level)
//===----------------------------------------------------------------------===//

class ConvertOnnxToHipPass
    : public PassWrapper<ConvertOnnxToHipPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertOnnxToHipPass)

  StringRef getArgument() const final { return "convert-onnx-to-hip"; }
  StringRef getDescription() const final {
    return "Convert ONNX dialect operations to HIP dialect operations (module-level for constant handling)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hip::HipDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();  // Needed for memref.dim and memref.copy
    registry.insert<arith::ArithDialect>();    // Needed for arith.constant (i32 status)
    registry.insert<ONNXDialect>();            // Needed for ONNX operations
    registry.insert<LLVM::LLVMDialect>();      // Needed for LLVM globals (constant storage)
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // Phase 2: Discover constants and assign global indices
    if (failed(discoverConstants(module))) {
      signalPassFailure();
      return;
    }

    // Phase 3: Generate LLVM globals for constants
    if (failed(generateConstantGlobals(module))) {
      signalPassFailure();
      return;
    }

    // Phase 4: Generate initialization functions
    if (failed(generateInitializationFunctions(module))) {
      signalPassFailure();
      return;
    }

    // Phase 1: Process each ONNX function

    for (auto func : module.getOps<func::FuncOp>()) {
      // Skip non-ONNX functions
      if (!isOnnxFunction(func)) {
        continue;
      }

      // Process this ONNX function
      if (failed(processOnnxFunction(func, context))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  /// Constant registry: maps SSA values to their global constant indices
  DenseMap<Value, ConstantInfo> constantRegistry_;

  /// Discover all onnx.Constant operations in the module and assign global indices
  LogicalResult discoverConstants(ModuleOp module) {
    int64_t nextIndex = 0;

    // Walk all operations in all functions to find onnx.Constant
    for (auto func : module.getOps<func::FuncOp>()) {
      // Only process ONNX functions
      if (!isOnnxFunction(func)) {
        continue;
      }

      func.walk([&](Operation *op) {
        // Check if this is an onnx.Constant operation
        if (auto constantOp = dyn_cast<ONNXConstantOp>(op)) {
          // Extract constant data
          auto valueAttr = constantOp.getValue();
          if (!valueAttr) {
            // onnx.Constant without value attribute - skip
            return WalkResult::advance();
          }

          auto elementsAttr = dyn_cast<ElementsAttr>(valueAttr.value());
          if (!elementsAttr) {
            // Not an ElementsAttr - skip (shouldn't happen for normal constants)
            return WalkResult::advance();
          }

          // Get tensor type information
          auto tensorType = cast<TensorType>(constantOp.getResult().getType());
          auto elementType = tensorType.getElementType();
          auto shape = tensorType.getShape();

          // Calculate size in bytes
          int64_t numElements = 1;
          for (int64_t dim : shape) {
            if (dim <= 0) {
              // Dynamic or zero dimension - skip (shouldn't happen for constants)
              return WalkResult::advance();
            }
            numElements *= dim;
          }

          size_t elementSize = elementType.getIntOrFloatBitWidth() / 8;
          size_t totalSize = numElements * elementSize;

          // Generate debug name from location
          std::string debugName = "constant_" + std::to_string(nextIndex);
          if (auto nameLoc = dyn_cast<NameLoc>(constantOp.getLoc())) {
            debugName = nameLoc.getName().str();
          }

          // Store constant info
          ConstantInfo info(nextIndex, elementsAttr, elementType, shape,
                           totalSize, debugName);

          constantRegistry_[constantOp.getResult()] = std::move(info);
          nextIndex++;
        }

        return WalkResult::advance();
      });
    }

    // Log discovery results
    if (nextIndex > 0) {
      llvm::errs() << "[ONNX→HIP] Discovered " << nextIndex << " constants:\n";
      for (const auto &entry : constantRegistry_) {
        const auto &info = entry.second;
        llvm::errs() << "  [" << info.globalIndex << "] " << info.name
                     << " : shape=[";
        for (size_t i = 0; i < info.shape.size(); ++i) {
          if (i > 0) llvm::errs() << "x";
          llvm::errs() << info.shape[i];
        }
        llvm::errs() << "], size=" << info.sizeInBytes << " bytes\n";
      }
    }

    return success();
  }

  /// Generate LLVM global variables for all discovered constants
  LogicalResult generateConstantGlobals(ModuleOp module) {
    if (constantRegistry_.empty()) {
      return success();  // No constants to generate
    }

    OpBuilder builder(module.getBodyRegion());

    // Set insertion point to the beginning of the module (before any functions)
    builder.setInsertionPointToStart(module.getBody());

    llvm::errs() << "[ONNX→HIP] Generating LLVM globals for "
                 << constantRegistry_.size() << " constants\n";

    for (const auto &entry : constantRegistry_) {
      const auto &info = entry.second;

      // Convert tensor type to LLVM array type
      // tensor<64x3x3x3xf32> → !llvm.array<1728 x f32>
      int64_t numElements = 1;
      for (int64_t dim : info.shape) {
        numElements *= dim;
      }

      // Create LLVM array type
      auto llvmElementType = info.elementType;  // f32, i64, etc.
      auto llvmArrayType = LLVM::LLVMArrayType::get(llvmElementType, numElements);

      // Create global variable with embedded constant data
      // Note: LLVM::GlobalOp requires an Attribute as initializer
      // ElementsAttr is already an Attribute, so we can use it directly
      auto globalOp = builder.create<LLVM::GlobalOp>(
          module.getLoc(),
          llvmArrayType,
          /*isConstant=*/true,
          LLVM::Linkage::Internal,
          info.name,
          info.value,  // Embed dense<...> data
          /*alignment=*/0,
          /*addr_space=*/0
      );

      llvm::errs() << "  Generated global: @" << info.name
                   << " : !llvm.array<" << numElements << " x "
                   << llvmElementType << ">\n";
    }

    return success();
  }

  /// Generate initialization functions for constant management
  LogicalResult generateInitializationFunctions(ModuleOp module) {
    if (constantRegistry_.empty()) {
      return success();  // No constants, no initialization needed
    }

    OpBuilder builder(module.getBodyRegion());
    auto loc = module.getLoc();

    llvm::errs() << "[ONNX→HIP] Generating initialization functions\n";

    // 1. Generate get_constant_count() -> i64
    {
      // Reset insertion point to end of module for each function
      builder.setInsertionPointToEnd(module.getBody());

      auto i64Type = builder.getI64Type();
      auto llvmFuncType = LLVM::LLVMFunctionType::get(i64Type, {});
      auto funcOp = builder.create<LLVM::LLVMFuncOp>(
          loc, "get_constant_count", llvmFuncType, LLVM::Linkage::External);

      Block *entryBlock = funcOp.addEntryBlock(builder);
      builder.setInsertionPointToStart(entryBlock);

      // Return constant count
      Value count = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(constantRegistry_.size()));
      builder.create<LLVM::ReturnOp>(loc, count);

      llvm::errs() << "  Generated: get_constant_count() -> "
                   << constantRegistry_.size() << "\n";
    }

    // 2. Generate initialize_constants(%ctx: !hip.context) -> i32
    {
      // Reset insertion point to end of module for each function
      builder.setInsertionPointToEnd(module.getBody());

      auto contextType = hip::ContextType::get(builder.getContext());
      auto i32Type = builder.getI32Type();
      auto funcType = builder.getFunctionType({contextType}, {i32Type});
      auto funcOp = builder.create<func::FuncOp>(
          loc, "initialize_constants", funcType);
      funcOp.setPublic();

      Block *entryBlock = funcOp.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);

      Value ctx = entryBlock->getArgument(0);

      // For each constant: upload to GPU
      for (const auto &entry : constantRegistry_) {
        const auto &info = entry.second;

        // Get address of global constant
        auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
        Value dataPtr = builder.create<LLVM::AddressOfOp>(
            loc, ptrType, info.name);

        // Create index constant
        Value index = builder.create<arith::ConstantOp>(
            loc, builder.getI64Type(),
            builder.getI64IntegerAttr(info.globalIndex));

        // Create size constant
        Value size = builder.create<arith::ConstantOp>(
            loc, builder.getI64Type(),
            builder.getI64IntegerAttr(info.sizeInBytes));

        // Call hip.upload_constant
        builder.create<hip::UploadConstantOp>(loc, ctx, index, dataPtr, size);
      }

      // Return success (0)
      Value success = builder.create<arith::ConstantOp>(
          loc, i32Type, builder.getI32IntegerAttr(0));
      builder.create<func::ReturnOp>(loc, success);

      llvm::errs() << "  Generated: initialize_constants() with "
                   << constantRegistry_.size() << " uploads\n";
    }

    // 3. Generate release_constants(%ctx: !hip.context) -> i32
    {
      // Reset insertion point to end of module for each function
      builder.setInsertionPointToEnd(module.getBody());

      auto contextType = hip::ContextType::get(builder.getContext());
      auto i32Type = builder.getI32Type();
      auto funcType = builder.getFunctionType({contextType}, {i32Type});
      auto funcOp = builder.create<func::FuncOp>(
          loc, "release_constants", funcType);
      funcOp.setPublic();

      Block *entryBlock = funcOp.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);

      Value ctx = entryBlock->getArgument(0);

      // For each constant: release from GPU
      for (const auto &entry : constantRegistry_) {
        const auto &info = entry.second;

        // Create index constant
        Value index = builder.create<arith::ConstantOp>(
            loc, builder.getI64Type(),
            builder.getI64IntegerAttr(info.globalIndex));

        // Call hip.release_constant
        builder.create<hip::ReleaseConstantOp>(loc, ctx, index);
      }

      // Return success (0)
      Value success = builder.create<arith::ConstantOp>(
          loc, i32Type, builder.getI32IntegerAttr(0));
      builder.create<func::ReturnOp>(loc, success);

      llvm::errs() << "  Generated: release_constants() with "
                   << constantRegistry_.size() << " releases\n";
    }

    return success();
  }

  /// Process a single ONNX function: add context, convert operations
  LogicalResult processOnnxFunction(func::FuncOp func, MLIRContext *context) {

    // Set up TypeConverter (tensor → memref with GPU address space)
    OnnxToHipTypeConverter typeConverter;

    // Add %ctx: !hip.context parameter to function if not present
    // Do this BEFORE conversion so patterns see the correct function signature
    //
    // NOTE: Pure ONNX-MLIR functions never have a ctx argument - that's
    // something we introduce during HIP lowering. However, we check for it
    // anyway to make this pass idempotent (safe to run multiple times).
    // This defensive check prevents adding duplicate context parameters if:
    // - The pass is accidentally run twice on the same function
    // - We're processing partially-lowered mixed IR
    // - The function was already processed in an earlier pipeline stage
    auto &entryBlock = func.getBody().front();
    bool hasContext = false;
    if (entryBlock.getNumArguments() > 0) {
      // Check if first argument is already a context
      if (isa<hip::ContextType>(entryBlock.getArgument(0).getType())) {
        hasContext = true;
      }
    }

    // If context already exists, the function was already lowered - skip it
    // Running conversion patterns on already-lowered code is both wasteful
    // and potentially incorrect (patterns expect ONNX ops, not HIP ops)
    if (hasContext) {
      // Function is in HIP dialect
      return success();
    }

    {
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

      // Destination-passing style: Add output arguments instead of return values
      // Convert result types to memref and add as function arguments
      for (Type resultType : funcType.getResults()) {
        Type convertedType = typeConverter.convertType(resultType);
        Type outputType = convertedType ? convertedType : resultType;
        newInputs.push_back(outputType);
        // Add corresponding block argument
        entryBlock.addArgument(outputType, func.getLoc());
      }

      // Return type is always i32 (status code: 0 = success)
      SmallVector<Type, 1> newResults;
      newResults.push_back(builder.getI32Type());

      auto newFuncType = builder.getFunctionType(newInputs, newResults);
      func.setFunctionType(newFuncType);

      // Update block argument types for inputs (except context which we just added)
      // Note: Output arguments were already added with correct types above
      unsigned numInputArgs = 1 + funcType.getInputs().size();  // context + original inputs
      for (unsigned i = 1; i < numInputArgs; ++i) {
        Type oldType = entryBlock.getArgument(i).getType();
        Type newType = typeConverter.convertType(oldType);
        if (newType && newType != oldType) {
          entryBlock.getArgument(i).setType(newType);
        }
      }
    }

    // Set up conversion target
    ConversionTarget target(*context);

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

    // Mark Arith dialect as legal (we generate arith.constant for i32 status)
    target.addLegalDialect<arith::ArithDialect>();

    // Mark ONNX Conv as illegal (must be lowered)
    target.addIllegalOp<ONNXConvOp>();

    // Mark ONNX Constant as illegal (must be lowered to hip.get_constant)
    target.addIllegalOp<ONNXConstantOp>();

    // All other ONNX ops are legal for now (only converting Conv and Constant)
    // NOTE: Cannot use addLegalDialect<ONNXDialect>() because it would override
    // the specific illegal ops above. Instead, operations not explicitly marked
    // illegal will be legal by default in partial conversion.
    // target.addLegalDialect<ONNXDialect>();

    // Set up rewrite patterns (pass typeConverter and constantRegistry to patterns)
    RewritePatternSet patterns(context);
    patterns.add<ConstantToHipPattern>(typeConverter, context, constantRegistry_);
    patterns.add<ConvToHipPattern>(typeConverter, context);
    patterns.add<ReturnOpConversion>(typeConverter, context);

    // Apply conversion
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      return failure();
    }

    return success();
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
