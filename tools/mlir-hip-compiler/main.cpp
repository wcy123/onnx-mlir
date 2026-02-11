/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Standalone MLIR to HIP DLL Compiler
// Enables independent testing of the MLIR → LLVM IR → Object → DLL pipeline
//
// Usage: mlir-hip-compiler input.mlir -o output.dll [--from-onnx-mlir]
// [options]
//
// This tool links together:
// - HipDialect passes (OnnxToHip, HipToLLVM, GenerateInterface) - optional with
// --from-onnx-mlir
// - LLVM Backend (MLIR→IR translation, optimization, object compilation)
// - DLL Linker (Object→DLL linking)

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "../../lib/Backend/DLLLinker.h"
#include "../../lib/Backend/LLVMBackend.h"

// Include HIP dialect and passes
#include "../../lib/HipDialect/HipDialect.h"
#include "../../lib/HipDialect/HipPasses.h"

// Include ONNX dialect from onnx-mlir
#include "src/Dialect/ONNX/ONNXDialect.hpp"

// Include MLIR pass headers
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include <iostream>
#include <string>

using namespace llvm;

// Helper function to check if file exists (LLVM 22 compatible)
static bool fileExists(const std::string &path) {
  llvm::sys::fs::file_status status;
  std::error_code EC = llvm::sys::fs::status(path, status);
  return !EC && llvm::sys::fs::exists(status);
}

// Command line options
static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input .mlir file>"), cl::Required);

static cl::opt<std::string> outputFilename("o", cl::desc("Output DLL filename"),
                                           cl::value_desc("filename"),
                                           cl::init("output.dll"));

static cl::opt<std::string>
    outputMode("mode", cl::desc("Output mode: ir, object, or dll"),
               cl::value_desc("mode"), cl::init("dll"));

static cl::opt<int> optLevel("O", cl::desc("Optimization level (0-3)"),
                             cl::value_desc("level"), cl::init(2));

static cl::opt<bool> verbose("v", cl::desc("Verbose output"), cl::init(false));

static cl::opt<bool>
    keepIntermediates("keep", cl::desc("Keep intermediate files (.ll, .obj)"),
                      cl::init(false));

static cl::opt<bool> fromOnnxMlir(
    "from-onnx-mlir",
    cl::desc(
        "Input is MLIR with ONNX dialect (run ONNX→HIP→LLVM→Interface passes)"),
    cl::init(false));

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  // Parse command line options
  cl::ParseCommandLineOptions(argc, argv, "MLIR to HIP DLL Compiler\n");

  if (verbose) {
    std::cout << "=== MLIR to HIP DLL Compiler ===\n";
    std::cout << "Input: " << inputFilename << "\n";
    std::cout << "Output: " << outputFilename << "\n";
    std::cout << "Mode: " << outputMode << "\n";
    std::cout << "Optimization: O" << optLevel << "\n\n";
  }

  // Initialize MLIR context and register dialects
  mlir::MLIRContext context;

  // Register base dialects
  context.loadDialect<mlir::BuiltinDialect>();
  context.loadDialect<mlir::LLVM::LLVMDialect>();
  context.loadDialect<mlir::func::FuncDialect>();

  // If processing ONNX-MLIR, register additional dialects
  if (fromOnnxMlir) {
    context.loadDialect<mlir::arith::ArithDialect>();
    context.loadDialect<mlir::memref::MemRefDialect>();
    context.loadDialect<mlir::hip::HipDialect>();
    context.loadDialect<mlir::ONNXDialect>();
  }

  mlir::registerLLVMDialectTranslation(context);

  // Parse input MLIR file
  if (verbose)
    std::cout << "--- Step 1: Parsing MLIR ---\n";

  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    std::cerr << "Error opening input file: " << errorMessage << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    std::cerr << "Error parsing MLIR file\n";
    return 1;
  }

  if (verbose)
    std::cout << "✓ MLIR parsed successfully\n\n";

  // Run MLIR transformation passes
  if (verbose)
    std::cout << "--- Step 2: Running MLIR Passes ---\n";

  mlir::PassManager pm(&context);

  // Add our custom passes if processing ONNX-MLIR
  if (fromOnnxMlir) {
    pm.addPass(mlir::hip::createConvertOnnxToHipPass());
    pm.addPass(mlir::hip::createConvertHipToLLVMPass());
    pm.addPass(mlir::hip::createGenerateInterfacePass());

    if (verbose) {
      std::cout << "Running ONNX→HIP→LLVM→Interface passes\n";
    }
  } else {
    if (verbose) {
      std::cout
          << "Skipping passes - assuming input is already in LLVM dialect\n";
    }
  }

  if (mlir::failed(pm.run(*module))) {
    std::cerr << "Error running MLIR passes\n";
    return 1;
  }

  if (verbose)
    std::cout << "✓ MLIR passes completed\n\n";

  // Translate MLIR to LLVM IR
  if (verbose)
    std::cout << "--- Step 3: Translating to LLVM IR ---\n";

  hipdnn::LLVMBackend backend;
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      backend.translateMLIRtoLLVMIR(*module, llvmContext);

  if (!llvmModule) {
    std::cerr << "Error translating MLIR to LLVM IR\n";
    return 1;
  }

  if (verbose)
    std::cout << "✓ LLVM IR generated\n\n";

  // Optimize LLVM IR
  if (verbose)
    std::cout << "--- Step 4: Optimizing LLVM IR (O" << optLevel << ") ---\n";
  backend.optimizeLLVMIR(llvmModule.get(), optLevel);
  if (verbose)
    std::cout << "✓ Optimization completed\n\n";

  // Emit LLVM IR to file (if requested or keeping intermediates)
  std::string llFilename = outputFilename;
  if (llFilename.size() >= 4 &&
      llFilename.substr(llFilename.size() - 4) == ".dll") {
    llFilename = llFilename.substr(0, llFilename.size() - 4) + ".ll";
  } else {
    llFilename += ".ll";
  }

  if (outputMode == "ir" || keepIntermediates) {
    if (verbose)
      std::cout << "--- Step 5: Emitting LLVM IR ---\n";

    if (!backend.emitLLVMIR(llvmModule.get(), llFilename)) {
      std::cerr << "Error emitting LLVM IR\n";
      return 1;
    }

    if (verbose)
      std::cout << "✓ LLVM IR written to: " << llFilename << "\n\n";

    if (outputMode == "ir") {
      std::cout << "Output: " << llFilename << "\n";
      return 0;
    }
  }

  // Compile to object file
  std::string objFilename = outputFilename;
  if (objFilename.size() >= 4 &&
      objFilename.substr(objFilename.size() - 4) == ".dll") {
    objFilename = objFilename.substr(0, objFilename.size() - 4) + ".obj";
  } else {
    objFilename += ".obj";
  }

  if (verbose)
    std::cout << "--- Step 6: Compiling to Object File ---\n";

  if (!backend.compileToObjectFile(llvmModule.get(), objFilename)) {
    std::cerr << "Error compiling to object file\n";
    return 1;
  }

  if (verbose)
    std::cout << "✓ Object file created: " << objFilename << "\n\n";

  if (outputMode == "object") {
    std::cout << "Output: " << objFilename << "\n";
    return 0;
  }

  // Link to DLL
  if (verbose)
    std::cout << "--- Step 7: Linking to DLL ---\n";

  hipdnn::DLLLinker linker;

  // Define exported functions
  std::vector<std::string> exports = {"inference_init", "inference_compute",
                                      "inference_cleanup"};

  // Link with runtime library
  // Note: Need to find HipDnnRuntime.lib in build directory
  std::vector<std::string> libraries;

  // Try to find runtime library
  std::vector<std::string> searchPaths = {
      "../../lib/Runtime/build/Release",
      "../../test/runtime/build_standalone/Release", "../../build/Release",
      "../build/Release", "./Release"};

  for (const auto &path : searchPaths) {
    std::string libPath = path + std::string("/HipDnnRuntime.lib");
    if (fileExists(libPath)) {
      libraries.push_back(libPath);
      if (verbose)
        std::cout << "Found runtime library: " << libPath << "\n";
      break;
    }
  }

  if (libraries.empty() && verbose) {
    std::cout << "Warning: Runtime library not found, DLL may have unresolved "
                 "symbols\n";
  }

  std::vector<std::string> libraryPaths; // Empty for now

  if (!linker.linkDLL(objFilename, outputFilename, libraries, libraryPaths,
                      exports)) {
    std::cerr << "Error linking DLL\n";
    return 1;
  }

  if (verbose)
    std::cout << "✓ DLL created: " << outputFilename << "\n\n";

  // Verify DLL exports
  if (verbose) {
    std::cout << "--- Step 8: Verifying DLL Exports ---\n";
    if (linker.verifyDLLExports(outputFilename, exports)) {
      std::cout << "✓ All expected exports present\n\n";
    } else {
      std::cout
          << "⚠ Could not verify exports (dumpbin may not be available)\n\n";
    }
  }

  // Clean up intermediate files if not keeping
  if (!keepIntermediates) {
    if (verbose)
      std::cout << "--- Cleaning up intermediate files ---\n";

    if (fileExists(llFilename)) {
      std::error_code EC = llvm::sys::fs::remove(llFilename);
      if (!EC && verbose)
        std::cout << "Removed: " << llFilename << "\n";
    }

    if (fileExists(objFilename)) {
      std::error_code EC = llvm::sys::fs::remove(objFilename);
      if (!EC && verbose)
        std::cout << "Removed: " << objFilename << "\n";
    }

    if (verbose)
      std::cout << "\n";
  }

  std::cout << "=== Compilation Successful ===\n";
  std::cout << "Output: " << outputFilename << "\n";

  return 0;
}
