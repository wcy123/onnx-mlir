/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <glog/logging.h>

// MLIR includes
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

// LLVM includes for IR context
#include "llvm/IR/LLVMContext.h"

// HipDialect passes
#include "../../lib/HipDialect/HipPasses.h"

// Backend infrastructure
#include "../../lib/Backend/LLVMBackend.h"
#include "../../lib/Backend/DLLLinker.h"

using namespace morphizen;
using namespace morphizen_cxx;

DEF_ENV_PARAM(MLIR_PRINT_WITH_VERBOSE, "0")
DEF_ENV_PARAM(COMPILATION_MODE, "native")  // "ir" or "native"
DEF_ENV_PARAM(OUTPUT_PATH, "inference")     // Output file base name

namespace {

struct Level1MlirPass {
  Level1MlirPass(IPass &self) : self_{self} {}

  void process(IPass &self, Graph &graph) {
    LOG(INFO) << "Level1MlirPass::process() called";

    auto graph_ref = GraphConstRef(graph);
    auto graph_string = graph_ref.save_string();
    LOG(INFO) << "Graph serialized to bytecode, size: " << graph_string->size();

    // Parse MLIR bytecode to mlir::ModuleOp
    LOG(INFO) << "Parsing MLIR bytecode to ModuleOp...";
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    context.allowUnregisteredDialects();

    // Create a MemoryBuffer from the bytecode data
    auto memBuffer = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(graph_string->data(), graph_string->size()),
        "mlir-bytecode",
        /*RequiresNullTerminator=*/false);

    // Create SourceMgr and add the buffer
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());

    // Parse bytecode using parseSourceFile with SourceMgr
    mlir::ParserConfig parserConfig(&context);
    auto moduleRef =
        mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, parserConfig);

    if (!moduleRef) {
      LOG(INFO) << "Failed to parse MLIR bytecode to ModuleOp";
    } else {
      // Get the module operation
      mlir::ModuleOp module = *moduleRef;

      // Print initial module if verbose
      if (ENV_PARAM(MLIR_PRINT_WITH_VERBOSE)) {
        mlir::OpPrintingFlags flags;
        flags.printGenericOpForm();
        flags.enableDebugInfo();
        flags.printValueUsers();
        std::cout << "Initial ModuleOp content:" << std::endl;
        module.print(llvm::outs(), flags);
        std::cout << std::endl;
      }

      // Get compilation mode from environment
      std::string mode = ENV_PARAM(COMPILATION_MODE);
      std::string outputBase = ENV_PARAM(OUTPUT_PATH);
      LOG(INFO) << "Compilation mode: " << mode;

      // Run MLIR transformation pipeline
      LOG(INFO) << "Running MLIR transformation passes...";

      // Load required dialects
      context.loadDialect<mlir::LLVM::LLVMDialect>();

      // Create pass manager
      mlir::PassManager pm(&context);

      // Add HipDialect passes
      pm.addPass(mlir::hip::createConvertOnnxToHipPass());
      pm.addPass(mlir::hip::createConvertHipToLLVMPass());
      pm.addPass(mlir::hip::createGenerateInterfacePass());

      // Run passes
      if (mlir::failed(pm.run(module))) {
        LOG(ERROR) << "MLIR pass pipeline failed";
        return;
      }

      LOG(INFO) << "MLIR passes completed successfully";

      // Print transformed module if verbose
      if (ENV_PARAM(MLIR_PRINT_WITH_VERBOSE)) {
        std::cout << "Transformed ModuleOp content:" << std::endl;
        module.print(llvm::outs());
        std::cout << std::endl;
      }

      // Create LLVM backend for dual-mode compilation
      hipdnn::LLVMBackend backend;
      llvm::LLVMContext llvmContext;

      // Translate MLIR to LLVM IR (used by both modes)
      LOG(INFO) << "Translating MLIR to LLVM IR...";
      auto llvmModule = backend.translateMLIRtoLLVMIR(module, llvmContext);
      if (!llvmModule) {
        LOG(ERROR) << "Failed to translate MLIR to LLVM IR";
        return;
      }

      // Optimize LLVM IR (used by both modes)
      LOG(INFO) << "Optimizing LLVM IR (O2)...";
      backend.optimizeLLVMIR(llvmModule.get(), 2);

      if (mode == "ir") {
        // IR MODE: Emit LLVM IR text file
        std::string llPath = outputBase + ".ll";
        LOG(INFO) << "IR mode: Emitting LLVM IR to " << llPath;

        if (!backend.emitLLVMIR(llvmModule.get(), llPath)) {
          LOG(ERROR) << "Failed to emit LLVM IR";
          return;
        }

        LOG(INFO) << "Successfully generated LLVM IR: " << llPath;

      } else if (mode == "native") {
        // NATIVE MODE: Compile to object file then link to DLL

        // Step 1: Emit object file
        std::string objPath = outputBase + ".obj";
        LOG(INFO) << "Native mode: Compiling to object file " << objPath;

        if (!backend.compileToObjectFile(llvmModule.get(), objPath)) {
          LOG(ERROR) << "Failed to compile to object file";
          return;
        }

        LOG(INFO) << "Successfully compiled object file: " << objPath;

        // Step 2: Link DLL using LLD library
        std::string dllPath = outputBase + ".dll";
        LOG(INFO) << "Linking DLL: " << dllPath;

        hipdnn::DLLLinker linker;

        // Configure libraries and paths for ROCm
        std::vector<std::string> libraries = {
            "HipDnnRuntime",  // Our runtime library
            "amdhip64",       // HIP runtime
            "MIOpen",         // MIOpen
            "hipblaslt"       // hipBLASLt
        };

        std::vector<std::string> libraryPaths = {
            "C:/Develop/m/source/onnx-hipdnn-ep/build/lib/Runtime",  // Our runtime
            "C:/Program Files/AMD/ROCm/5.7/bin",                      // ROCm libraries
            "C:/Program Files/AMD/ROCm/5.7/lib"
        };

        std::vector<std::string> exportSymbols = {
            "inference_init",
            "inference_compute",
            "inference_cleanup"
        };

        if (!linker.linkDLL(objPath, dllPath, libraries, libraryPaths, exportSymbols)) {
          LOG(ERROR) << "Failed to link DLL";
          return;
        }

        LOG(INFO) << "Successfully linked DLL: " << dllPath;

        // Step 3: Verify exports
        if (!linker.verifyDLLExports(dllPath, exportSymbols)) {
          LOG(WARNING) << "DLL export verification failed";
        }

      } else {
        LOG(ERROR) << "Unknown compilation mode: " << mode << " (use 'ir' or 'native')";
        return;
      }

      LOG(INFO) << "Compilation completed successfully in " << mode << " mode";
    }
  }

  IPass &self_;
};

} // namespace

DEFINE_MORPHIZEN_PASS(Level1MlirPass, morphizen_pass_level1_mlir)
