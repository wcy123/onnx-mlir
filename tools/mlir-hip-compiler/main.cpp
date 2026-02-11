// Standalone MLIR to HIP DLL Compiler
// Enables independent testing of the MLIR → LLVM IR → Object → DLL pipeline
//
// Usage: mlir-hip-compiler input.mlir -o output.dll [options]
//
// This tool links together:
// - HipDialect passes (OnnxToHip, HipToLLVM, GenerateInterface)
// - LLVM Backend (MLIR→IR translation, optimization, object compilation)
// - DLL Linker (Object→DLL linking)

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/FileSystem.h"

#include "../../lib/Backend/LLVMBackend.h"
#include "../../lib/Backend/DLLLinker.h"

// Include MLIR pass headers
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include <iostream>
#include <string>

using namespace llvm;

// Helper function to check if file exists (LLVM 22 compatible)
static bool fileExists(const std::string& path) {
    llvm::sys::fs::file_status status;
    std::error_code EC = llvm::sys::fs::status(path, status);
    return !EC && llvm::sys::fs::exists(status);
}

// Command line options
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input .mlir file>"),
                                          cl::Required);

static cl::opt<std::string> outputFilename("o",
                                           cl::desc("Output DLL filename"),
                                           cl::value_desc("filename"),
                                           cl::init("output.dll"));

static cl::opt<std::string> outputMode("mode",
                                        cl::desc("Output mode: ir, object, or dll"),
                                        cl::value_desc("mode"),
                                        cl::init("dll"));

static cl::opt<int> optLevel("O",
                              cl::desc("Optimization level (0-3)"),
                              cl::value_desc("level"),
                              cl::init(2));

static cl::opt<bool> verbose("v",
                              cl::desc("Verbose output"),
                              cl::init(false));

static cl::opt<bool> keepIntermediates("keep",
                                        cl::desc("Keep intermediate files (.ll, .obj)"),
                                        cl::init(false));

// Forward declarations for our custom passes
// These would be implemented in lib/HipDialect/
namespace mlir {
namespace hipdnn {
std::unique_ptr<mlir::Pass> createOnnxToHipPass();
std::unique_ptr<mlir::Pass> createHipToLLVMPass();
std::unique_ptr<mlir::Pass> createGenerateInterfacePass();
} // namespace hipdnn
} // namespace mlir

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
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    mlir::registerLLVMDialectTranslation(context);

    // Parse input MLIR file
    if (verbose) std::cout << "--- Step 1: Parsing MLIR ---\n";

    std::string errorMessage;
    auto file = mlir::openInputFile(inputFilename, &errorMessage);
    if (!file) {
        std::cerr << "Error opening input file: " << errorMessage << "\n";
        return 1;
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) {
        std::cerr << "Error parsing MLIR file\n";
        return 1;
    }

    if (verbose) std::cout << "✓ MLIR parsed successfully\n\n";

    // Run MLIR transformation passes
    if (verbose) std::cout << "--- Step 2: Running MLIR Passes ---\n";

    mlir::PassManager pm(&context);

    // Add our custom passes
    // Note: These passes are defined in lib/HipDialect/
    // For now, we'll assume they're linked in
    // pm.addPass(mlir::hipdnn::createOnnxToHipPass());
    // pm.addPass(mlir::hipdnn::createHipToLLVMPass());
    // pm.addPass(mlir::hipdnn::createGenerateInterfacePass());

    // If the passes aren't available, we'll skip and assume input is already in LLVM dialect
    if (verbose) {
        std::cout << "Note: Custom passes (OnnxToHip, HipToLLVM, GenerateInterface) should be run separately\n";
        std::cout << "This tool assumes input MLIR is already in LLVM dialect with interface functions\n\n";
    }

    if (mlir::failed(pm.run(*module))) {
        std::cerr << "Error running MLIR passes\n";
        return 1;
    }

    if (verbose) std::cout << "✓ MLIR passes completed\n\n";

    // Translate MLIR to LLVM IR
    if (verbose) std::cout << "--- Step 3: Translating to LLVM IR ---\n";

    hipdnn::LLVMBackend backend;
    llvm::LLVMContext llvmContext;
    std::unique_ptr<llvm::Module> llvmModule = backend.translateMLIRtoLLVMIR(*module, llvmContext);

    if (!llvmModule) {
        std::cerr << "Error translating MLIR to LLVM IR\n";
        return 1;
    }

    if (verbose) std::cout << "✓ LLVM IR generated\n\n";

    // Optimize LLVM IR
    if (verbose) std::cout << "--- Step 4: Optimizing LLVM IR (O" << optLevel << ") ---\n";
    backend.optimizeLLVMIR(llvmModule.get(), optLevel);
    if (verbose) std::cout << "✓ Optimization completed\n\n";

    // Emit LLVM IR to file (if requested or keeping intermediates)
    std::string llFilename = outputFilename;
    if (llFilename.size() >= 4 && llFilename.substr(llFilename.size() - 4) == ".dll") {
        llFilename = llFilename.substr(0, llFilename.size() - 4) + ".ll";
    } else {
        llFilename += ".ll";
    }

    if (outputMode == "ir" || keepIntermediates) {
        if (verbose) std::cout << "--- Step 5: Emitting LLVM IR ---\n";

        if (!backend.emitLLVMIR(llvmModule.get(), llFilename)) {
            std::cerr << "Error emitting LLVM IR\n";
            return 1;
        }

        if (verbose) std::cout << "✓ LLVM IR written to: " << llFilename << "\n\n";

        if (outputMode == "ir") {
            std::cout << "Output: " << llFilename << "\n";
            return 0;
        }
    }

    // Compile to object file
    std::string objFilename = outputFilename;
    if (objFilename.size() >= 4 && objFilename.substr(objFilename.size() - 4) == ".dll") {
        objFilename = objFilename.substr(0, objFilename.size() - 4) + ".obj";
    } else {
        objFilename += ".obj";
    }

    if (verbose) std::cout << "--- Step 6: Compiling to Object File ---\n";

    if (!backend.compileToObjectFile(llvmModule.get(), objFilename)) {
        std::cerr << "Error compiling to object file\n";
        return 1;
    }

    if (verbose) std::cout << "✓ Object file created: " << objFilename << "\n\n";

    if (outputMode == "object") {
        std::cout << "Output: " << objFilename << "\n";
        return 0;
    }

    // Link to DLL
    if (verbose) std::cout << "--- Step 7: Linking to DLL ---\n";

    hipdnn::DLLLinker linker;

    // Define exported functions
    std::vector<std::string> exports = {
        "inference_init",
        "inference_compute",
        "inference_cleanup"
    };

    // Link with runtime library
    // Note: Need to find HipDnnRuntime.lib in build directory
    std::vector<std::string> libraries;

    // Try to find runtime library
    std::vector<std::string> searchPaths = {
        "../../lib/Runtime/build/Release",
        "../../test/runtime/build_standalone/Release",
        "../../build/Release",
        "../build/Release",
        "./Release"
    };

    for (const auto& path : searchPaths) {
        std::string libPath = path + std::string("/HipDnnRuntime.lib");
        if (fileExists(libPath)) {
            libraries.push_back(libPath);
            if (verbose) std::cout << "Found runtime library: " << libPath << "\n";
            break;
        }
    }

    if (libraries.empty() && verbose) {
        std::cout << "Warning: Runtime library not found, DLL may have unresolved symbols\n";
    }

    std::vector<std::string> libraryPaths;  // Empty for now

    if (!linker.linkDLL(objFilename, outputFilename, libraries, libraryPaths, exports)) {
        std::cerr << "Error linking DLL\n";
        return 1;
    }

    if (verbose) std::cout << "✓ DLL created: " << outputFilename << "\n\n";

    // Verify DLL exports
    if (verbose) {
        std::cout << "--- Step 8: Verifying DLL Exports ---\n";
        if (linker.verifyDLLExports(outputFilename, exports)) {
            std::cout << "✓ All expected exports present\n\n";
        } else {
            std::cout << "⚠ Could not verify exports (dumpbin may not be available)\n\n";
        }
    }

    // Clean up intermediate files if not keeping
    if (!keepIntermediates) {
        if (verbose) std::cout << "--- Cleaning up intermediate files ---\n";

        if (fileExists(llFilename)) {
            std::error_code EC = llvm::sys::fs::remove(llFilename);
            if (!EC && verbose) std::cout << "Removed: " << llFilename << "\n";
        }

        if (fileExists(objFilename)) {
            std::error_code EC = llvm::sys::fs::remove(objFilename);
            if (!EC && verbose) std::cout << "Removed: " << objFilename << "\n";
        }

        if (verbose) std::cout << "\n";
    }

    std::cout << "=== Compilation Successful ===\n";
    std::cout << "Output: " << outputFilename << "\n";

    return 0;
}
