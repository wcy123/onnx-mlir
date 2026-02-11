#include "LLVMBackend.h"

#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/MC/TargetRegistry.h>

#include <iostream>
#include <system_error>

namespace hipdnn {

LLVMBackend::LLVMBackend() : target_initialized_(false) {
    // Target initialization is deferred to first use
}

LLVMBackend::~LLVMBackend() = default;

void LLVMBackend::initializeTarget() {
    if (target_initialized_) {
        return;
    }

    // Initialize native target for object file emission
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    target_initialized_ = true;
}

std::unique_ptr<llvm::Module> LLVMBackend::translateMLIRtoLLVMIR(
    mlir::ModuleOp mlirModule,
    llvm::LLVMContext& llvmContext)
{
    // Register LLVM IR translation dialects
    mlir::registerBuiltinDialectTranslation(*mlirModule->getContext());
    mlir::registerLLVMDialectTranslation(*mlirModule->getContext());

    // Translate MLIR to LLVM IR using C++ library API
    auto llvmModule = mlir::translateModuleToLLVMIR(mlirModule, llvmContext);
    if (!llvmModule) {
        std::cerr << "Failed to translate MLIR to LLVM IR\n";
        return nullptr;
    }

    // Verify the generated LLVM IR
    std::string error_msg;
    llvm::raw_string_ostream error_stream(error_msg);
    if (llvm::verifyModule(*llvmModule, &error_stream)) {
        std::cerr << "LLVM IR verification failed:\n" << error_msg << "\n";
        return nullptr;
    }

    return llvmModule;
}

void LLVMBackend::optimizeLLVMIR(llvm::Module* module, int optLevel) {
    if (!module || optLevel < 0 || optLevel > 3) {
        std::cerr << "Invalid arguments to optimizeLLVMIR\n";
        return;
    }

    if (optLevel == 0) {
        // No optimization
        return;
    }

    // Create pass builder with optimization level
    llvm::PassBuilder PB;

    // Create analysis managers
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;

    // Register all the basic analyses with the managers
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // Create optimization pipeline based on level
    llvm::ModulePassManager MPM;
    if (optLevel == 1) {
        MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O1);
    } else if (optLevel == 2) {
        MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
    } else if (optLevel == 3) {
        MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    }

    // Run optimization passes
    MPM.run(*module, MAM);
}

bool LLVMBackend::emitLLVMIR(llvm::Module* module, const std::string& outputPath) {
    if (!module) {
        std::cerr << "Null module in emitLLVMIR\n";
        return false;
    }

    // Open output file
    std::error_code EC;
    llvm::raw_fd_ostream out(outputPath, EC, llvm::sys::fs::OF_None);
    if (EC) {
        std::cerr << "Failed to open output file: " << EC.message() << "\n";
        return false;
    }

    // Print LLVM IR in human-readable text format
    module->print(out, nullptr);

    std::cout << "Emitted LLVM IR to: " << outputPath << "\n";
    return true;
}

llvm::TargetMachine* LLVMBackend::createTargetMachine() {
    initializeTarget();

    // Get target triple for current platform
    std::string target_triple = llvm::sys::getDefaultTargetTriple();

    // Look up target
    std::string error_msg;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(target_triple, error_msg);
    if (!target) {
        std::cerr << "Failed to lookup target: " << error_msg << "\n";
        return nullptr;
    }

    // Configure target machine
    std::string cpu = "generic";
    std::string features = "";
    llvm::TargetOptions options;
    llvm::Reloc::Model RM = llvm::Reloc::PIC_;  // Position-independent code for DLL

    llvm::TargetMachine* TM = target->createTargetMachine(
        target_triple, cpu, features, options, RM);

    if (!TM) {
        std::cerr << "Failed to create target machine\n";
        return nullptr;
    }

    return TM;
}

bool LLVMBackend::compileToObjectFile(llvm::Module* module, const std::string& outputPath) {
    if (!module) {
        std::cerr << "Null module in compileToObjectFile\n";
        return false;
    }

    // Create target machine
    std::unique_ptr<llvm::TargetMachine> TM(createTargetMachine());
    if (!TM) {
        return false;
    }

    // Set module data layout and target triple
    module->setDataLayout(TM->createDataLayout());
    module->setTargetTriple(TM->getTargetTriple());

    // Open output file
    std::error_code EC;
    llvm::raw_fd_ostream out(outputPath, EC, llvm::sys::fs::OF_None);
    if (EC) {
        std::cerr << "Failed to open output file: " << EC.message() << "\n";
        return false;
    }

    // Create legacy pass manager for code generation
    llvm::legacy::PassManager pass;

    // Add pass to emit object file
    if (TM->addPassesToEmitFile(pass, out, nullptr, llvm::CodeGenFileType::ObjectFile)) {
        std::cerr << "TargetMachine can't emit object file\n";
        return false;
    }

    // Run code generation passes
    pass.run(*module);
    out.flush();

    std::cout << "Compiled object file to: " << outputPath << "\n";
    return true;
}

} // namespace hipdnn
