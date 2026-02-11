#ifndef LLVM_BACKEND_H
#define LLVM_BACKEND_H

#include <mlir/IR/BuiltinOps.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>

#include <memory>
#include <string>

namespace hipdnn {

// LLVM Backend: Translates MLIR to LLVM IR and optionally compiles to native object file
// Supports two compilation modes:
//   1. IR Mode: MLIR → LLVM IR (.ll text file) - for debugging and cross-platform
//   2. Native Mode: MLIR → LLVM IR → Object File (.obj/.o) - for production DLL generation

class LLVMBackend {
public:
    LLVMBackend();
    ~LLVMBackend();

    // MLIR → LLVM IR translation (used by both modes)
    // Translates MLIR module in LLVM dialect to LLVM IR using C++ library API
    // Returns nullptr on failure
    std::unique_ptr<llvm::Module> translateMLIRtoLLVMIR(
        mlir::ModuleOp mlirModule,
        llvm::LLVMContext& llvmContext);

    // LLVM IR optimization (used by both modes)
    // Runs standard optimization passes at specified level (0-3)
    // Level 0: No optimization
    // Level 1: Basic optimization
    // Level 2: Default optimization (recommended)
    // Level 3: Aggressive optimization
    void optimizeLLVMIR(llvm::Module* module, int optLevel);

    // IR Mode: Emit LLVM IR to text file (.ll)
    // Returns true on success, false on failure
    bool emitLLVMIR(llvm::Module* module, const std::string& outputPath);

    // Native Mode: Compile LLVM IR to object file (.obj on Windows, .o on Linux)
    // Uses LLVM TargetMachine to emit native code
    // Returns true on success, false on failure
    bool compileToObjectFile(llvm::Module* module, const std::string& outputPath);

private:
    // Helper: Initialize LLVM target for current platform
    void initializeTarget();

    // Helper: Create TargetMachine for native compilation
    llvm::TargetMachine* createTargetMachine();

    bool target_initialized_;
};

} // namespace hipdnn

#endif // LLVM_BACKEND_H
