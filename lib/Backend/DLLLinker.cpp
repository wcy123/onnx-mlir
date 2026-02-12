/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "DLLLinker.h"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>

#include <fstream>
#include <iostream>
#include <sstream>

// LLD linker driver entry points
// These are declared in LLD headers but we declare them here to avoid complex
// includes NOTE: LLD is optional - only needed for Native mode compilation
#ifdef HAVE_LLD
namespace lld {
namespace coff {
bool link(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
}
namespace elf {
bool link(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
}
} // namespace lld
#endif

namespace hipdnn {

DLLLinker::DLLLinker() = default;
DLLLinker::~DLLLinker() = default;

bool DLLLinker::linkDLL(const std::string &objectFile,
                        const std::string &outputDLL,
                        const std::vector<std::string> &libraries,
                        const std::vector<std::string> &libraryPaths,
                        const std::vector<std::string> &exportSymbols) {
#ifdef _WIN32
  return linkDLL_Windows(objectFile, outputDLL, libraries, libraryPaths,
                         exportSymbols);
#else
  return linkDLL_Linux(objectFile, outputDLL, libraries, libraryPaths);
#endif
}

#ifdef _WIN32

bool DLLLinker::createModuleDefinitionFile(
    const std::string &defPath, const std::vector<std::string> &exportSymbols) {
  std::ofstream defFile(defPath);
  if (!defFile) {
    std::cerr << "Failed to create .def file: " << defPath << "\n";
    return false;
  }

  defFile << "EXPORTS\n";
  for (const auto &symbol : exportSymbols) {
    defFile << "    " << symbol << "\n";
  }

  defFile.close();
  return true;
}

bool DLLLinker::linkDLL_Windows(const std::string &objectFile,
                                const std::string &outputDLL,
                                const std::vector<std::string> &libraries,
                                const std::vector<std::string> &libraryPaths,
                                const std::vector<std::string> &exportSymbols) {
#ifdef HAVE_LLD
  // Create temporary .def file for exports
  std::string defFile = objectFile + ".def";
  if (!createModuleDefinitionFile(defFile, exportSymbols)) {
    return false;
  }

  // Build LLD-LINK command line arguments
  // Note: LLD's link() API doesn't need argv[0] (program name)
  std::vector<std::string> argStrings;
  argStrings.push_back("/DLL");     // Create DLL
  argStrings.push_back("/OUT:" + outputDLL);
  argStrings.push_back("/DEF:" + defFile);
  argStrings.push_back(objectFile);

  // Add library paths
  for (const auto &libPath : libraryPaths) {
    argStrings.push_back("/LIBPATH:" + libPath);
  }

  // Add libraries
  for (const auto &lib : libraries) {
    // Check if library path already has .lib extension
    if (lib.size() >= 4 && lib.substr(lib.size() - 4) == ".lib") {
      argStrings.push_back(lib);
    } else {
      argStrings.push_back(lib + ".lib");
    }
  }

  // Add default libraries and flags
  argStrings.push_back("/NOLOGO");
  argStrings.push_back("/MACHINE:X64");

  // Convert to C-style args for LLD
  std::vector<const char *> args;
  for (const auto &arg : argStrings) {
    args.push_back(arg.c_str());
  }

  // Debug: Print LLD command line
  std::cout << "LLD-LINK command (" << args.size() << " args): ";
  for (size_t i = 0; i < args.size(); ++i) {
    std::cout << "[" << i << "]='" << args[i] << "' ";
  }
  std::cout << "\n";

  // Call LLD linker library
  std::string stdoutStr, stderrStr;
  llvm::raw_string_ostream stdoutOS(stdoutStr);
  llvm::raw_string_ostream stderrOS(stderrStr);

  // Create ArrayRef explicitly
  llvm::ArrayRef<const char *> argsRef(args);
  std::cout << "ArrayRef size: " << argsRef.size() << "\n";

  // Reset command line parser state before calling LLD
  // LLD has its own command line options that might conflict
  llvm::cl::ResetAllOptionOccurrences();

  bool success = lld::coff::link(argsRef, stdoutOS, stderrOS,
                                 /*exitEarly=*/false,
                                 /*disableOutput=*/false);

  // Print linker output
  if (!stdoutStr.empty()) {
    std::cout << stdoutStr;
  }
  if (!stderrStr.empty()) {
    std::cerr << stderrStr;
  }

  if (!success) {
    std::cerr << "LLD-LINK failed to link DLL\n";
    return false;
  }

  std::cout << "Successfully linked DLL: " << outputDLL << "\n";

  // Cleanup .def file
  llvm::sys::fs::remove(defFile);

  return true;
#else
  std::cerr << "ERROR: LLD library not available. Native mode requires LLVM "
               "built with LLD.\n";
  std::cerr
      << "       Rebuild LLVM with: -DLLVM_ENABLE_PROJECTS=\"mlir;lld\"\n";
  std::cerr << "       For now, use IR mode: export COMPILATION_MODE=ir\n";
  return false;
#endif
}

#else // Linux

bool DLLLinker::linkDLL_Linux(const std::string &objectFile,
                              const std::string &outputDLL,
                              const std::vector<std::string> &libraries,
                              const std::vector<std::string> &libraryPaths) {
#ifdef HAVE_LLD
  // Build LLD-ELF command line arguments for shared library
  std::vector<std::string> argStrings;
  argStrings.push_back("ld.lld");  // Program name (required by LLD)
  argStrings.push_back("-shared"); // Create shared library
  argStrings.push_back("-o");
  argStrings.push_back(outputDLL);
  argStrings.push_back(objectFile);

  // Add library paths
  for (const auto &libPath : libraryPaths) {
    argStrings.push_back("-L" + libPath);
  }

  // Add libraries
  for (const auto &lib : libraries) {
    argStrings.push_back("-l" + lib);
  }

  // Add RPATH for runtime library search
  for (const auto &libPath : libraryPaths) {
    argStrings.push_back("-rpath");
    argStrings.push_back(libPath);
  }

  // Add default flags
  argStrings.push_back("--export-dynamic"); // Export all symbols by default
  argStrings.push_back("--no-undefined");   // Error on undefined symbols

  // Convert to C-style args for LLD
  std::vector<const char *> args;
  for (const auto &arg : argStrings) {
    args.push_back(arg.c_str());
  }

  // Call LLD linker library
  std::string stdoutStr, stderrStr;
  llvm::raw_string_ostream stdoutOS(stdoutStr);
  llvm::raw_string_ostream stderrOS(stderrStr);

  bool success = lld::elf::link(args, stdoutOS, stderrOS,
                                /*exitEarly=*/false,
                                /*disableOutput=*/false);

  // Print linker output
  if (!stdoutStr.empty()) {
    std::cout << stdoutStr;
  }
  if (!stderrStr.empty()) {
    std::cerr << stderrStr;
  }

  if (!success) {
    std::cerr << "LLD-ELF failed to link shared library\n";
    return false;
  }

  std::cout << "Successfully linked shared library: " << outputDLL << "\n";
  return true;
#else
  std::cerr << "ERROR: LLD library not available. Native mode requires LLVM "
               "built with LLD.\n";
  std::cerr
      << "       Rebuild LLVM with: -DLLVM_ENABLE_PROJECTS=\"mlir;lld\"\n";
  std::cerr << "       For now, use IR mode: export COMPILATION_MODE=ir\n";
  return false;
#endif
}

#endif

bool DLLLinker::verifyDLLExports(
    const std::string &dllPath,
    const std::vector<std::string> &requiredSymbols) {
  // This is a simplified implementation
  // A complete implementation would use LLVM's object file libraries
  // to parse the DLL/SO and verify exported symbols

  std::cout << "Verifying DLL exports for: " << dllPath << "\n";
  for (const auto &symbol : requiredSymbols) {
    std::cout << "  Required symbol: " << symbol << "\n";
  }

  // TODO: Implement actual symbol verification using LLVM object file API
  // For now, just check if file exists
  if (!llvm::sys::fs::exists(dllPath)) {
    std::cerr << "DLL file does not exist: " << dllPath << "\n";
    return false;
  }

  return true;
}

} // namespace hipdnn
