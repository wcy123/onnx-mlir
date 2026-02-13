##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Backend library for MLIR → LLVM IR → Object File compilation
# Supports dual modes: IR mode (text .ll) and Native mode (object file)

cmake_minimum_required(VERSION 3.18)

# Setup LLVM/MLIR
include(LLVMSetup)
setup_llvm_mlir()

# Create static libraries
add_library(LLVMBackend STATIC
    LLVMBackend.cpp
)

add_library(DLLLinker STATIC
    DLLLinker.cpp
)

# Include directories
target_include_directories(LLVMBackend
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
)

# Link LLVM and MLIR libraries
include(MlirLibraries)
target_link_libraries(LLVMBackend
    PUBLIC
        # MLIR translation
        ${MLIR_LLVM_TRANSLATION_LIBS}

        # LLVM core
        LLVMCore
        LLVMSupport
        LLVMTarget
        LLVMPasses

        # LLVM code generation
        LLVMCodeGen
        LLVMSelectionDAG
        LLVMAsmPrinter
        LLVMBinaryFormat
        LLVMMC
        LLVMMCParser
        LLVMTargetParser

        # Native target (will be initialized at runtime)
        LLVMX86CodeGen
        LLVMX86AsmParser
        LLVMX86Desc
        LLVMX86Info
)

# Compiler flags
include(CompilerFlags)
set_project_warnings(LLVMBackend)

# LLVM/MLIR require these definitions
target_compile_definitions(LLVMBackend PRIVATE
    ${LLVM_DEFINITIONS}
    $<$<BOOL:${HAVE_LLD}>:HAVE_LLD=1>
)

# DLLLinker include directories and link libraries
target_include_directories(DLLLinker
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${LLVM_INCLUDE_DIRS}
)

# Find LLD libraries (must be built with LLVM)
include(LldSetup)
find_and_setup_lld()

if(HAVE_LLD)
    message(STATUS "LLD libraries found - enabling DLL linking support")
else()
    message(WARNING "LLD libraries not found - DLL linking will not be available")
    message(WARNING "  Rebuild LLVM with: -DLLVM_ENABLE_PROJECTS=\"mlir;lld\"")
endif()

target_link_libraries(DLLLinker
    PUBLIC
        LLVMSupport

        # LLD linker libraries for DLL generation (if available)
        $<$<BOOL:${HAVE_LLD}>:${LLD_LIBRARIES}>
)

set_project_warnings(DLLLinker)

target_compile_definitions(DLLLinker PRIVATE
    ${LLVM_DEFINITIONS}
    $<$<BOOL:${HAVE_LLD}>:HAVE_LLD=1>
)

# Install rules
install(TARGETS LLVMBackend DLLLinker
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES LLVMBackend.h DLLLinker.h
    DESTINATION include
)
