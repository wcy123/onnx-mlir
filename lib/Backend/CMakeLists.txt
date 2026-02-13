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

# Python 3 required for xxd.py (converts bitcode to C array)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ============================================================================
# Generate runtime_ir_data.cpp from runtime.bc
# ============================================================================

# Get runtime.bc path from Runtime project
if(NOT RUNTIME_BC_PATH)
    set(RUNTIME_BC_PATH ${CMAKE_BINARY_DIR}/lib/Runtime/runtime.bc)
endif()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp

    # Convert runtime.bc to C array
    COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/3rd-party/morphizen/tools/xxd.py
            --var runtime_bc_data
            --output ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp.tmp
            ${RUNTIME_BC_PATH}

    # Post-process: static const -> extern "C" const AND add size variable
    COMMAND ${Python3_EXECUTABLE} -c
            "import os; data = open('${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp.tmp').read(); data = data.replace('static const unsigned char', 'extern \"C\" const unsigned char'); bc_size = os.path.getsize('${RUNTIME_BC_PATH}'); data += f'\\nextern \"C\" const size_t runtime_bc_data_size = {bc_size};\\n'; open('${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp', 'w').write(data)"

    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp.tmp

    DEPENDS ${RUNTIME_BC_PATH}
    COMMENT "Generating runtime_ir_data.cpp from runtime.bc"
    VERBATIM
)

# Create static libraries
add_library(LLVMBackend STATIC
    LLVMBackend.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp  # Generated file
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
        ${CMAKE_CURRENT_BINARY_DIR}  # For generated runtime_ir_data.cpp
)

# Ensure runtime.bc is generated before building LLVMBackend
add_dependencies(LLVMBackend RuntimeBitcode)

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
    HAVE_LLD=1
)

# DLLLinker include directories and link libraries
target_include_directories(DLLLinker
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${LLVM_INCLUDE_DIRS}
)

# Find LLD libraries (REQUIRED - must be built with LLVM)
include(LldSetup)
find_and_setup_lld()

target_link_libraries(DLLLinker
    PUBLIC
        LLVMSupport
        ${LLD_LIBRARIES}
)

set_project_warnings(DLLLinker)

target_compile_definitions(DLLLinker PRIVATE
    ${LLVM_DEFINITIONS}
    HAVE_LLD=1
)

# Install rules
install(TARGETS LLVMBackend DLLLinker
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES LLVMBackend.h DLLLinker.h
    DESTINATION include
)
