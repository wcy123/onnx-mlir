##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Backend library for MLIR → LLVM IR → Object File compilation
# Supports dual modes: IR mode (text .ll) and Native mode (object file)

cmake_minimum_required(VERSION 3.18)

# Find LLVM and MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Include LLVM and MLIR CMake modules
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)

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
target_link_libraries(LLVMBackend
    PUBLIC
        # MLIR translation
        MLIRTargetLLVMIRExport
        MLIRBuiltinToLLVMIRTranslation
        MLIRLLVMToLLVMIRTranslation

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
target_compile_features(LLVMBackend PRIVATE cxx_std_17)

# LLVM/MLIR require these definitions
target_compile_definitions(LLVMBackend PRIVATE
    ${LLVM_DEFINITIONS}
    $<$<BOOL:${HAVE_LLD}>:HAVE_LLD=1>
)

if(MSVC)
    target_compile_options(LLVMBackend PRIVATE /W4 /wd4141 /wd4146 /wd4244 /wd4267)
else()
    target_compile_options(LLVMBackend PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# DLLLinker include directories and link libraries
target_include_directories(DLLLinker
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${LLVM_INCLUDE_DIRS}
)

# Find LLD libraries (must be built with LLVM)
find_library(LLD_COFF_LIB NAMES lldCOFF PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
find_library(LLD_ELF_LIB NAMES lldELF PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
find_library(LLD_COMMON_LIB NAMES lldCommon PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)

if(LLD_COFF_LIB AND LLD_ELF_LIB AND LLD_COMMON_LIB)
    message(STATUS "LLD libraries found - enabling DLL linking support")
    message(STATUS "  lldCOFF: ${LLD_COFF_LIB}")
    message(STATUS "  lldELF: ${LLD_ELF_LIB}")
    message(STATUS "  lldCommon: ${LLD_COMMON_LIB}")
    set(HAVE_LLD ON)
else()
    message(WARNING "LLD libraries not found - DLL linking will not be available")
    message(WARNING "  Rebuild LLVM with: -DLLVM_ENABLE_PROJECTS=\"mlir;lld\"")
    set(HAVE_LLD OFF)
endif()

target_link_libraries(DLLLinker
    PUBLIC
        LLVMSupport

        # LLD linker libraries for DLL generation (if available)
        $<$<BOOL:${LLD_COFF_LIB}>:${LLD_COFF_LIB}>
        $<$<BOOL:${LLD_ELF_LIB}>:${LLD_ELF_LIB}>
        $<$<BOOL:${LLD_COMMON_LIB}>:${LLD_COMMON_LIB}>
)

target_compile_features(DLLLinker PRIVATE cxx_std_17)
target_compile_definitions(DLLLinker PRIVATE
    ${LLVM_DEFINITIONS}
    $<$<BOOL:${HAVE_LLD}>:HAVE_LLD=1>
)

if(MSVC)
    target_compile_options(DLLLinker PRIVATE /W4 /wd4141 /wd4146 /wd4244 /wd4267)
else()
    target_compile_options(DLLLinker PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# Install rules
install(TARGETS LLVMBackend DLLLinker
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES LLVMBackend.h DLLLinker.h
    DESTINATION include
)
