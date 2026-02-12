##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.18)
project(mlir-hip-compiler)

# TODO: Fix zlib dependency issue
# Currently mlir-hip-compiler.exe requires zlibd.dll at runtime because:
# - We link against LLVMSupport.lib which depends on zlib
# - zlib is dynamically linked
# Solution options:
# 1. Static link zlib (preferred - single executable, no runtime deps)
# 2. Remove zlib dependency from LLVM build (rebuild LLVM with -DLLVM_ENABLE_ZLIB=OFF)
# 3. Bundle zlibd.dll with the tool (workaround, not ideal)

# Find LLVM and MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib)

# Add LLVM definitions
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Get LLVM and MLIR libraries
llvm_map_components_to_libnames(llvm_libs support core irreader option windowsmanifest windowsdriver lto libdriver)

# MLIR libraries
set(MLIR_LIBS
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRTargetLLVMIRExport
    MLIRLLVMDialect
    MLIRFuncDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefDialect
    MLIRArithDialect
)

# Our backend libraries
set(BACKEND_LIBS
    LLVMBackend
    DLLLinker
    HipDialect
    HipDnnEpRuntime  # Required for runtime_bc_data symbol
)

# Create executable
add_executable(mlir-hip-compiler main.cpp)

# Set C++ standard
set_property(TARGET mlir-hip-compiler PROPERTY CXX_STANDARD 17)

# Link libraries
target_link_libraries(mlir-hip-compiler
    PRIVATE
    ${MLIR_LIBS}
    ${llvm_libs}
    ${BACKEND_LIBS}
)

# Installation
install(TARGETS mlir-hip-compiler DESTINATION bin)
