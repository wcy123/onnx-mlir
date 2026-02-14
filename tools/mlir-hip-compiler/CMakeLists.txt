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

# Setup LLVM/MLIR
include(LLVMSetup)
setup_llvm_mlir()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib)

# Get LLVM and MLIR libraries
llvm_map_components_to_libnames(llvm_libs support core irreader option windowsmanifest windowsdriver lto libdriver)

# MLIR libraries
include(MlirLibraries)
set(MLIR_LIBS
    ${MLIR_CORE_LIBS}
    ${MLIR_LLVM_TRANSLATION_LIBS}
    ${MLIR_DIALECT_LIBS}
)

# Our backend libraries
set(BACKEND_LIBS
    LLVMBackend      # Contains runtime_bc_data (from runtime_ir_data.cpp)
    DLLLinker
    HipDialect
)

# Define library output directory for runtime linking
set(RUNTIME_LIB_DIR "${CMAKE_BINARY_DIR}/lib/Runtime/${CMAKE_BUILD_TYPE}")

# Create executable
add_executable(mlir-hip-compiler main.cpp)

# Pass library directory as compile definition
target_compile_definitions(mlir-hip-compiler PRIVATE
    RUNTIME_LIB_DIR="${RUNTIME_LIB_DIR}"
)

# Link libraries
target_link_libraries(mlir-hip-compiler
    PRIVATE
    ${MLIR_LIBS}
    ${llvm_libs}
    ${BACKEND_LIBS}
)

# Installation
install(TARGETS mlir-hip-compiler DESTINATION bin)
