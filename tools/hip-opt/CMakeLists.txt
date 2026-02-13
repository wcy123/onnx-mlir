##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.20)
project(hip-opt)

# Setup LLVM/MLIR
include(LLVMSetup)
setup_llvm_mlir()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add the executable (link against HipDialect library, don't rebuild it)
add_executable(hip-opt
  hip-opt.cpp
)

# Link LLVM components
llvm_map_components_to_libnames(llvm_libs Support Core)

# Link against HipDialect library and MLIR libraries
include(MlirLibraries)
target_link_libraries(hip-opt PRIVATE
  HipDialect
  ${MLIR_OPT_LIBS}
  ${MLIR_CORE_LIBS}
  ${MLIR_DIALECT_LIBS}
  ${MLIR_CONVERSION_LIBS}
  ${llvm_libs}
)
