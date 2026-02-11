##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.20)
project(hip-opt)

# Find LLVM using LLVM_DIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Add LLVM and MLIR definitions
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add LLVM TableGen function
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

# Add the executable (link against HipDialect library, don't rebuild it)
add_executable(hip-opt
  hip-opt.cpp
)

# Link LLVM components
llvm_map_components_to_libnames(llvm_libs Support Core)

# Link against HipDialect library and MLIR libraries
target_link_libraries(hip-opt PRIVATE
  HipDialect
  MLIROptLib
  MLIRSupport
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRAnalysis
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRTransformUtils
  ${llvm_libs}
)

# Set C++ standard
set_target_properties(hip-opt PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
