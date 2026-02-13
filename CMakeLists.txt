##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.29)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
project(
  onnx-hipdnn-ep
  VERSION 1.0.0
  LANGUAGES C CXX)

# Add custom CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

# Set C++ standard for entire project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(BUILD_HIP_DIALECT "Build HIP MLIR dialect library" ON)
option(BUILD_HIP_OPT_TOOL "Build hip-opt standalone tool" OFF)
option(BUILD_MLIR_COMPILER "Build MLIR compiler (Level-1 Pass)" ON)
option(BUILD_MLIR_HIP_COMPILER "Build standalone MLIR-HIP compiler tool" ON)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)

# Component builds
if(BUILD_HIP_DIALECT)
  add_subdirectory(lib/HipDialect)
endif()

# Add runtime library (required for DLL generation)
add_subdirectory(lib/Runtime)

# Add backend library (LLVM Backend + DLL Linker)
add_subdirectory(lib/Backend)

if(BUILD_HIP_OPT_TOOL)
  add_subdirectory(tools/hip-opt)
endif()

if(BUILD_MLIR_COMPILER)
  add_subdirectory(level-1-pass-mlir-compiler)
endif()

if(BUILD_MLIR_HIP_COMPILER)
  add_subdirectory(tools/mlir-hip-compiler)
endif()

add_subdirectory(test)
