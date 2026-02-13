# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

# Centralized LLVM/MLIR configuration
function(setup_llvm_mlir)
  find_package(LLVM REQUIRED CONFIG)
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} PARENT_SCOPE)

  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  add_definitions(${LLVM_DEFINITIONS_LIST})

  include_directories(SYSTEM
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS})

  include(AddLLVM)
  include(AddMLIR)
endfunction()
