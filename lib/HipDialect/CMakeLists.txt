##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Find LLVM and MLIR (already configured by top-level cmake/deps.cmake)
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Building HIP MLIR Dialect Library")
message(STATUS "  LLVM_DIR: ${LLVM_DIR}")
message(STATUS "  MLIR_DIR: ${MLIR_DIR}")

# Add LLVM and MLIR definitions
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Add LLVM TableGen function
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# Set TableGen executables with absolute paths
# Visual Studio MSBuild doesn't inherit PATH, so we need absolute paths to mlir-tblgen/llvm-tblgen.
# Try LLVM_TOOLS_BINARY_DIR first (FetchContent), then CMAKE_INSTALL_PREFIX/bin (pre-installed).
if(EXISTS "${LLVM_TOOLS_BINARY_DIR}/mlir-tblgen${CMAKE_EXECUTABLE_SUFFIX}")
  set(MLIR_TABLEGEN_EXE "${LLVM_TOOLS_BINARY_DIR}/mlir-tblgen${CMAKE_EXECUTABLE_SUFFIX}")
else()
  set(MLIR_TABLEGEN_EXE "${CMAKE_INSTALL_PREFIX}/bin/mlir-tblgen${CMAKE_EXECUTABLE_SUFFIX}")
endif()

if(EXISTS "${LLVM_TOOLS_BINARY_DIR}/llvm-tblgen${CMAKE_EXECUTABLE_SUFFIX}")
  set(LLVM_TABLEGEN_EXE "${LLVM_TOOLS_BINARY_DIR}/llvm-tblgen${CMAKE_EXECUTABLE_SUFFIX}")
else()
  set(LLVM_TABLEGEN_EXE "${CMAKE_INSTALL_PREFIX}/bin/llvm-tblgen${CMAKE_EXECUTABLE_SUFFIX}")
endif()

message(STATUS "Found mlir-tblgen: ${MLIR_TABLEGEN_EXE}")
message(STATUS "Found llvm-tblgen: ${LLVM_TABLEGEN_EXE}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Generate dialect definition
set(LLVM_TARGET_DEFINITIONS HipDialect.td)
mlir_tablegen(HipDialect.h.inc -gen-dialect-decls -dialect=hip)
mlir_tablegen(HipDialect.cpp.inc -gen-dialect-defs -dialect=hip)
add_public_tablegen_target(HipDialectIncGen)

# Generate type definitions
set(LLVM_TARGET_DEFINITIONS HipTypes.td)
mlir_tablegen(HipTypes.h.inc -gen-typedef-decls -typedefs-dialect=hip)
mlir_tablegen(HipTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=hip)
add_public_tablegen_target(HipTypesIncGen)

# Generate operation definitions
set(LLVM_TARGET_DEFINITIONS HipOps.td)
mlir_tablegen(HipOps.h.inc -gen-op-decls)
mlir_tablegen(HipOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(HipOpsIncGen)

# Create HIP dialect library
add_library(HipDialect STATIC
  HipDialect.cpp
  HipToLLVM.cpp
  OnnxToHip.cpp
)

# Add dependencies on TableGen targets
add_dependencies(HipDialect
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
)

# Link LLVM components
llvm_map_components_to_libnames(llvm_libs Support Core)

# Link against MLIR libraries and onnx-mlir
target_link_libraries(HipDialect PUBLIC
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
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  OMONNXOps  # ONNX dialect from onnx-mlir (required)
  ${llvm_libs}
)

# Add onnx-mlir include directories
target_include_directories(HipDialect PUBLIC
  ${CMAKE_BINARY_DIR}/3rd-party/onnx-mlir/src/Dialect/ONNX  # Generated headers
  ${CMAKE_SOURCE_DIR}/3rd-party/onnx-mlir/src               # Source headers
)

message(STATUS "HipDialect: Linked with onnx-mlir ONNX dialect")

# Set C++ standard
set_target_properties(HipDialect PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  FOLDER "mlir"
)

# Export include directories for downstream targets
target_include_directories(HipDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
