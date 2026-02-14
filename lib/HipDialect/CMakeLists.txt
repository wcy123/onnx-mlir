##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Setup LLVM/MLIR
include(LLVMSetup)
setup_llvm_mlir()

message(STATUS "Building HIP MLIR Dialect Library")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

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
  GenerateInterfacePass.cpp
  MemoryPoolingPass.cpp
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
include(MlirLibraries)
target_link_libraries(HipDialect PUBLIC
  ${MLIR_CORE_LIBS}
  ${MLIR_DIALECT_LIBS}
  ${MLIR_CONVERSION_LIBS}
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  OMONNXOps  # ONNX dialect from onnx-mlir (required)
  ${llvm_libs}
)

# Add onnx-mlir include directories
target_include_directories(HipDialect PUBLIC
  ${CMAKE_BINARY_DIR}/3rd-party/onnx-mlir/src/Dialect/ONNX  # Generated headers
  ${CMAKE_SOURCE_DIR}/3rd-party/onnx-mlir/src               # Source headers
)

message(STATUS "HipDialect: Linked with onnx-mlir ONNX dialect")

# Set folder for IDE organization
set_target_properties(HipDialect PROPERTIES FOLDER "mlir")

# Export include directories for downstream targets
target_include_directories(HipDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
