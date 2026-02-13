##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
find_package(glog CONFIG REQUIRED)

# Setup LLVM/MLIR
include(LLVMSetup)
setup_llvm_mlir()

set(LIB_NAME morphizen-level1-pass-mlir-compiler)
add_library(${LIB_NAME} STATIC src/pass_main.cpp)

include(MlirLibraries)
target_link_libraries(${LIB_NAME} PUBLIC
	morphizen::morphizen-core-static
	glog::glog
	${MLIR_CORE_LIBS}
	${MLIR_DIALECT_LIBS}

	# HipDialect passes
	HipDialect

	# Backend libraries (LLVM Backend + DLL Linker)
	LLVMBackend
	DLLLinker

	# Runtime library
	HipDnnEpRuntime
)
target_include_directories(${LIB_NAME} PRIVATE
	include
	${LLVM_INCLUDE_DIRS}
	${MLIR_INCLUDE_DIRS}
	${CMAKE_SOURCE_DIR}/lib/HipDialect
	${CMAKE_SOURCE_DIR}/lib/Backend
	${CMAKE_SOURCE_DIR}/lib/Runtime
)
target_compile_definitions(${LIB_NAME} PRIVATE -DOUTPUT_NAME="morphizen-level1-pass-mlir-compiler")
target_link_libraries(${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>")
set_target_properties(${LIB_NAME} PROPERTIES
	FOLDER "mlir"
)
