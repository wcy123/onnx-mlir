##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# HipDnnEpRuntime - Runtime library for ONNX HIP DNN Execution Provider
# Provides HIP/MIOpen/hipBLASLt wrappers called by generated LLVM IR
#
# NOTE: "HipDnnEP" = HIP DNN Execution Provider (this project)
#       "hipDNN" = AMD's separate DNN library (different project - not used here)
#
# CRITICAL BUILD FLOW:
# This CMakeLists.txt implements a non-standard two-stage compilation:
#
# Stage 1 (Build Time - Once):
#   hipdnn_ep_runtime.cpp → [Clang -emit-llvm] → runtime.bc → [xxd.py] → runtime_ir_data.cpp
#   Result: Runtime compiled to LLVM bitcode and embedded as C array in libHipDnnEpRuntime.a
#
# Stage 2 (Model Compilation - Per Model):
#   Generated MLIR IR + embedded runtime.bc → [llvm::Linker] → Combined IR → [LLVM O2] → model.dll
#   Result: Runtime functions inlined into generated code (zero-cost abstraction)
#
# Why this is necessary:
# - Enables cross-module inlining (generated code + Runtime implementation)
# - Accessor functions (hipdnn_ep_get_stream, etc.) become single load instructions
# - Zero runtime overhead while maintaining clean abstraction
# - See: ARCHITECTURE.md Design Decision #7 (LLVM IR Merging)

cmake_minimum_required(VERSION 3.18)

# ============================================================================
# STEP 1: Find Required Tools
# ============================================================================

# Python 3 required for xxd.py (embeds binary data as C array)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Clang REQUIRED for bitcode generation
# Why Clang specifically?
#   - MSVC (cl.exe) doesn't support -emit-llvm flag
#   - GCC's -emit-llvm produces incompatible format
#   - Clang is the reference LLVM frontend
find_program(CLANG_EXECUTABLE NAMES clang clang-cl
    PATHS
        ${LLVM_BINARY_DIR}/bin       # When LLVM built from source
        ${LLVM_TOOLS_BINARY_DIR}     # When LLVM installed via package
)

if(NOT CLANG_EXECUTABLE)
    message(FATAL_ERROR
        "Clang compiler not found. Runtime IR merging is REQUIRED.\n"
        "\n"
        "Why Clang is needed:\n"
        "  - Compiles Runtime to LLVM bitcode (.bc) for zero-cost abstraction\n"
        "  - Enables cross-module inlining (generated code + Runtime)\n"
        "  - MSVC and GCC cannot produce compatible LLVM bitcode\n"
        "\n"
        "To fix:\n"
        "  - Install LLVM with Clang: https://releases.llvm.org/\n"
        "  - Or build LLVM from source with -DLLVM_ENABLE_PROJECTS=clang\n"
        "\n"
        "Searched in:\n"
        "  - ${LLVM_BINARY_DIR}/bin\n"
        "  - ${LLVM_TOOLS_BINARY_DIR}\n"
        "  - System PATH\n")
endif()

message(STATUS "Found Clang for bitcode generation: ${CLANG_EXECUTABLE}")

# llvm-link REQUIRED for combining multiple bitcode files
find_program(LLVM_LINK_EXECUTABLE NAMES llvm-link
    PATHS
        ${LLVM_BINARY_DIR}/bin
        ${LLVM_TOOLS_BINARY_DIR}
)

if(NOT LLVM_LINK_EXECUTABLE)
    message(FATAL_ERROR "llvm-link not found (required for combining runtime modules)")
endif()

message(STATUS "Found llvm-link: ${LLVM_LINK_EXECUTABLE}")

# ============================================================================
# STEP 2: Find ROCm Components (Optional - Mock Mode Available)
# ============================================================================

# ROCm components are optional for development/testing
# If not found, build mock runtime (prints debug messages, no GPU calls)
# Controlled by BUILD_MOCK_RUNTIME variable (default: auto-detect)

if(NOT BUILD_MOCK_RUNTIME)
    find_package(hip QUIET)         # HIP runtime (hipMalloc, hipMemcpy, etc.)
    find_package(MIOpen QUIET)      # DNN operations (convolution)
    find_package(hipblaslt QUIET)   # BLAS operations (matrix multiply)

    if(NOT hip_FOUND OR NOT MIOpen_FOUND OR NOT hipblaslt_FOUND)
        message(WARNING "ROCm components not found. Building mock runtime library for testing.")
        message(WARNING "  To build full runtime, install ROCm and ensure it's in CMAKE_PREFIX_PATH")
        set(BUILD_MOCK_RUNTIME TRUE)
    endif()
else()
    message(STATUS "Building mock runtime (BUILD_MOCK_RUNTIME explicitly set)")
endif()

# ============================================================================
# STEP 3: Generate LLVM Bitcode from Runtime Modules
# ============================================================================

# WHY THIS STEP IS NECESSARY:
# We need Runtime implementation as LLVM bitcode so it can be merged with
# generated code during model compilation. This enables the LLVM optimizer
# to inline Runtime functions across module boundaries.
#
# Example transformation after IR merging + optimization:
#
#   Before (generated code):
#     %stream = call ptr @hipdnn_ep_get_stream(ptr %state)
#
#   After (inlined - zero overhead):
#     %stream_ptr = getelementptr inbounds %struct.RuntimeState, ptr %state, i32 0, i32 0
#     %stream = load ptr, ptr %stream_ptr
#
# This is the "zero-cost abstraction" - clean code with no performance penalty.

# Macro to compile a single C++ file to bitcode
macro(compile_to_bitcode SOURCE_FILE OUTPUT_BC)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_BC}
        COMMAND ${CLANG_EXECUTABLE}
                -c -emit-llvm -O2 -std=c++17
                -I${CMAKE_CURRENT_SOURCE_DIR}
                -DBUILD_MOCK_RUNTIME=$<BOOL:${BUILD_MOCK_RUNTIME}>
                $<$<BOOL:${BUILD_MOCK_RUNTIME}>:-DBUILD_MOCK_RUNTIME=1>
                ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}
                -o ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_BC}
        DEPENDS ${SOURCE_FILE} hipdnn_ep_runtime.h
        COMMENT "Compiling ${SOURCE_FILE} to bitcode (${OUTPUT_BC})"
        VERBATIM
    )
endmacro()

# Core modules (always compiled)
compile_to_bitcode(hipdnn_ep_runtime_state.cpp runtime_state.bc)
compile_to_bitcode(hipdnn_ep_runtime_constants.cpp runtime_constants.bc)
compile_to_bitcode(hipdnn_ep_runtime_tensor.cpp runtime_tensor.bc)

# Conditional modules based on BUILD_MOCK_RUNTIME
if(NOT BUILD_MOCK_RUNTIME)
    # Real implementations
    compile_to_bitcode(hipdnn_ep_runtime_hip.cpp runtime_hip.bc)
    compile_to_bitcode(hipdnn_ep_runtime_miopen.cpp runtime_miopen.bc)
    compile_to_bitcode(hipdnn_ep_runtime_hipblas.cpp runtime_hipblas.bc)

    set(RUNTIME_BC_MODULES
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_state.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_constants.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_tensor.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_hip.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_miopen.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_hipblas.bc
    )
else()
    # Mock implementation
    compile_to_bitcode(hipdnn_ep_runtime_mock.cpp runtime_mock.bc)

    set(RUNTIME_BC_MODULES
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_state.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_constants.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_tensor.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_mock.bc
    )
endif()

# ============================================================================
# STEP 4: Link Bitcode Modules into Single runtime.bc
# ============================================================================

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc

    # Combine all bitcode modules into single runtime.bc
    COMMAND ${LLVM_LINK_EXECUTABLE}
            ${RUNTIME_BC_MODULES}
            -o ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc

    DEPENDS ${RUNTIME_BC_MODULES}

    COMMENT "Linking runtime modules into runtime.bc"
    VERBATIM
)

# ============================================================================
# STEP 5: Create Custom Target for Runtime Bitcode
# ============================================================================

# Runtime library's responsibility ends at runtime.bc generation.
# The Backend project will consume runtime.bc and convert it to C array as needed.

add_custom_target(RuntimeBitcode
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc
)

# Export runtime.bc path for consumers (Backend project)
set(RUNTIME_BC_PATH ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc PARENT_SCOPE)

# ============================================================================
# SUMMARY OF BUILD ARTIFACTS AND PIPELINE
# ============================================================================
#
# Runtime project output:
#   - runtime.bc                   - LLVM bitcode (final artifact)
#
# How runtime.bc is consumed (by Backend project):
#   1. Backend/CMakeLists.txt runs xxd.py to convert runtime.bc → runtime_ir_data.cpp
#   2. runtime_ir_data.cpp contains: runtime_bc_data[] array + runtime_bc_data_size
#   3. Backend/LLVMBackend.cpp includes runtime_ir_data.cpp as source file
#   4. LLVMBackend.lib (containing runtime_bc_data) is linked into mlir-hip-compiler.exe
#
# What happens at model compilation (mlir-hip-compiler):
#   1. LLVMBackend::linkRuntimeModule() reads runtime_bc_data from its own memory
#   2. Parses bitcode to LLVM IR using llvm::parseBitcodeFile()
#   3. Merges runtime IR with generated model IR using llvm::Linker::linkInModule()
#   4. LLVM optimizer inlines runtime functions into model code (zero-cost abstraction)
#   5. Result: model.dll contains inlined runtime code, no separate runtime library needed
#
# Key insight: Runtime code flows through LLVM IR, not through traditional linking:
#   C++ source → bitcode → IR merge → optimization → model.dll
#
# See: doc/RUNTIME-ARCHITECTURE.md for detailed architecture explanation
# ============================================================================
