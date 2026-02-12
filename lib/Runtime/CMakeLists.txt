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
set(ENABLE_RUNTIME_IR_MERGING ON)

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
# STEP 5: Embed Bitcode as C Array
# ============================================================================

# WHY EMBED AS C ARRAY:
# The LLVM bitcode needs to be available at model compilation time (not just
# build time). By embedding it as a C array in the EP DLL, it's always available
# when the model compiler runs.
#
# Alternative approaches considered and rejected:
#   - External .bc file: Deployment complexity, file path management
#   - Resource embedding: Platform-specific, complex extraction
#   - Base64 string: Larger size, decode overhead
#
# C array approach:
#   - Cross-platform (works on Windows, Linux, macOS)
#   - Zero runtime overhead (just a pointer dereference)
#   - Simple to use (extern const unsigned char runtime_bc_data[])

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp

    # Convert binary bitcode to C array using xxd.py
    # Input:  runtime.bc (binary LLVM bitcode)
    # Output: runtime_ir_data.cpp containing:
    #           extern "C" const unsigned char runtime_bc_data[] = { 0x42, 0x43, ... };
    #           extern "C" const size_t runtime_bc_data_size = 12345;

    # Step 1: Generate C array with xxd.py
    COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/3rd-party/morphizen/tools/xxd.py
            --var runtime_bc_data                                    # Variable name
            --output ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp.tmp # Temp output
            ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc                  # Input bitcode

    # Step 2: Post-process to export symbol (static const -> extern "C" const)
    # xxd.py generates: static const unsigned char runtime_bc_data[]
    # We need:          extern "C" const unsigned char runtime_bc_data[]
    # This ensures the symbol is exported from HipDnnEpRuntime.lib for linking
    COMMAND ${CMAKE_COMMAND} -E echo "Post-processing runtime_ir_data.cpp to export symbol..."
    COMMAND ${Python3_EXECUTABLE} -c
            "import sys; data = open('${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp.tmp', 'r').read(); open('${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp', 'w').write(data.replace('static const unsigned char', 'extern \"C\" const unsigned char'))"
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp.tmp

    # Regenerate if bitcode changes
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc

    COMMENT "Stage 2: Embedding Runtime bitcode as C array (runtime_ir_data.cpp)"
    VERBATIM
)

# Track generated source for build system
set(RUNTIME_IR_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp)

# ============================================================================
# STEP 6: Create Runtime Static Library
# ============================================================================

# CRITICAL: hipdnn_ep_runtime.cpp is NOT compiled to native code here (tricky!)
#
# This library contains only embedded LLVM bitcode data, not native runtime functions.
#
# Why native runtime functions are not needed:
#
#   The EP DLL never calls runtime functions directly. Runtime functions are only
#   called by model DLLs, and they get the code through LLVM IR merging, not linking.
#
# How runtime code reaches model DLLs:
#
#   Build time (once):
#     hipdnn_ep_runtime.cpp → [Clang -emit-llvm] → runtime.bc
#                           → [xxd.py embed] → runtime_ir_data.cpp (C array)
#                           → [compile] → libHipDnnEpRuntime.a (embedded bitcode)
#
#   Model compilation (per model):
#     1. Model compiler reads runtime_bc_data[] from EP DLL memory
#     2. Parses LLVM bitcode into runtime IR module
#     3. Merges runtime IR with generated model IR (llvm::Linker)
#     4. LLVM optimizer inlines runtime functions into model code
#     5. Result: model.dll contains inlined runtime code (zero overhead)
#
# Why hipdnn_ep_runtime.cpp was previously listed here:
#
#   It appeared to be needed because "libraries need source files", but this is wrong.
#   The runtime source is already compiled to bitcode at STEP 3 (line 113).
#   Compiling it again to native code creates dead code that's never called.
#
# What this library actually contains:
#
#   - runtime_ir_data.cpp: Embedded LLVM bitcode as C array
#   - No native runtime functions (they're inlined into each model.dll from bitcode)
#
# See: ARCHITECTURE.md Design Decision #7 for the full IR merging explanation

add_library(HipDnnEpRuntime STATIC
    ${RUNTIME_IR_SOURCES}    # runtime_ir_data.cpp - Embedded bitcode as C array
)

target_include_directories(HipDnnEpRuntime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}    # Export hipdnn_ep_runtime.h
)

# ============================================================================
# STEP 7: No ROCm Linking Required
# ============================================================================

# Why no target_link_libraries() for HIP/MIOpen/hipBLASLt:
#
# 1. HipDnnEpRuntime only contains embedded bitcode data (runtime_ir_data.cpp), not native code
# 2. Real/mock choice is already baked into runtime.bc at STEP 3/4 (via -DBUILD_MOCK_RUNTIME)
# 3. EP DLL never calls runtime functions - confirmed by codebase exploration
# 4. Model DLLs get runtime code through IR merging, not CMake linking
#
# The runtime functions (with HIP/MIOpen calls) are inlined into each model DLL
# during model compilation via llvm::Linker. Model DLLs link ROCm libraries at
# their own compilation time (in LLVMBackend), not through this CMake target.

# ============================================================================
# STEP 8: Compiler Settings
# ============================================================================

target_compile_features(HipDnnEpRuntime PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(HipDnnEpRuntime PRIVATE /W4)   # Warning level 4
else()
    target_compile_options(HipDnnEpRuntime PRIVATE -Wall -Wextra)
endif()

# ============================================================================
# STEP 9: Install Rules - NOT NEEDED (Static Linking)
# ============================================================================

# NOTE: Install rules are intentionally omitted for HipDnnEpRuntime
#
# Why install is NOT needed:
#   1. libHipDnnEpRuntime.a is STATICALLY linked into the EP DLL at build time
#   2. The embedded bitcode is accessed from EP DLL's own memory (not external file)
#   3. External consumers only need:
#      - EP DLL (onnxruntime_morphizen_ep.dll) - contains Runtime + bitcode
#      - Model DLLs (generated, also contain inlined Runtime code)
#   4. No one needs to link against libHipDnnEpRuntime.a separately
#
# If install is needed (for custom tooling or testing):
#   Uncomment the following:
#
# install(TARGETS HipDnnEpRuntime
#     ARCHIVE DESTINATION lib
# )
# install(FILES hipdnn_ep_runtime.h
#     DESTINATION include
# )

# ============================================================================
# SUMMARY OF BUILD ARTIFACTS
# ============================================================================
#
# Build-time outputs:
#   1. runtime.bc                  - LLVM bitcode (intermediate)
#   2. runtime_ir_data.cpp         - Embedded bitcode as C array (intermediate)
#   3. libHipDnnEpRuntime.a        - Static library (linked into EP DLL)
#
# What gets embedded in EP DLL:
#   - Runtime implementation (native code from hipdnn_ep_runtime.cpp)
#   - Embedded bitcode (data from runtime_ir_data.cpp)
#
# What happens at model compilation:
#   1. Model compiler extracts runtime_bc_data from EP DLL memory
#   2. Parses it to LLVM IR using llvm::parseBitcodeFile()
#   3. Merges with generated IR using llvm::Linker::linkInModule()
#   4. LLVM optimizer inlines Runtime functions
#   5. Result: Zero-overhead abstraction in final model.dll
#
# See: doc/RUNTIME-ARCHITECTURE.md for detailed pipeline diagram
# ============================================================================
