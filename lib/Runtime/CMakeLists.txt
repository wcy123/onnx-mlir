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
# STEP 3: Generate LLVM Bitcode from Runtime Source (CRITICAL)
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

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc

    # Compile Runtime to LLVM bitcode (not native object code)
    COMMAND ${CLANG_EXECUTABLE}
            -c                              # Compile only (no linking)
            -emit-llvm                      # CRITICAL: Output LLVM bitcode instead of object file
            -O2                             # Optimize (makes inlining more effective)
            -std=c++17                      # C++17 features used in Runtime
            -I${CMAKE_CURRENT_SOURCE_DIR}  # Include path for headers

            # Pass BUILD_MOCK_RUNTIME flag to conditionally compile mock or real Runtime
            -DBUILD_MOCK_RUNTIME=$<BOOL:${BUILD_MOCK_RUNTIME}>
            $<$<BOOL:${BUILD_MOCK_RUNTIME}>:-DBUILD_MOCK_RUNTIME=1>

            ${CMAKE_CURRENT_SOURCE_DIR}/hipdnn_ep_runtime.cpp
            -o ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc

    # Recompile if Runtime source changes
    DEPENDS hipdnn_ep_runtime.cpp hipdnn_ep_runtime.h

    COMMENT "Stage 1: Compiling Runtime to LLVM bitcode (runtime.bc)"
    VERBATIM
)

# ============================================================================
# STEP 4: Embed Bitcode as C Array
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
    COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/3rd-party/morphizen/tools/xxd.py
            --var runtime_bc_data                                    # Variable name
            --output ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp # Output C++ file
            ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc                  # Input bitcode

    # Regenerate if bitcode changes
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc

    COMMENT "Stage 2: Embedding Runtime bitcode as C array (runtime_ir_data.cpp)"
    VERBATIM
)

# Track generated source for build system
set(RUNTIME_IR_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp)

# ============================================================================
# STEP 5: Create Runtime Static Library
# ============================================================================

# Build static library containing:
#   1. hipdnn_ep_runtime.cpp - Runtime implementation (normal C++ compilation)
#   2. runtime_ir_data.cpp - Embedded bitcode as C array
#
# This library is linked into the EP DLL. At model compilation time, the
# model compiler extracts the embedded bitcode and merges it with generated IR.

add_library(HipDnnEpRuntime STATIC
    hipdnn_ep_runtime.cpp    # Runtime implementation (compiled to native code)
    ${RUNTIME_IR_SOURCES}    # Embedded bitcode (data only, not compiled)
)

target_include_directories(HipDnnEpRuntime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}    # Export hipdnn_ep_runtime.h
    PRIVATE
        ${ROCM_PATH}/include           # HIP/MIOpen/hipBLAS headers (if available)
)

# ============================================================================
# STEP 6: Link Dependencies
# ============================================================================

if(NOT BUILD_MOCK_RUNTIME)
    # Real Runtime: Link ROCm libraries
    # These will be dynamically loaded at runtime by the generated DLL
    target_link_libraries(HipDnnEpRuntime
        PUBLIC
            hip::host       # HIP runtime (hipMalloc, hipMemcpy, hipStreamCreate, etc.)
            MIOpen          # DNN operations (miopenConvolutionForward, etc.)
            hipblaslt       # BLAS operations (hipblasLtMatmul, etc.)
    )
else()
    # Mock Runtime: No ROCm dependencies
    # All GPU functions are stubbed (malloc/free + debug prints)
    target_compile_definitions(HipDnnEpRuntime PRIVATE BUILD_MOCK_RUNTIME=1)
endif()

# ============================================================================
# STEP 7: Compiler Settings
# ============================================================================

target_compile_features(HipDnnEpRuntime PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(HipDnnEpRuntime PRIVATE /W4)   # Warning level 4
else()
    target_compile_options(HipDnnEpRuntime PRIVATE -Wall -Wextra)
endif()

# ============================================================================
# STEP 8: Install Rules - NOT NEEDED (Static Linking)
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
