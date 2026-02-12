##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Runtime library for generated DLL
# Provides HIP/MIOpen/hipBLASLt wrappers called by generated LLVM IR

cmake_minimum_required(VERSION 3.18)

# Find Python3 for xxd.py embedding script
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Check for Clang compiler (required for bitcode generation)
# MSVC (cl.exe) doesn't support -emit-llvm flag
find_program(CLANG_EXECUTABLE NAMES clang clang-cl
    PATHS
        ${LLVM_BINARY_DIR}/bin
        ${LLVM_TOOLS_BINARY_DIR}
        C:/Develop/m/local/bin
        C:/Develop/m/build/llvm-project/Debug/bin
        C:/Develop/m/build/llvm-project/Release/bin
    NO_DEFAULT_PATH
)

if(NOT CLANG_EXECUTABLE)
    find_program(CLANG_EXECUTABLE NAMES clang clang-cl)
endif()

if(CLANG_EXECUTABLE)
    message(STATUS "Found Clang for bitcode generation: ${CLANG_EXECUTABLE}")
    set(ENABLE_RUNTIME_IR_MERGING ON)
else()
    message(WARNING "Clang not found. Runtime IR merging disabled (zero-cost abstraction not available).")
    message(WARNING "  To enable IR merging, install Clang or build LLVM with Clang enabled.")
    set(ENABLE_RUNTIME_IR_MERGING OFF)
endif()

# Find ROCm components (optional - mock versions used if not available)
# Allow explicit BUILD_MOCK_RUNTIME=1 to force mock mode
if(NOT BUILD_MOCK_RUNTIME)
    find_package(hip QUIET)
    find_package(MIOpen QUIET)
    find_package(hipblaslt QUIET)

    if(NOT hip_FOUND OR NOT MIOpen_FOUND OR NOT hipblaslt_FOUND)
        message(WARNING "ROCm components not found. Building mock runtime library for testing.")
        message(WARNING "  To build full runtime, install ROCm and ensure it's in CMAKE_PREFIX_PATH")
        set(BUILD_MOCK_RUNTIME TRUE)
    endif()
else()
    message(STATUS "Building mock runtime (BUILD_MOCK_RUNTIME explicitly set)")
endif()

# Generate LLVM bitcode and embed it (only if Clang is available)
if(ENABLE_RUNTIME_IR_MERGING)
    # Generate LLVM bitcode from Runtime source for IR merging
    # This enables zero-cost abstraction via cross-module inlining
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc
        COMMAND ${CLANG_EXECUTABLE}
                -c -emit-llvm -O2
                -std=c++17
                -I${CMAKE_CURRENT_SOURCE_DIR}
                -DBUILD_MOCK_RUNTIME=$<BOOL:${BUILD_MOCK_RUNTIME}>
                $<$<BOOL:${BUILD_MOCK_RUNTIME}>:-DBUILD_MOCK_RUNTIME=1>
                ${CMAKE_CURRENT_SOURCE_DIR}/hipdnn_ep_runtime.cpp
                -o ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc
        DEPENDS hipdnn_ep_runtime.cpp hipdnn_ep_runtime.h
        COMMENT "Compiling Runtime to LLVM bitcode"
        VERBATIM
    )

    # Embed bitcode as C array using xxd.py
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/3rd-party/morphizen/tools/xxd.py
                --var runtime_bc_data
                --output ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp
                ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc
        COMMENT "Embedding Runtime bitcode as C array"
        VERBATIM
    )

    set(RUNTIME_IR_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp)
else()
    # Create dummy runtime_ir_data.cpp with empty array
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data_stub.cpp
        "// Stub file - Runtime IR merging disabled (Clang not found)\n"
        "extern \"C\" const unsigned char runtime_bc_data[] = {0x00};\n"
    )
    set(RUNTIME_IR_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data_stub.cpp)
endif()

# Create static library
add_library(HipDnnRuntime STATIC
    hipdnn_ep_runtime.cpp
    ${RUNTIME_IR_SOURCES}
)

target_include_directories(HipDnnRuntime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${ROCM_PATH}/include
)

if(NOT BUILD_MOCK_RUNTIME)
    target_link_libraries(HipDnnRuntime
        PUBLIC
            hip::host
            MIOpen
            hipblaslt
    )
else()
    # Mock runtime doesn't link ROCm libraries
    target_compile_definitions(HipDnnRuntime PRIVATE BUILD_MOCK_RUNTIME=1)
endif()

# Compiler flags
target_compile_features(HipDnnRuntime PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(HipDnnRuntime PRIVATE /W4)
else()
    target_compile_options(HipDnnRuntime PRIVATE -Wall -Wextra)
endif()

# Install rules
install(TARGETS HipDnnRuntime
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES hipdnn_ep_runtime.h
    DESTINATION include
)
