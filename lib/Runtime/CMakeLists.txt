##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Runtime library for generated DLL
# Provides HIP/MIOpen/hipBLASLt wrappers called by generated LLVM IR

cmake_minimum_required(VERSION 3.18)

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

# Create static library
add_library(HipDnnRuntime STATIC
    hip_ep_runtime.cpp
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

install(FILES hip_ep_runtime.h
    DESTINATION include
)
