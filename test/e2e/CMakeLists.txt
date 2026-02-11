# End-to-End Test CMakeLists.txt
# Tests the complete MLIR to DLL pipeline
#
# Build with out-of-source pattern:
#   cmake -S . -B ../../build/onnx-hipdnn-ep/test/e2e -G "Visual Studio 17 2022" -A x64
#   cmake --build ../../build/onnx-hipdnn-ep/test/e2e --config Release

cmake_minimum_required(VERSION 3.18)
project(E2ETests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force mock runtime mode
set(BUILD_MOCK_RUNTIME TRUE)

# Add the runtime library
add_library(HipDnnRuntime STATIC
    ../../lib/Runtime/hip_ep_runtime.cpp
)

target_include_directories(HipDnnRuntime
    PUBLIC
        ../../lib/Runtime
)

target_compile_definitions(HipDnnRuntime PRIVATE BUILD_MOCK_RUNTIME=1)

if(MSVC)
    target_compile_options(HipDnnRuntime PRIVATE /W4)
else()
    target_compile_options(HipDnnRuntime PRIVATE -Wall -Wextra)
endif()

# Add the DLL loading test executable
add_executable(test_dll_loading
    test_dll_loading.cpp
)

target_link_libraries(test_dll_loading PRIVATE
    HipDnnRuntime
)

target_compile_definitions(test_dll_loading PRIVATE
    BUILD_MOCK_RUNTIME=1
)

# Add the accuracy validation test executable
add_executable(test_accuracy
    test_accuracy.cpp
)

target_compile_definitions(test_accuracy PRIVATE
    BUILD_MOCK_RUNTIME=1
)

# Add the performance benchmarking test executable
add_executable(test_performance
    test_performance.cpp
)

target_compile_definitions(test_performance PRIVATE
    BUILD_MOCK_RUNTIME=1
)

# Add target to create DLL from pre-compiled object file
# This uses the test_identity.obj that was created from test_identity_manual.mlir
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_identity.obj")
    add_library(test_identity_dll SHARED
        test_identity.obj
    )

    target_link_libraries(test_identity_dll PRIVATE
        HipDnnRuntime
    )

    set_target_properties(test_identity_dll PROPERTIES
        OUTPUT_NAME "test_identity"
        PREFIX ""
        SUFFIX ".dll"
        LINKER_LANGUAGE CXX
    )

    # Export the interface functions
    if(MSVC)
        set_target_properties(test_identity_dll PROPERTIES
            LINK_FLAGS "/EXPORT:inference_init /EXPORT:inference_compute /EXPORT:inference_cleanup"
        )
    endif()

    message(STATUS "test_identity.obj found - will build test_identity.dll")
else()
    message(STATUS "test_identity.obj not found - skipping DLL build")
    message(STATUS "  Run: mlir-translate --mlir-to-llvmir test_identity_manual.mlir -o test_identity.ll")
    message(STATUS "  Then: llc test_identity.ll -filetype=obj -o test_identity.obj")
endif()
