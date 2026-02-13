##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building tests")

enable_testing()

#=============================================================================
# MLIR → DLL Pipeline Tests
#=============================================================================
# These tests compile MLIR files to DLLs and verify they work correctly.
# All outputs go to CMAKE_CURRENT_BINARY_DIR (build directory).

# Test: Compile identity_llvm.mlir → identity.dll
add_test(NAME CompileIdentity
    COMMAND mlir-hip-compiler
        "${CMAKE_CURRENT_SOURCE_DIR}/mlir/identity_llvm.mlir"
        -o "${CMAKE_CURRENT_BINARY_DIR}/identity.dll"
        --mode dll
        --keep
        -v
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

#=============================================================================
# ORT Integration Test (optional)
#=============================================================================

find_package(onnxruntime QUIET)

if(TARGET onnxruntime::onnxruntime)
    find_package(GTest QUIET)

    if(TARGET GTest::gtest)
        message(STATUS "Building ORT integration test")

        add_executable(ort_integration_test test_ort_integration.cpp)
        target_link_libraries(ort_integration_test PRIVATE
            onnxruntime::onnxruntime
            GTest::gtest
            GTest::gtest_main
        )

        if(WIN32)
            target_compile_definitions(ort_integration_test PRIVATE "_CRT_SECURE_NO_WARNINGS")
        endif()

        add_test(NAME OrtIntegration COMMAND ort_integration_test)

        set_target_properties(ort_integration_test PROPERTIES FOLDER "tests")
    endif()
else()
    message(STATUS "ONNX Runtime not found, skipping ORT integration test")
endif()
