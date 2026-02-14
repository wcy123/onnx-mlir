##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building tests")

enable_testing()

#=============================================================================
# MLIR → DLL Pipeline Tests
#=============================================================================
# E2E test: Compile demo model from DEMO.md and verify it works correctly.
# All outputs go to CMAKE_CURRENT_BINARY_DIR (build directory).
#
# Note: test-model-dll executable is now built in tools/test-model-dll/
# but is still used by tests here. The executable is installed to bin/
# so tests find it in the same location as before.

# Test 1: Compile demo_two_layer_conv.mlir → demo_two_layer.dll
add_test(NAME CompileDemoConvDLL
    COMMAND mlir-hip-compiler
        "${CMAKE_SOURCE_DIR}/tools/hip-opt/demo_two_layer_conv.mlir"
        --from-onnx-mlir
        -o "${CMAKE_CURRENT_BINARY_DIR}/demo_two_layer.dll"
        --mode dll
        --keep
        -v
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Test 2: Load and test demo_two_layer.dll
add_test(NAME TestDemoConvDLL
    COMMAND test-model-dll "${CMAKE_CURRENT_BINARY_DIR}/demo_two_layer.dll" --verbose --validate
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# TestDemoConvDLL depends on CompileDemoConvDLL
set_tests_properties(TestDemoConvDLL PROPERTIES
    DEPENDS CompileDemoConvDLL
)

#=============================================================================
# Runtime Unit Tests (if enabled)
#=============================================================================

if(BUILD_MOCK_RUNTIME)
    message(STATUS "Building Mock Runtime Tests (BUILD_MOCK_RUNTIME=1)")

    # GTest is provided by MorphiZen's FetchContent with OVERRIDE_FIND_PACKAGE
    find_package(GTest QUIET)

    if(TARGET GTest::gtest)
        add_executable(test_mock_runtime
            runtime/test_mock_runtime.cpp
        )

        target_link_libraries(test_mock_runtime PRIVATE
            HipDnnRuntime
            GTest::gtest
            GTest::gtest_main
        )

        target_compile_definitions(test_mock_runtime PRIVATE
            BUILD_MOCK_RUNTIME=1
        )

        target_compile_features(test_mock_runtime PRIVATE cxx_std_17)

        add_test(NAME MockRuntime COMMAND test_mock_runtime)

        set_target_properties(test_mock_runtime PROPERTIES FOLDER "tests")
    else()
        message(STATUS "GTest not found, skipping test_mock_runtime")
    endif()
else()
    message(STATUS "Mock Runtime Tests disabled (set BUILD_MOCK_RUNTIME=1 to enable)")
endif()


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
