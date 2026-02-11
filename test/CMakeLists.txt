##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building MLIR tests")

# GTest is provided by MorphiZen's FetchContent with OVERRIDE_FIND_PACKAGE
find_package(GTest REQUIRED)

# ORT Integration test (for MorphiZen EP testing with MLIR pass)
# This test links against ONNX Runtime and can trigger the MLIR pass

# Try to find ONNX Runtime using find_package
find_package(onnxruntime QUIET)

if(TARGET onnxruntime::onnxruntime)
  message(STATUS "Building ORT integration test (using onnxruntime::onnxruntime target)")

  add_executable(ort_integration_test test_ort_integration.cpp)
  target_link_libraries(ort_integration_test PRIVATE
    onnxruntime::onnxruntime
    GTest::gtest
    GTest::gtest_main
  )
  target_compile_features(ort_integration_test PRIVATE cxx_std_17)
  if(WIN32)
    target_compile_definitions(ort_integration_test PRIVATE "_CRT_SECURE_NO_WARNINGS")
  endif()
  set_target_properties(ort_integration_test PROPERTIES FOLDER "mlir-tests")
else()
  message(STATUS "ONNX Runtime not found, skipping ORT integration test")
  message(STATUS "  Hint: Ensure onnxruntime is installed and CMAKE_PREFIX_PATH includes the installation directory")
endif()

# Mock Runtime Test - Tests runtime library without ROCm dependencies
# Only built when BUILD_MOCK_RUNTIME=1 is set
if(BUILD_MOCK_RUNTIME)
    message(STATUS "Building Mock Runtime Test (BUILD_MOCK_RUNTIME=1)")

    add_executable(test_mock_runtime
        runtime/test_mock_runtime.cpp
    )

    target_link_libraries(test_mock_runtime PRIVATE
        HipDnnRuntime
    )

    target_compile_definitions(test_mock_runtime PRIVATE
        BUILD_MOCK_RUNTIME=1
    )

    # Enable testing
    enable_testing()
    add_test(NAME MockRuntime COMMAND test_mock_runtime)

    set_target_properties(test_mock_runtime PROPERTIES FOLDER "mlir-tests")
else()
    message(STATUS "Mock Runtime Test disabled (set BUILD_MOCK_RUNTIME=1 to enable)")
endif()

# Integration tests for MLIR to DLL pipeline
add_subdirectory(integration)
