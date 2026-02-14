##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building tests")

enable_testing()

#=============================================================================
# LIT-based Unit Tests (MLIR pass testing)
#=============================================================================

if(ONNX_HIP_ENABLE_LIT_TESTS)
    message(STATUS "Configuring LIT tests")

    # Configure the site specific configuration file
    set(ONNX_HIP_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        @ONLY
    )

    # Add LIT test suite as a CTest target
    add_test(
        NAME LitTests
        COMMAND ${LLVM_LIT_EXECUTABLE} -v ${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # Add a custom target for running LIT tests manually
    add_custom_target(check-onnx-hip-lit
        COMMAND ${LLVM_LIT_EXECUTABLE} -v ${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running LIT tests"
        USES_TERMINAL
    )

    message(STATUS "LIT tests configured at: ${CMAKE_CURRENT_BINARY_DIR}")
else()
    message(STATUS "LIT tests disabled (llvm-lit not found or ONNX_HIP_INCLUDE_LIT_TESTS=OFF)")
endif()

#=============================================================================
# E2E Integration Tests (slow, compile + execute)
#=============================================================================

add_subdirectory(e2e)

#=============================================================================
# Runtime Unit Tests (optional, requires GTest)
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
