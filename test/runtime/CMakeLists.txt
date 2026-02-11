# Standalone CMakeLists.txt for testing mock runtime in isolation
# This allows testing without building the entire LLVM/MLIR stack

cmake_minimum_required(VERSION 3.18)
project(MockRuntimeTest CXX)

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

# Add the mock runtime test executable
add_executable(test_mock_runtime
    test_mock_runtime.cpp
)

target_link_libraries(test_mock_runtime PRIVATE
    HipDnnRuntime
)

target_compile_definitions(test_mock_runtime PRIVATE
    BUILD_MOCK_RUNTIME=1
)

# Add the runtime state test executable
add_executable(test_runtime_state
    test_runtime_state.cpp
)

target_link_libraries(test_runtime_state PRIVATE
    HipDnnRuntime
)

target_compile_definitions(test_runtime_state PRIVATE
    BUILD_MOCK_RUNTIME=1
)

# Add the inference helpers test executable
add_executable(test_inference_helpers
    test_inference_helpers.cpp
)

target_link_libraries(test_inference_helpers PRIVATE
    HipDnnRuntime
)

target_compile_definitions(test_inference_helpers PRIVATE
    BUILD_MOCK_RUNTIME=1
)

# Enable testing
enable_testing()
add_test(NAME MockRuntime COMMAND test_mock_runtime)
add_test(NAME RuntimeState COMMAND test_runtime_state)
add_test(NAME InferenceHelpers COMMAND test_inference_helpers)
