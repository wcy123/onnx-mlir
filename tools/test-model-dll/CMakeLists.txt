##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.18)
project(test-model-dll)

# Create executable
add_executable(test-model-dll test-model-dll.cpp)

# C++17 requirement
target_compile_features(test-model-dll PRIVATE cxx_std_17)

# Platform-specific libraries
if(WIN32)
    # Windows: No extra libs needed (uses LoadLibrary)
else()
    # Linux: Link dl for dlopen
    target_link_libraries(test-model-dll PRIVATE dl)
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(test-model-dll PRIVATE /W4)
else()
    target_compile_options(test-model-dll PRIVATE -Wall -Wextra)
endif()

# Install to bin directory
install(TARGETS test-model-dll DESTINATION bin)
