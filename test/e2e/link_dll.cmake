# Simple CMake script to link test_identity.obj to DLL
cmake_minimum_required(VERSION 3.18)
project(LinkTestDLL)

# Create the DLL from the object file
add_library(test_identity_dll SHARED test_identity.obj)

# Link with runtime library
target_link_libraries(test_identity_dll
    PRIVATE
    "C:/Develop/m/build/onnx-hipdnn-ep/lib/Runtime/HipDnnRuntime.lib"
)

# Set output name
set_target_properties(test_identity_dll PROPERTIES
    OUTPUT_NAME "test_identity"
    PREFIX ""
    SUFFIX ".dll"
)

# Export the interface functions
if(MSVC)
    set_target_properties(test_identity_dll PROPERTIES
        LINK_FLAGS "/EXPORT:inference_init /EXPORT:inference_compute /EXPORT:inference_cleanup"
    )
endif()
