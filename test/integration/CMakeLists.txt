# Integration tests for MLIR to DLL generation pipeline

cmake_minimum_required(VERSION 3.18)

# Build EndToEndTest executable
add_executable(EndToEndTest
    EndToEndTest.cpp
)

# No special libraries needed - just platform DLL loading
if(WIN32)
    # Windows: No extra libs needed (uses LoadLibrary)
else()
    # Linux: Link dl for dlopen
    target_link_libraries(EndToEndTest PRIVATE dl)
endif()

# Compiler flags
target_compile_features(EndToEndTest PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(EndToEndTest PRIVATE /W4)
else()
    target_compile_options(EndToEndTest PRIVATE -Wall -Wextra)
endif()

# Install test executable
install(TARGETS EndToEndTest
    RUNTIME DESTINATION bin/test
)

# Install verification scripts
install(FILES
    verify_dll_exports.bat
    verify_dll_exports.sh
    DESTINATION bin/test
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)
