##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Centralized LLD discovery (REQUIRED)
function(find_and_setup_lld)
  find_library(LLD_COFF_LIB NAMES lldCOFF PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
  find_library(LLD_ELF_LIB NAMES lldELF PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
  find_library(LLD_COMMON_LIB NAMES lldCommon PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)

  if(NOT LLD_COFF_LIB OR NOT LLD_ELF_LIB OR NOT LLD_COMMON_LIB)
    message(FATAL_ERROR "LLD libraries are REQUIRED but not found.\n"
            "  Rebuild LLVM with: -DLLVM_ENABLE_PROJECTS=\"mlir;lld\"\n"
            "  Missing: lldCOFF=${LLD_COFF_LIB} lldELF=${LLD_ELF_LIB} lldCommon=${LLD_COMMON_LIB}")
  endif()

  message(STATUS "LLD libraries found")
  message(STATUS "  lldCOFF: ${LLD_COFF_LIB}")
  message(STATUS "  lldELF: ${LLD_ELF_LIB}")
  message(STATUS "  lldCommon: ${LLD_COMMON_LIB}")
  set(LLD_LIBRARIES ${LLD_COFF_LIB} ${LLD_ELF_LIB} ${LLD_COMMON_LIB} PARENT_SCOPE)
endfunction()
