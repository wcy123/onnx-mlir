# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

# Centralized LLD discovery
function(find_and_setup_lld)
  find_library(LLD_COFF_LIB NAMES lldCOFF PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
  find_library(LLD_ELF_LIB NAMES lldELF PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
  find_library(LLD_COMMON_LIB NAMES lldCommon PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)

  if(LLD_COFF_LIB AND LLD_ELF_LIB AND LLD_COMMON_LIB)
    message(STATUS "LLD libraries found")
    set(HAVE_LLD ON PARENT_SCOPE)
    set(LLD_LIBRARIES ${LLD_COFF_LIB} ${LLD_ELF_LIB} ${LLD_COMMON_LIB} PARENT_SCOPE)
  else()
    message(WARNING "LLD libraries not found")
    set(HAVE_LLD OFF PARENT_SCOPE)
  endif()
endfunction()
