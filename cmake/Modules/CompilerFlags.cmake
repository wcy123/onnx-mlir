# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

# Centralized compiler warning configuration
function(set_project_warnings TARGET)
  if(MSVC)
    target_compile_options(${TARGET} PRIVATE
      /W4
      /wd4141  # 'inline' keyword used more than once
      /wd4146  # unary minus applied to unsigned type
      /wd4244  # conversion warnings
      /wd4267  # size_t conversion warnings
    )
  else()
    target_compile_options(${TARGET} PRIVATE
      -Wall
      -Wextra
      -Wno-unused-parameter
    )
  endif()
endfunction()
