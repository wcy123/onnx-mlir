# -*- Python -*-
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import re
import platform
import lit.formats
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'ONNX-HIP-EP'

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.onnx_hip_obj_root, 'test')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.onnx_hip_tools_dir, append_path=True)

# Propagate some variables from the host environment.
llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

# Tool substitutions.
tool_dirs = [config.onnx_hip_tools_dir, config.llvm_tools_dir]
tools = [
    'hip-opt',
    'FileCheck',
    'mlir-hip-compiler',
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# Enable features based on available tools
if os.path.isfile(os.path.join(config.onnx_hip_tools_dir, 'hip-opt.exe')) or \
   os.path.isfile(os.path.join(config.onnx_hip_tools_dir, 'hip-opt')):
    config.available_features.add('hip-opt')

# Platform-specific features
if platform.system() == 'Windows':
    config.available_features.add('windows')
elif platform.system() == 'Linux':
    config.available_features.add('linux')
