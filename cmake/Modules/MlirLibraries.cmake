# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

# Reusable MLIR library groups

# Core MLIR functionality
set(MLIR_CORE_LIBS
  MLIRIR
  MLIRSupport
  MLIRParser
  MLIRPass
  MLIRAnalysis
)

# Dialect libraries
set(MLIR_DIALECT_LIBS
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
)

# Conversion libraries
set(MLIR_CONVERSION_LIBS
  MLIRLLVMCommonConversion
  MLIRTransforms
  MLIRTransformUtils
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
)

# Translation libraries
set(MLIR_LLVM_TRANSLATION_LIBS
  MLIRTargetLLVMIRExport
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
)

# Optimization tools
set(MLIR_OPT_LIBS
  MLIROptLib
)
