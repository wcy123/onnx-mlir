// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP GEMM operations are correctly lowered to LLVM calls to
// rocBLAS library.
//
// This test validates:
// - hip.gemm → llvm.call @rocblas_sgemm (single precision)
// - Type conversion: !hip.context → !llvm.ptr
// - Memref descriptor conversion: memref<...> → !llvm.struct
// - GEMM parameter passing (alpha, beta, transA, transB)
// - Proper function signature for rocBLAS API
//
// Note: This tests the HIP→LLVM lowering, not ONNX→HIP.
// Input is already in HIP dialect (hip.gemm).
//
// GEMM operation: Y = alpha * A * B + beta * C
// Expected: LLVM call to rocblas_sgemm with correct signature
// ============================================================================

// RUN: hip-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @gemm_llvm_test(
      %ctx: !hip.context,
      %A: memref<128x256xf32, 1>,
      %B: memref<256x512xf32, 1>,
      %C: memref<128x512xf32, 1>,
      %Y: memref<128x512xf32, 1>) {
    // CHECK-LABEL: func.func @gemm_llvm_test
    // CHECK-SAME: %[[CTX:.*]]: !llvm.ptr
    // CHECK-SAME: %[[A:.*]]: !llvm.struct
    // CHECK-SAME: %[[B:.*]]: !llvm.struct
    // CHECK-SAME: %[[C:.*]]: !llvm.struct
    // CHECK-SAME: %[[Y:.*]]: !llvm.struct

    // HIP GEMM: Y = alpha * A * B + beta * C
    hip.gemm(%ctx, %A, %B, %C, %Y)
      {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0, transB = 0}
      : (!hip.context, memref<128x256xf32, 1>, memref<256x512xf32, 1>,
         memref<128x512xf32, 1>, memref<128x512xf32, 1>)

    // Should lower to rocBLAS single-precision GEMM call
    // CHECK: llvm.call @rocblas_sgemm
    // CHECK-SAME: (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32)

    return
  }
}
