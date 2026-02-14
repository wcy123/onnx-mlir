// Test ONNX → HIP lowering for GEMM (General Matrix Multiply) operation
// RUN: hip-opt %s --convert-onnx-to-hip | FileCheck %s

module {
  func.func @gemm_test(
      %ctx: !hip.context,
      %A: memref<128x256xf32>,
      %B: memref<256x512xf32>,
      %C: memref<128x512xf32>) -> memref<128x512xf32> {
    // CHECK-LABEL: func.func @gemm_test
    // CHECK-SAME: %[[CTX:.*]]: !hip.context
    // CHECK-SAME: %[[A:.*]]: memref<128x256xf32>
    // CHECK-SAME: %[[B:.*]]: memref<256x512xf32>
    // CHECK-SAME: %[[C:.*]]: memref<128x512xf32>

    // ONNX GEMM: Y = alpha * A * B + beta * C
    %output = "onnx.Gemm"(%A, %B, %C) {
      alpha = 1.0 : f32,
      beta = 1.0 : f32,
      transA = 0 : si64,
      transB = 0 : si64
    } : (memref<128x256xf32>, memref<256x512xf32>, memref<128x512xf32>) -> memref<128x512xf32>

    // CHECK: %[[OUTPUT:.*]] = hip.gemm(%[[CTX]], %[[A]], %[[B]], %[[C]])
    // CHECK-SAME: {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : i64, transB = 0 : i64}
    // CHECK-SAME: : (memref<128x256xf32>, memref<256x512xf32>, memref<128x512xf32>) -> memref<128x512xf32>

    return %output : memref<128x512xf32>
  }
}
