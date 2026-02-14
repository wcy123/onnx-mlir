// Test ONNX → HIP lowering for Conv operation
// RUN: hip-opt %s --convert-onnx-to-hip | FileCheck %s

module {
  func.func @conv_test(%state: !hip.handle, %input: memref<1x3x224x224xf32>, %weights: memref<64x3x7x7xf32>, %bias: memref<64xf32>) -> memref<1x64x112x112xf32> {
    // CHECK-LABEL: func.func @conv_test
    // CHECK-SAME: %[[STATE:.*]]: !hip.handle
    // CHECK-SAME: %[[INPUT:.*]]: memref<1x3x224x224xf32>
    // CHECK-SAME: %[[WEIGHTS:.*]]: memref<64x3x7x7xf32>
    // CHECK-SAME: %[[BIAS:.*]]: memref<64xf32>

    // ONNX Conv operation
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [7, 7],
      strides = [2, 2],
      pads = [3, 3, 3, 3],
      dilations = [1, 1],
      group = 1 : si64
    } : (memref<1x3x224x224xf32>, memref<64x3x7x7xf32>, memref<64xf32>) -> memref<1x64x112x112xf32>

    // CHECK: %[[OUTPUT:.*]] = hip.conv(%[[STATE]], %[[INPUT]], %[[WEIGHTS]], %[[BIAS]])
    // CHECK-SAME: {dilations = [1, 1], group = 1 : i64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]}
    // CHECK-SAME: : (memref<1x3x224x224xf32>, memref<64x3x7x7xf32>, memref<64xf32>) -> memref<1x64x112x112xf32>

    return %output : memref<1x64x112x112xf32>
  }
}
