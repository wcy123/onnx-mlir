// Test ONNX → HIP lowering for constant tensors
// RUN: hip-opt %s --convert-onnx-to-hip | FileCheck %s

module {
  func.func @constant_test(%ctx: !hip.context) -> memref<3xf32> {
    // CHECK-LABEL: func.func @constant_test
    // CHECK-SAME: %[[CTX:.*]]: !hip.context

    // ONNX Constant operation with dense elements
    %const = "onnx.Constant"() {
      value = dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    } : () -> memref<3xf32>

    // CHECK: %[[CONST:.*]] = hip.get_constant(%[[CTX]])
    // CHECK-SAME: {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>}
    // CHECK-SAME: : (!hip.context) -> memref<3xf32>

    return %const : memref<3xf32>
  }

  func.func @constant_matrix(%ctx: !hip.context) -> memref<2x2xf32> {
    // CHECK-LABEL: func.func @constant_matrix
    // CHECK-SAME: %[[CTX:.*]]: !hip.context

    // 2D constant tensor
    %matrix = "onnx.Constant"() {
      value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    } : () -> memref<2x2xf32>

    // CHECK: %[[MATRIX:.*]] = hip.get_constant(%[[CTX]])
    // CHECK-SAME: {value = dense<{{\[}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>}
    // CHECK-SAME: : (!hip.context) -> memref<2x2xf32>

    return %matrix : memref<2x2xf32>
  }
}
