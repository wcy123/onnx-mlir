// Test ONNX → HIP lowering for ReLU activation operation
// RUN: hip-opt %s --convert-onnx-to-hip | FileCheck %s

module {
  func.func @test_relu(
      %ctx: !hip.context,
      %input: memref<1x64x224x224xf32, 1>) -> memref<1x64x224x224xf32, 1> {
    // CHECK-LABEL: func.func @test_relu
    // CHECK-SAME: %[[CTX:.*]]: !hip.context
    // CHECK-SAME: %[[INPUT:.*]]: memref<1x64x224x224xf32, 1>

    // ONNX ReLU operation
    %output = "onnx.Relu"(%input) : (memref<1x64x224x224xf32, 1>) -> memref<1x64x224x224xf32, 1>

    // CHECK: %[[OUTPUT:.*]] = hip.relu(%[[CTX]], %[[INPUT]])
    // CHECK-SAME: : (!hip.context, memref<1x64x224x224xf32, 1>) -> memref<1x64x224x224xf32, 1>

    return %output : memref<1x64x224x224xf32, 1>
  }
}
