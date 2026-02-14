// RUN: mlir-hip-compiler %s --from-onnx-mlir -o %t.dll --verbose
// Test ReLU activation operation

module {
  func.func @test_relu(
      %ctx: !hip.context,
      %input: memref<1x64x224x224xf32, 1>,
      %output: memref<1x64x224x224xf32, 1>) -> i32 {

    // Apply ReLU activation
    hip.relu(%ctx, %input, %output)
      : (!hip.context, memref<1x64x224x224xf32, 1>, memref<1x64x224x224xf32, 1>)

    // CHECK: hip.relu
    // After BufferDeallocation, no hip.free should be inserted for function arguments

    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
