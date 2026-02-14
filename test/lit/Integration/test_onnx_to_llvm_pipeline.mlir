// Test full ONNX → HIP → LLVM compilation pipeline
// RUN: hip-opt %s --convert-onnx-to-hip --bufferization-buffer-deallocation --convert-hip-to-llvm | FileCheck %s

module {
  func.func @full_pipeline(
      %ctx: !hip.context,
      %input: memref<1x64x56x56xf32>) -> memref<1x64x56x56xf32> {
    // CHECK-LABEL: llvm.func @full_pipeline

    // Stage 1: ONNX ReLU → HIP ReLU
    %activated = "onnx.Relu"(%input) : (memref<1x64x56x56xf32>) -> memref<1x64x56x56xf32>

    // Stage 2: BufferDeallocation (no allocations in this simple case)

    // Stage 3: HIP → LLVM (should lower to runtime calls)
    // CHECK: llvm.call @miopenActivationForward
    // CHECK-SAME: (!llvm.ptr, {{.*}}) -> ()

    return %activated : memref<1x64x56x56xf32>
  }

  func.func @conv_relu_pipeline(
      %ctx: !hip.context,
      %input: memref<1x3x224x224xf32>,
      %weights: memref<64x3x7x7xf32>,
      %bias: memref<64xf32>) -> memref<1x64x112x112xf32> {
    // CHECK-LABEL: llvm.func @conv_relu_pipeline

    // Conv operation
    %conv_out = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [7, 7],
      strides = [2, 2],
      pads = [3, 3, 3, 3],
      dilations = [1, 1],
      group = 1 : si64
    } : (memref<1x3x224x224xf32>, memref<64x3x7x7xf32>, memref<64xf32>) -> memref<1x64x112x112xf32>

    // ReLU activation
    %output = "onnx.Relu"(%conv_out) : (memref<1x64x112x112xf32>) -> memref<1x64x112x112xf32>

    // Should see both Conv and ReLU lowered to LLVM calls
    // CHECK: llvm.call @miopenConvolutionForward
    // CHECK: llvm.call @miopenActivationForward

    return %output : memref<1x64x112x112xf32>
  }
}
