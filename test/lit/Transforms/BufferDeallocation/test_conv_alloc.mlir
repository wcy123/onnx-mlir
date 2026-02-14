// Test BufferDeallocation with convolution operation
// RUN: hip-opt %s --bufferization-buffer-deallocation | FileCheck %s

module {
  func.func @conv_test(
      %ctx: !hip.context,
      %input: memref<1x3x224x224xf32, 1>,
      %weights: memref<64x3x3x3xf32, 1>,
      %output: memref<1x64x224x224xf32, 1>) -> i32 {
    // CHECK-LABEL: func.func @conv_test

    // Allocate temporary buffer
    // CHECK: %[[TEMP:.*]] = hip.alloc(%{{.*}})
    %temp = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>

    // Perform convolution
    hip.conv(%ctx, %input, %weights, %none, %temp)
      {kernel_shape = [3, 3], strides = [1, 1],
       pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
      : (!hip.context, memref<1x3x224x224xf32, 1>,
         memref<64x3x3x3xf32, 1>, none, memref<1x64x224x224xf32, 1>)

    // Copy temp to output
    memref.copy %temp, %output : memref<1x64x224x224xf32, 1> to memref<1x64x224x224xf32, 1>

    // BufferDeallocation should insert: hip.free(%ctx, %temp)
    // CHECK: hip.free(%{{.*}}, %[[TEMP]])
    // CHECK-NEXT: arith.constant

    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
