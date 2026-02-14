// Test HIP → LLVM lowering for convolution operation
// RUN: hip-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @conv_llvm_test(
      %ctx: !hip.context,
      %input: memref<1x3x224x224xf32, 1>,
      %weights: memref<64x3x7x7xf32, 1>,
      %bias: memref<64xf32, 1>,
      %output: memref<1x64x112x112xf32, 1>) {
    // CHECK-LABEL: func.func @conv_llvm_test
    // CHECK-SAME: %[[CTX:.*]]: !llvm.ptr
    // CHECK-SAME: %[[INPUT:.*]]: !llvm.struct
    // CHECK-SAME: %[[WEIGHTS:.*]]: !llvm.struct
    // CHECK-SAME: %[[BIAS:.*]]: !llvm.struct
    // CHECK-SAME: %[[OUTPUT:.*]]: !llvm.struct

    // HIP convolution operation
    hip.conv(%ctx, %input, %weights, %bias, %output)
      {kernel_shape = [7, 7], strides = [2, 2],
       pads = [3, 3, 3, 3], dilations = [1, 1], group = 1}
      : (!hip.context, memref<1x3x224x224xf32, 1>,
         memref<64x3x7x7xf32, 1>, memref<64xf32, 1>,
         memref<1x64x112x112xf32, 1>)

    // CHECK: llvm.call @miopenConvolutionForward
    // CHECK-SAME: (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i64)

    return
  }
}
