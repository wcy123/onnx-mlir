// Test C-ABI wrapper generation for ONNX functions
// RUN: hip-opt %s --generate-interface | FileCheck %s

module {
  // Original MLIR function
  func.func @main_graph(
      %ctx: !hip.context,
      %input: memref<1x3x224x224xf32, 1>,
      %weights: memref<64x3x7x7xf32, 1>,
      %bias: memref<64xf32, 1>) -> memref<1x64x112x112xf32, 1> {

    %output = hip.alloc(%ctx) : memref<1x64x112x112xf32, 1>

    hip.conv(%ctx, %input, %weights, %bias, %output)
      {kernel_shape = [7, 7], strides = [2, 2],
       pads = [3, 3, 3, 3], dilations = [1, 1], group = 1}
      : (!hip.context, memref<1x3x224x224xf32, 1>,
         memref<64x3x7x7xf32, 1>, memref<64xf32, 1>,
         memref<1x64x112x112xf32, 1>)

    return %output : memref<1x64x112x112xf32, 1>
  }

  // CHECK-LABEL: llvm.func @main_graph_cabi
  // CHECK-SAME: (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr

  // C-ABI wrapper should:
  // 1. Convert raw pointers to memref descriptors
  // CHECK: llvm.mlir.undef : !llvm.struct

  // 2. Call the original function
  // CHECK: llvm.call @main_graph

  // 3. Extract result pointer and return
  // CHECK: llvm.extractvalue
  // CHECK: llvm.return
}
