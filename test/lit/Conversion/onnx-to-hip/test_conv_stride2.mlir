// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Conv operations with various stride configurations are correctly
// lowered to hip.conv operations.
//
// This test validates:
// - Strided convolution lowering (stride = 2 for downsampling)
// - Asymmetric stride handling (different horizontal/vertical strides)
// - Proper output size calculation with strides
// - Attribute preservation for non-unit strides
//
// Test cases:
// 1. Symmetric stride=2: Common in ResNet bottleneck layers
// 2. Asymmetric stride=[2,3]: Less common but valid configuration
//
// Expected: hip.conv operations with correct stride attributes
// ============================================================================

// RUN: hip-opt %s --convert-onnx-to-hip | FileCheck %s

module {
  func.func @conv_stride2_test(
      %ctx: !hip.context,
      %input: memref<1x64x56x56xf32>,
      %weights: memref<128x64x3x3xf32>,
      %bias: memref<128xf32>) -> memref<1x128x28x28xf32> {
    // CHECK-LABEL: func.func @conv_stride2_test
    // CHECK-SAME: %[[CTX:.*]]: !hip.context
    // CHECK-SAME: %[[INPUT:.*]]: memref<1x64x56x56xf32>
    // CHECK-SAME: %[[WEIGHTS:.*]]: memref<128x64x3x3xf32>
    // CHECK-SAME: %[[BIAS:.*]]: memref<128xf32>

    // ONNX Conv with stride=2 (downsampling)
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [2, 2],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 1 : si64
    } : (memref<1x64x56x56xf32>, memref<128x64x3x3xf32>, memref<128xf32>) -> memref<1x128x28x28xf32>

    // CHECK: %[[OUTPUT:.*]] = hip.conv(%[[CTX]], %[[INPUT]], %[[WEIGHTS]], %[[BIAS]])
    // CHECK-SAME: {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]}
    // CHECK-SAME: : (memref<1x64x56x56xf32>, memref<128x64x3x3xf32>, memref<128xf32>) -> memref<1x128x28x28xf32>

    return %output : memref<1x128x28x28xf32>
  }

  func.func @conv_asymmetric_stride(
      %ctx: !hip.context,
      %input: memref<1x3x224x224xf32>,
      %weights: memref<64x3x7x3xf32>,
      %bias: memref<64xf32>) -> memref<1x64x112x74xf32> {
    // CHECK-LABEL: func.func @conv_asymmetric_stride

    // Conv with asymmetric stride (2x3)
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [7, 3],
      strides = [2, 3],
      pads = [3, 1, 3, 1],
      dilations = [1, 1],
      group = 1 : si64
    } : (memref<1x3x224x224xf32>, memref<64x3x7x3xf32>, memref<64xf32>) -> memref<1x64x112x74xf32>

    // CHECK: hip.conv(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
    // CHECK-SAME: strides = [2, 3]

    return %output : memref<1x64x112x74xf32>
  }
}
