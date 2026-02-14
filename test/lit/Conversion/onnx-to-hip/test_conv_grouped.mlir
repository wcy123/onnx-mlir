// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Conv operations with grouped convolutions are correctly lowered
// to hip.conv operations.
//
// This test validates:
// - Grouped convolution lowering (group > 1)
// - Depthwise convolution (group = num_channels)
// - Proper channel partitioning with groups
// - Weight tensor size validation for grouped convolutions
//
// Test cases:
// 1. Grouped conv (group=2): 64→128 channels, each group processes 32→64
// 2. Depthwise conv (group=64): Each of 64 channels processed independently
//
// Note: Grouped convolutions are common in MobileNet and ResNeXt architectures
// Expected: hip.conv operations with correct group attribute
// ============================================================================

// RUN: hip-opt %s --convert-onnx-to-hip | FileCheck %s

module {
  func.func @conv_grouped_test(
      %ctx: !hip.context,
      %input: memref<1x64x56x56xf32>,
      %weights: memref<128x32x3x3xf32>,
      %bias: memref<128xf32>) -> memref<1x128x56x56xf32> {
    // CHECK-LABEL: func.func @conv_grouped_test
    // CHECK-SAME: %[[CTX:.*]]: !hip.context
    // CHECK-SAME: %[[INPUT:.*]]: memref<1x64x56x56xf32>
    // CHECK-SAME: %[[WEIGHTS:.*]]: memref<128x32x3x3xf32>
    // CHECK-SAME: %[[BIAS:.*]]: memref<128xf32>

    // Grouped convolution with group=2
    // Input channels: 64, Output channels: 128, Groups: 2
    // Each group processes 32 input channels → 64 output channels
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 2 : si64
    } : (memref<1x64x56x56xf32>, memref<128x32x3x3xf32>, memref<128xf32>) -> memref<1x128x56x56xf32>

    // CHECK: %[[OUTPUT:.*]] = hip.conv(%[[CTX]], %[[INPUT]], %[[WEIGHTS]], %[[BIAS]])
    // CHECK-SAME: group = 2 : i64
    // CHECK-SAME: : (memref<1x64x56x56xf32>, memref<128x32x3x3xf32>, memref<128xf32>) -> memref<1x128x56x56xf32>

    return %output : memref<1x128x56x56xf32>
  }

  func.func @depthwise_conv(
      %ctx: !hip.context,
      %input: memref<1x64x56x56xf32>,
      %weights: memref<64x1x3x3xf32>,
      %bias: memref<64xf32>) -> memref<1x64x56x56xf32> {
    // CHECK-LABEL: func.func @depthwise_conv

    // Depthwise convolution (group = num_channels)
    // Each of the 64 channels is processed independently with its own 3x3 filter
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 64 : si64
    } : (memref<1x64x56x56xf32>, memref<64x1x3x3xf32>, memref<64xf32>) -> memref<1x64x56x56xf32>

    // CHECK: hip.conv(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
    // CHECK-SAME: group = 64 : i64

    return %output : memref<1x64x56x56xf32>
  }
}
