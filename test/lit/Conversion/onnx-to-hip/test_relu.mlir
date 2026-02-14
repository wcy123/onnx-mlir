// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX ReLU (Rectified Linear Unit) activation is correctly lowered
// to hip.relu operation.
//
// This test validates:
// - Basic activation function lowering (onnx.Relu → hip.relu)
// - Element-wise operation handling
// - In-place operation semantics (input → output)
// - Proper !hip.context threading through operations
//
// Input: ONNX ReLU on 4D tensor (batch x channels x height x width)
// Expected: hip.relu operation preserving tensor shape and context
// ============================================================================

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
