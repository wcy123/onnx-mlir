// Test that function arguments are NOT freed (caller-owned buffers)
// RUN: hip-opt %s --bufferization-buffer-deallocation | FileCheck %s

module {
  func.func @test_ownership(
      %ctx: !hip.context,
      %input: memref<1x3x224x224xf32, 1>) -> i32 {
    // CHECK-LABEL: func.func @test_ownership

    // Allocate temp buffer (function owns this)
    // CHECK: %[[TEMP:.*]] = hip.alloc(%{{.*}})
    %temp = hip.alloc(%ctx) : memref<1x3x224x224xf32, 1>

    memref.copy %input, %temp : memref<1x3x224x224xf32, 1> to memref<1x3x224x224xf32, 1>

    // BufferDeallocation should insert: hip.free(%ctx, %temp)
    // BufferDeallocation should NOT insert: hip.free(%ctx, %input)
    // CHECK: hip.free(%{{.*}}, %[[TEMP]])
    // CHECK-NOT: hip.free(%{{.*}}, %{{.*input.*}})

    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
