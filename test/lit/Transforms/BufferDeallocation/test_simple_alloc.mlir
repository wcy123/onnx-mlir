// Test that BufferDeallocation automatically inserts hip.free operations
// RUN: hip-opt %s --bufferization-buffer-deallocation | FileCheck %s

module {
  func.func @simple_alloc(%ctx: !hip.context) -> i32 {
    // CHECK-LABEL: func.func @simple_alloc
    // CHECK-SAME: %[[CTX:.*]]: !hip.context

    // Allocate two buffers
    // CHECK: %[[BUF1:.*]] = hip.alloc(%[[CTX]])
    %buf1 = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
    // CHECK: %[[BUF2:.*]] = hip.alloc(%[[CTX]])
    %buf2 = hip.alloc(%ctx) : memref<1x64x112x112xf32, 1>

    // After BufferDeallocation, hip.free should be inserted here:
    // CHECK: hip.free(%[[CTX]], %[[BUF2]])
    // CHECK-NEXT: hip.free(%[[CTX]], %[[BUF1]])

    %c0 = arith.constant 0 : i32
    // CHECK: arith.constant 0
    return %c0 : i32
  }
}
