// RUN: mlir-hip-compiler %s --from-onnx-mlir -o %t.dll --verbose
// Test that BufferDeallocation automatically inserts hip.free operations

module {
  func.func @simple_alloc(%ctx: !hip.context) -> i32 {
    // Allocate two buffers
    %buf1 = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
    %buf2 = hip.alloc(%ctx) : memref<1x64x112x112xf32, 1>

    // Use buffers (simulate some computation)
    // After BufferDeallocation, hip.free should be inserted here:
    // CHECK: hip.free(%ctx, %buf1)
    // CHECK: hip.free(%ctx, %buf2)

    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
