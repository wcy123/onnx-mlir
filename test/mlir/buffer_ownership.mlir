// RUN: mlir-hip-compiler %s --from-onnx-mlir -o %t.dll --verbose
// Test that function arguments are NOT freed (caller-owned buffers)

module {
  func.func @test_ownership(
      %ctx: !hip.context,
      %input: memref<1x3x224x224xf32, 1>) -> i32 {

    // Allocate temp buffer (function owns this)
    %temp = hip.alloc(%ctx) : memref<1x3x224x224xf32, 1>

    memref.copy %input, %temp : memref<1x3x224x224xf32, 1> to memref<1x3x224x224xf32, 1>

    // BufferDeallocation should insert: hip.free(%ctx, %temp)
    // BufferDeallocation should NOT insert: hip.free(%ctx, %input)
    // CHECK: hip.free(%ctx, %temp)
    // CHECK-NOT: hip.free(%ctx, %input)

    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
