// Simple test to verify BufferDeallocation works
module {
  func.func @test(%ctx: !hip.context) -> i32 {
    %buf = hip.alloc(%ctx) : memref<10xf32, 1>
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
