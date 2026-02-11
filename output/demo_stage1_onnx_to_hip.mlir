module attributes {hipdnn.input_count = 1 : i64, hipdnn.input_ranks = array<i64: 4>, hipdnn.output_count = 1 : i64, hipdnn.output_ranks = array<i64: 4>} {
  llvm.mlir.global internal constant @constant_0(dense<1.000000e+00> : tensor<64x3x3x3xf32>) {addr_space = 0 : i32} : !llvm.array<1728 x f32>
  llvm.mlir.global internal constant @constant_2(dense<2.000000e+00> : tensor<64x64x3x3xf32>) {addr_space = 0 : i32} : !llvm.array<36864 x f32>
  llvm.mlir.global internal constant @constant_1(dense<5.000000e-01> : tensor<64xf32>) {addr_space = 0 : i32} : !llvm.array<64 x f32>
  llvm.mlir.global internal constant @constant_3(dense<1.000000e-01> : tensor<64xf32>) {addr_space = 0 : i32} : !llvm.array<64 x f32>
  func.func @main(%arg0: !hip.context, %arg1: memref<1x3x224x224xf32, 1>, %arg2: memref<1x64x112x112xf32, 1>) -> i32 {
    %c0_i64 = arith.constant 0 : i64
    %0 = hip.get_constant(%arg0, %c0_i64) : memref<64x3x3x3xf32, 1>
    %c1_i64 = arith.constant 1 : i64
    %1 = hip.get_constant(%arg0, %c1_i64) : memref<64xf32, 1>
    %2 = hip.alloc(%arg0) : memref<1x64x224x224xf32, 1>
    hip.conv(%arg0, %arg1, %0, %1, %2) {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (!hip.context, memref<1x3x224x224xf32, 1>, memref<64x3x3x3xf32, 1>, memref<64xf32, 1>, memref<1x64x224x224xf32, 1>)
    %c2_i64 = arith.constant 2 : i64
    %3 = hip.get_constant(%arg0, %c2_i64) : memref<64x64x3x3xf32, 1>
    %c3_i64 = arith.constant 3 : i64
    %4 = hip.get_constant(%arg0, %c3_i64) : memref<64xf32, 1>
    %5 = hip.alloc(%arg0) : memref<1x64x112x112xf32, 1>
    hip.conv(%arg0, %2, %3, %4, %5) {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]} : (!hip.context, memref<1x64x224x224xf32, 1>, memref<64x64x3x3xf32, 1>, memref<64xf32, 1>, memref<1x64x112x112xf32, 1>)
    memref.copy %5, %arg2 : memref<1x64x112x112xf32, 1> to memref<1x64x112x112xf32, 1>
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
  llvm.func @get_constant_count() -> i64 {
    %0 = llvm.mlir.constant(4 : i64) : i64
    llvm.return %0 : i64
  }
  func.func @initialize_constants(%arg0: !hip.context) -> i32 {
    %0 = llvm.mlir.addressof @constant_0 : !llvm.ptr
    %c0_i64 = arith.constant 0 : i64
    %c6912_i64 = arith.constant 6912 : i64
    hip.upload_constant(%arg0, %c0_i64, %0, %c6912_i64) : (!llvm.ptr)
    %1 = llvm.mlir.addressof @constant_2 : !llvm.ptr
    %c2_i64 = arith.constant 2 : i64
    %c147456_i64 = arith.constant 147456 : i64
    hip.upload_constant(%arg0, %c2_i64, %1, %c147456_i64) : (!llvm.ptr)
    %2 = llvm.mlir.addressof @constant_1 : !llvm.ptr
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    hip.upload_constant(%arg0, %c1_i64, %2, %c256_i64) : (!llvm.ptr)
    %3 = llvm.mlir.addressof @constant_3 : !llvm.ptr
    %c3_i64 = arith.constant 3 : i64
    %c256_i64_0 = arith.constant 256 : i64
    hip.upload_constant(%arg0, %c3_i64, %3, %c256_i64_0) : (!llvm.ptr)
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
  func.func @release_constants(%arg0: !hip.context) -> i32 {
    %c0_i64 = arith.constant 0 : i64
    hip.release_constant(%arg0, %c0_i64)
    %c2_i64 = arith.constant 2 : i64
    hip.release_constant(%arg0, %c2_i64)
    %c1_i64 = arith.constant 1 : i64
    hip.release_constant(%arg0, %c1_i64)
    %c3_i64 = arith.constant 3 : i64
    hip.release_constant(%arg0, %c3_i64)
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}

