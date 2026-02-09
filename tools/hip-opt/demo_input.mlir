module {
  func.func @inference_compute(%state: !hip.handle) -> memref<1x64x112x112xf32> {
    // Input tensors (normally from model weights)
    %input = memref.alloc() : memref<1x3x224x224xf32>
    %weights = memref.alloc() : memref<64x3x7x7xf32>
    %bias = memref.alloc() : memref<64xf32>

    // ONNX Conv operation
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [7, 7],
      strides = [2, 2],
      pads = [3, 3, 3, 3],
      dilations = [1, 1],
      group = 1 : si64
    } : (memref<1x3x224x224xf32>, memref<64x3x7x7xf32>, memref<64xf32>)
      -> memref<1x64x112x112xf32>

    return %output : memref<1x64x112x112xf32>
  }
}
