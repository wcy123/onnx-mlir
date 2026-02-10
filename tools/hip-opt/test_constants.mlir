// Test constant handling in ONNX→HIP conversion
// This demonstrates the constant discovery, global generation, and retrieval

func.func @main(%input: tensor<1x3x224x224xf32>) -> tensor<1x64x224x224xf32> {
  // Constants embedded in the model (weights and bias)
  %weights = "onnx.Constant"() {
    value = dense<1.0> : tensor<64x3x3x3xf32>
  } : () -> tensor<64x3x3x3xf32>

  %bias = "onnx.Constant"() {
    value = dense<0.5> : tensor<64xf32>
  } : () -> tensor<64xf32>

  // Convolution using the constants
  %output = "onnx.Conv"(%input, %weights, %bias) {
    kernel_shape = [3, 3],
    strides = [1, 1],
    pads = [1, 1, 1, 1],
    dilations = [1, 1],
    group = 1 : si64
  } : (tensor<1x3x224x224xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>)
      -> tensor<1x64x224x224xf32>

  return %output : tensor<1x64x224x224xf32>
}
