// Test constant handling in ONNX→HIP conversion
// This demonstrates the constant discovery, global generation, and retrieval
// Two sequential convolutions with different constant weights

func.func @main(%input: tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32> {
  // First Conv: Constants for layer 1
  %weights1 = "onnx.Constant"() {
    value = dense<1.0> : tensor<64x3x3x3xf32>
  } : () -> tensor<64x3x3x3xf32>

  %bias1 = "onnx.Constant"() {
    value = dense<0.5> : tensor<64xf32>
  } : () -> tensor<64xf32>

  // First convolution (same size output)
  %conv1 = "onnx.Conv"(%input, %weights1, %bias1) {
    kernel_shape = [3, 3],
    strides = [1, 1],
    pads = [1, 1, 1, 1],
    dilations = [1, 1],
    group = 1 : si64
  } : (tensor<1x3x224x224xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>)
      -> tensor<1x64x224x224xf32>

  // Second Conv: Constants for layer 2
  %weights2 = "onnx.Constant"() {
    value = dense<2.0> : tensor<64x64x3x3xf32>
  } : () -> tensor<64x64x3x3xf32>

  %bias2 = "onnx.Constant"() {
    value = dense<0.1> : tensor<64xf32>
  } : () -> tensor<64xf32>

  // Second convolution (stride=2, halves spatial dimensions)
  %conv2 = "onnx.Conv"(%conv1, %weights2, %bias2) {
    kernel_shape = [3, 3],
    strides = [2, 2],
    pads = [1, 1, 1, 1],
    dilations = [1, 1],
    group = 1 : si64
  } : (tensor<1x64x224x224xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>)
      -> tensor<1x64x112x112xf32>

  return %conv2 : tensor<1x64x112x112xf32>
}
