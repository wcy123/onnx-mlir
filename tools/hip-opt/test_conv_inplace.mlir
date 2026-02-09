// Test file: ONNX Conv → HIP Conv with in-place semantics
//
// This test demonstrates the conversion of a simple ONNX Conv operation
// to HIP dialect with in-place semantics.
//
// DESIGN:
// The final compiled function will have signature:
//   int inference_compute(void* state, span_t inputs, span_t outputs)
//
// Where:
// - state: Points to State struct with GPU handles and pre-allocated buffers
// - inputs: span_t containing input tensors (passed by caller)
// - outputs: span_t containing pre-allocated output buffers (passed by caller)
//
// The function writes results to output buffers (in-place semantics),
// does NOT return tensor values.
//
// For this simple test:
// - Input function: ONNX operations with value semantics
// - After ONNX→HIP conversion: HIP operations with in-place semantics
// - After HIP→LLVM conversion: C interface with span_t parameters
//
// In Phase 1 (current): Intermediate buffers allocated inline with hip.alloc
// In Phase 2 (future): All buffers hoisted to inference_init for 4-12x speedup

module {
  // Simple inference function: single Conv operation
  // Input: tensor<1x3x224x224xf32>
  // Output: tensor<1x64x224x224xf32>
  func.func @main(%input: tensor<1x3x224x224xf32>,
                  %weights: tensor<64x3x3x3xf32>,
                  %bias: tensor<64xf32>) -> tensor<1x64x224x224xf32> {
    // ONNX Conv operation (value semantics - returns result)
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 1 : si64
    } : (tensor<1x3x224x224xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x64x224x224xf32>

    return %output : tensor<1x64x224x224xf32>
  }
}

// Expected result after --convert-onnx-to-hip:
//
// func.func @main(%ctx: !hip.context,
//                 %input: memref<1x3x224x224xf32, 1>,
//                 %weights: memref<64x3x3x3xf32, 1>,
//                 %bias: memref<64xf32, 1>) -> memref<1x64x224x224xf32, 1> {
//
//   // Phase 1: Allocate output buffer inline (TODO Phase 2: hoist to init)
//   %output = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
//
//   // HIP Conv operation (in-place semantics - writes to %output buffer)
//   hip.conv(%ctx, %input, %weights, %bias, %output)
//            {kernel_shape = [3, 3], strides = [1, 1],
//             pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
//            : (!hip.context, memref<1x3x224x224xf32, 1>,
//               memref<64x3x3x3xf32, 1>, memref<64xf32, 1>,
//               memref<1x64x224x224xf32, 1>)
//
//   return %output : memref<1x64x224x224xf32, 1>
// }
//
// Design summary:
// - Function signature: Returns memref (value semantics at HIP dialect level)
// - HIP operations: Use in-place semantics (output buffer passed as argument)
// - Later HIP→LLVM lowering transforms function to destination-passing style:
//   func.func @inference_compute(%state: !llvm.ptr, %inputs: !llvm.ptr, %outputs: !llvm.ptr) -> i32
