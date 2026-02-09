// Test file: ONNX Conv → HIP Conv with destination-passing style
//
// This test demonstrates the conversion of a simple ONNX Conv operation
// to HIP dialect with destination-passing function signature and in-place
// operation semantics.
//
// DESIGN:
// - HIP dialect functions use destination-passing style:
//   Outputs are passed as arguments, function returns i32 status code
// - HIP dialect operations use in-place semantics:
//   Operations take output buffer as argument, no return value
//
// Function signature transformation:
//   BEFORE: func.func @main(%input: tensor<...>) -> tensor<...>
//   AFTER:  func.func @main(%ctx: !hip.context,
//                          %input: memref<..., 1>,
//                          %output: memref<..., 1>) -> i32
//
// Operation transformation:
//   BEFORE: %result = "onnx.Conv"(...) -> tensor<...>
//   AFTER:  hip.conv(%ctx, ..., %intermediate)  // in-place, no return
//           memref.copy %intermediate, %output  // write to output argument
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
// func.func @main(%arg0: !hip.context,
//                 %arg1: memref<1x3x224x224xf32, 1>,
//                 %arg2: memref<64x3x3x3xf32, 1>,
//                 %arg3: memref<64xf32, 1>,
//                 %arg4: memref<1x64x224x224xf32, 1>) -> i32 {
//
//   // Phase 1: Allocate intermediate buffer inline
//   // TODO Phase 2: hoist to init and load from state
//   %0 = hip.alloc(%arg0) : memref<1x64x224x224xf32, 1>
//
//   // HIP Conv operation (in-place semantics - writes to %0 buffer)
//   hip.conv(%arg0, %arg1, %arg2, %arg3, %0)
//            {kernel_shape = [3, 3], strides = [1, 1],
//             pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
//            : (!hip.context, memref<1x3x224x224xf32, 1>,
//               memref<64x3x3x3xf32, 1>, memref<64xf32, 1>,
//               memref<1x64x224x224xf32, 1>)
//
//   // Destination-passing: copy result to output argument
//   memref.copy %0, %arg4 : memref<1x64x224x224xf32, 1> to memref<1x64x224x224xf32, 1>
//
//   // Return success status
//   %c0_i32 = arith.constant 0 : i32
//   return %c0_i32 : i32
// }
//
// Design summary:
// - Function signature: Destination-passing (outputs as arguments, return i32)
// - HIP operations: In-place semantics (output buffer passed as argument)
// - No function returns memref - all outputs via destination-passing
// - Consistent design from HIP dialect through to final C interface
