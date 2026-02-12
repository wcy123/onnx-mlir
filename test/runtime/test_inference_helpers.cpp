/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../../lib/Runtime/hipdnn_ep_runtime.h"
#include "../../lib/Runtime/hip_ep_runtime_mock.h"
#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>

int main() {
  std::cout << "=== Inference Helpers Test ===\n\n";

#ifndef BUILD_MOCK_RUNTIME
  std::cerr << "ERROR: This test must be built with BUILD_MOCK_RUNTIME=1\n";
  return 1;
#endif

  // Test 1: Basic inference prepare and cleanup
  std::cout << "--- Test 1: Single Input/Output Tensor ---\n";

  // Initialize runtime state
  RuntimeState *state = nullptr;
  int result = hipdnn_ep_state_init(&state);
  assert(result == 0);
  std::cout << "✓ State initialized\n";

  // Create input tensor (1x3x4x4 = 48 elements)
  std::vector<float> input_data(48, 1.0f);
  int64_t input_shape[] = {1, 3, 4, 4};
  tensor_t input_tensor;
  input_tensor.data = input_data.data();
  input_tensor.shape = input_shape;
  input_tensor.rank = 4;

  tensor_t inputs_array[] = {input_tensor};
  span_t inputs;
  inputs.data = inputs_array;
  inputs.count = 1;

  // Create output tensor (1x10 = 10 elements)
  std::vector<float> output_data(10, 0.0f);
  int64_t output_shape[] = {1, 10};
  tensor_t output_tensor;
  output_tensor.data = output_data.data();
  output_tensor.shape = output_shape;
  output_tensor.rank = 2;

  tensor_t outputs_array[] = {output_tensor};
  span_t outputs;
  outputs.data = outputs_array;
  outputs.count = 1;

  // Prepare inference
  InferenceData *inf_data = nullptr;
  result = runtime_prepare_inference(state, &inputs, &outputs, &inf_data);
  assert(result == 0);
  assert(inf_data != nullptr);
  std::cout << "✓ Inference prepared\n";

  // Verify data structure
  assert(inf_data->input_count == 1);
  assert(inf_data->output_count == 1);
  assert(inf_data->input_gpu_buffers[0] != nullptr);
  assert(inf_data->output_gpu_buffers[0] != nullptr);
  assert(inf_data->input_sizes[0] == 48 * 4);  // 48 floats
  assert(inf_data->output_sizes[0] == 10 * 4); // 10 floats
  std::cout << "✓ Data structure verified\n";

  // Cleanup inference
  result = runtime_cleanup_inference(state, inf_data, &outputs);
  assert(result == 0);
  std::cout << "✓ Inference cleaned up\n";

  // Cleanup state
  result = hipdnn_ep_state_cleanup(state);
  assert(result == 0);
  std::cout << "✓ State cleaned up\n\n";

  // Test 2: Multiple tensors
  std::cout << "--- Test 2: Multiple Input/Output Tensors ---\n";

  result = hipdnn_ep_state_init(&state);
  assert(result == 0);

  // Create multiple input tensors
  std::vector<float> input1_data(12, 1.0f); // 1x3x2x2
  int64_t input1_shape[] = {1, 3, 2, 2};
  tensor_t input1_tensor;
  input1_tensor.data = input1_data.data();
  input1_tensor.shape = input1_shape;
  input1_tensor.rank = 4;

  std::vector<float> input2_data(8, 2.0f); // 1x2x2x2
  int64_t input2_shape[] = {1, 2, 2, 2};
  tensor_t input2_tensor;
  input2_tensor.data = input2_data.data();
  input2_tensor.shape = input2_shape;
  input2_tensor.rank = 4;

  tensor_t multi_inputs[] = {input1_tensor, input2_tensor};
  span_t multi_inputs_span;
  multi_inputs_span.data = multi_inputs;
  multi_inputs_span.count = 2;

  // Create multiple output tensors
  std::vector<float> output1_data(5, 0.0f);
  int64_t output1_shape[] = {1, 5};
  tensor_t output1_tensor;
  output1_tensor.data = output1_data.data();
  output1_tensor.shape = output1_shape;
  output1_tensor.rank = 2;

  std::vector<float> output2_data(3, 0.0f);
  int64_t output2_shape[] = {1, 3};
  tensor_t output2_tensor;
  output2_tensor.data = output2_data.data();
  output2_tensor.shape = output2_shape;
  output2_tensor.rank = 2;

  tensor_t multi_outputs[] = {output1_tensor, output2_tensor};
  span_t multi_outputs_span;
  multi_outputs_span.data = multi_outputs;
  multi_outputs_span.count = 2;

  // Prepare
  result = runtime_prepare_inference(state, &multi_inputs_span,
                                     &multi_outputs_span, &inf_data);
  assert(result == 0);
  std::cout << "✓ Multiple tensors prepared\n";

  // Verify
  assert(inf_data->input_count == 2);
  assert(inf_data->output_count == 2);
  assert(inf_data->input_sizes[0] == 12 * 4);
  assert(inf_data->input_sizes[1] == 8 * 4);
  assert(inf_data->output_sizes[0] == 5 * 4);
  assert(inf_data->output_sizes[1] == 3 * 4);
  std::cout << "✓ Multiple tensors verified\n";

  // Cleanup
  result = runtime_cleanup_inference(state, inf_data, &multi_outputs_span);
  assert(result == 0);
  result = hipdnn_ep_state_cleanup(state);
  assert(result == 0);
  std::cout << "✓ Multiple tensors cleaned up\n\n";

  // Test 3: Variable rank tensors
  std::cout << "--- Test 3: Variable Rank Tensors ---\n";

  result = hipdnn_ep_state_init(&state);
  assert(result == 0);

  // Rank-2 tensor
  std::vector<float> rank2_data(6, 1.0f); // 2x3
  int64_t rank2_shape[] = {2, 3};
  tensor_t rank2_tensor;
  rank2_tensor.data = rank2_data.data();
  rank2_tensor.shape = rank2_shape;
  rank2_tensor.rank = 2;

  // Rank-5 tensor
  std::vector<float> rank5_data(32, 2.0f); // 1x2x2x2x2
  int64_t rank5_shape[] = {1, 2, 2, 2, 2};
  tensor_t rank5_tensor;
  rank5_tensor.data = rank5_data.data();
  rank5_tensor.shape = rank5_shape;
  rank5_tensor.rank = 5;

  tensor_t var_rank_inputs[] = {rank2_tensor, rank5_tensor};
  span_t var_rank_inputs_span;
  var_rank_inputs_span.data = var_rank_inputs;
  var_rank_inputs_span.count = 2;

  // Single output
  std::vector<float> var_output_data(4, 0.0f);
  int64_t var_output_shape[] = {1, 4};
  tensor_t var_output_tensor;
  var_output_tensor.data = var_output_data.data();
  var_output_tensor.shape = var_output_shape;
  var_output_tensor.rank = 2;

  tensor_t var_outputs[] = {var_output_tensor};
  span_t var_outputs_span;
  var_outputs_span.data = var_outputs;
  var_outputs_span.count = 1;

  // Prepare
  result = runtime_prepare_inference(state, &var_rank_inputs_span,
                                     &var_outputs_span, &inf_data);
  assert(result == 0);
  std::cout << "✓ Variable rank tensors prepared\n";

  // Verify sizes
  assert(inf_data->input_sizes[0] == 6 * 4);  // rank-2: 2*3
  assert(inf_data->input_sizes[1] == 32 * 4); // rank-5: 1*2*2*2*2
  assert(inf_data->output_sizes[0] == 4 * 4); // rank-2: 1*4
  std::cout << "✓ Variable rank sizes verified\n";

  // Cleanup
  result = runtime_cleanup_inference(state, inf_data, &var_outputs_span);
  assert(result == 0);
  result = hipdnn_ep_state_cleanup(state);
  assert(result == 0);
  std::cout << "✓ Variable rank cleaned up\n\n";

  // Test 4: Null parameter handling
  std::cout << "--- Test 4: Null Parameter Handling ---\n";

  result = runtime_prepare_inference(nullptr, &inputs, &outputs, &inf_data);
  assert(result == 1); // Invalid parameters
  std::cout << "✓ Null state handled\n";

  result = runtime_cleanup_inference(nullptr, nullptr, nullptr);
  assert(result == 0); // Best-effort cleanup
  std::cout << "✓ Null cleanup handled\n\n";

  std::cout << "=== All Inference Helper Tests PASSED ===\n";
  std::cout << "\nExpected output above should show:\n";
  std::cout << "  - hipMalloc for input/output GPU buffers\n";
  std::cout << "  - hipMemcpyAsync H2D for inputs\n";
  std::cout << "  - hipMemcpyAsync D2H for outputs\n";
  std::cout << "  - hipStreamSynchronize after D2H\n";
  std::cout << "  - hipFree for all allocated buffers\n";

  return 0;
}
