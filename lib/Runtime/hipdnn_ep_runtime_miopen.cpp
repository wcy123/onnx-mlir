/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hipdnn_ep_runtime.h"
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

#include <cstdio>

// Error checking macros
#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t error = (cmd);                                                  \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(error));                                       \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define MIOPEN_CHECK(cmd)                                                      \
  do {                                                                         \
    miopenStatus_t status = (cmd);                                             \
    if (status != miopenStatusSuccess) {                                       \
      fprintf(stderr, "MIOpen error at %s:%d: %d\n", __FILE__, __LINE__,       \
              status);                                                         \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// MIOpen convolution forward implementation

int wrap_miopenConvolutionForward(void *handle, void *stream, const void *input,
                             const int64_t *input_shape, const void *weights,
                             const int64_t *weights_shape, void *output,
                             const int64_t *output_shape, int64_t pad_h,
                             int64_t pad_w, int64_t stride_h, int64_t stride_w,
                             int64_t dilation_h, int64_t dilation_w) {
  if (!handle || !stream || !input || !weights || !output) {
    fprintf(stderr, "Invalid arguments to wrap_miopenConvolutionForward\n");
    return -1;
  }

  miopenHandle_t miopen_handle = static_cast<miopenHandle_t>(handle);
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  // Create tensor descriptors
  miopenTensorDescriptor_t input_desc, weights_desc, output_desc;
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&input_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&weights_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&output_desc));

  // Set tensor descriptors (assuming float32 data type)
  MIOPEN_CHECK(miopenSet4dTensorDescriptor(input_desc, miopenFloat,
                                           input_shape[0], input_shape[1],
                                           input_shape[2], input_shape[3]));

  MIOPEN_CHECK(miopenSet4dTensorDescriptor(weights_desc, miopenFloat,
                                           weights_shape[0], weights_shape[1],
                                           weights_shape[2], weights_shape[3]));

  MIOPEN_CHECK(miopenSet4dTensorDescriptor(output_desc, miopenFloat,
                                           output_shape[0], output_shape[1],
                                           output_shape[2], output_shape[3]));

  // Create convolution descriptor
  miopenConvolutionDescriptor_t conv_desc;
  MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&conv_desc));
  MIOPEN_CHECK(miopenInitConvolutionDescriptor(conv_desc, miopenConvolution,
                                               pad_h, pad_w, stride_h, stride_w,
                                               dilation_h, dilation_w));

  // Find best algorithm
  miopenConvFwdAlgorithm_t algo;
  MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
      miopen_handle, input_desc, input, weights_desc, weights, conv_desc,
      output_desc, output,
      1, // requestAlgoCount
      &algo,
      nullptr, // returnedAlgoCount
      nullptr, // workspace (nullptr to query size)
      0,       // workspaceSize
      false)); // exhaustiveSearch

  // Get workspace size
  size_t workspace_size = 0;
  MIOPEN_CHECK(miopenConvolutionForwardGetWorkSpaceSize(
      miopen_handle, weights_desc, input_desc, conv_desc, output_desc,
      &workspace_size));

  // Allocate workspace
  void *workspace = nullptr;
  if (workspace_size > 0) {
    HIP_CHECK(hipMalloc(&workspace, workspace_size));
  }

  // Perform convolution
  float alpha = 1.0f;
  float beta = 0.0f;
  MIOPEN_CHECK(miopenConvolutionForward(
      miopen_handle, &alpha, input_desc, input, weights_desc, weights,
      conv_desc, algo, &beta, output_desc, output, workspace, workspace_size));

  // Cleanup
  if (workspace) {
    hipFree(workspace);
  }
  miopenDestroyTensorDescriptor(input_desc);
  miopenDestroyTensorDescriptor(weights_desc);
  miopenDestroyTensorDescriptor(output_desc);
  miopenDestroyConvolutionDescriptor(conv_desc);

  return 0;
}
