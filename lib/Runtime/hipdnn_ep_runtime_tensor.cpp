/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hipdnn_ep_runtime.h"

#ifndef BUILD_MOCK_RUNTIME
#include <hip/hip_runtime.h>
#else
// Mock definitions when ROCm is not available
typedef void *hipStream_t;
typedef int hipError_t;
#define hipSuccess 0

// Forward declarations for mock HIP functions (defined in
// hipdnn_ep_runtime_mock.cpp)
hipError_t hipMalloc(void **ptr, size_t size);
hipError_t hipFree(void *ptr);
hipError_t hipMemcpyAsync(void *dst, const void *src, size_t size, int kind,
                          hipStream_t stream);
hipError_t hipStreamSynchronize(hipStream_t stream);
#define hipMemcpyHostToDevice 0
#define hipMemcpyDeviceToHost 1
#endif

#include <cstdio>
#include <cstring>

// Internal runtime state structure (must match hipdnn_ep_runtime_state.cpp)
struct RuntimeState {
  void *stream;
  void *miopen_handle;
  void *hipblas_handle;
  void **gpu_constants;
  size_t num_constants;
};

// Tensor element size (Phase 1: assume float32)
static constexpr size_t kElementSize = 4; // float32 = 4 bytes

// Helper: Calculate total size in bytes for a tensor
// Returns 0 on error (overflow or invalid dimensions)
static size_t calculateTensorSize(const int64_t *shape, size_t rank) {
  if (!shape || rank == 0) {
    return 0;
  }

  // Validate all dimensions are positive
  for (size_t i = 0; i < rank; i++) {
    if (shape[i] <= 0) {
      fprintf(stderr, "Invalid dimension at index %zu: %lld\n", i,
              (long long)shape[i]);
      return 0;
    }
  }

  // Calculate total number of elements with overflow check
  size_t total_elements = 1;
  for (size_t i = 0; i < rank; i++) {
    // Check for overflow before multiplication
    if (total_elements > SIZE_MAX / (size_t)shape[i]) {
      fprintf(stderr, "Tensor size overflow at dimension %zu\n", i);
      return 0;
    }
    total_elements *= (size_t)shape[i];
  }

  // Check for overflow when multiplying by element size
  if (total_elements > SIZE_MAX / kElementSize) {
    fprintf(stderr, "Tensor size overflow when applying element size\n");
    return 0;
  }

  return total_elements * kElementSize;
}

// Prepare input tensor: parse, validate, allocate GPU buffer, H2D transfer
int hipdnn_ep_tensor_prepare_input(RuntimeState *state, span_t *inputs,
                                   size_t index, size_t expected_rank,
                                   TensorBuffer *out_buffer) {
  // Validate arguments
  if (!state) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_input: null state\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!inputs) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_input: null inputs\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!out_buffer) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_input: null out_buffer\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  // Validate index bounds
  if (index >= inputs->count) {
    fprintf(
        stderr,
        "hipdnn_ep_tensor_prepare_input: index %zu out of bounds (count=%zu)\n",
        index, inputs->count);
    return HIPDNN_EP_ERR_INDEX_OUT_OF_BOUNDS;
  }

  // Extract tensor from span
  tensor_t *tensor = &inputs->data[index];

  // Validate tensor pointers
  if (!tensor->data) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_input: tensor[%zu].data is null\n",
            index);
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!tensor->shape) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_input: tensor[%zu].shape is null\n",
            index);
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  // Validate rank
  if (tensor->rank != expected_rank) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_input: rank mismatch (expected %zu, got "
            "%zu)\n",
            expected_rank, tensor->rank);
    return HIPDNN_EP_ERR_RANK_MISMATCH;
  }

  // Calculate buffer size
  size_t size_bytes = calculateTensorSize(tensor->shape, tensor->rank);
  if (size_bytes == 0) {
    return HIPDNN_EP_ERR_INVALID_DIMENSION;
  }

  // Phase 1: Allocate GPU buffer (Phase 2: reuse from pool)
  void *gpu_ptr = nullptr;
  if (hipMalloc(&gpu_ptr, size_bytes) != hipSuccess) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_input: failed to allocate %zu bytes\n",
            size_bytes);
    return HIPDNN_EP_ERR_GPU_ALLOC_FAILED;
  }

  // H2D transfer
  if (hipMemcpyAsync(gpu_ptr, tensor->data, size_bytes, hipMemcpyHostToDevice,
                     static_cast<hipStream_t>(state->stream)) != hipSuccess) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_input: H2D transfer failed\n");
    hipFree(gpu_ptr); // Cleanup on failure
    return HIPDNN_EP_ERR_H2D_TRANSFER_FAILED;
  }

  // Populate output buffer
  out_buffer->gpu_ptr = gpu_ptr;
  out_buffer->host_ptr = tensor->data;
  out_buffer->shape_ptr = tensor->shape;
  out_buffer->rank = tensor->rank;
  out_buffer->size_bytes = size_bytes;
  out_buffer->is_pooled = false; // Phase 1: always allocated, not pooled

  return HIPDNN_EP_SUCCESS;
}

// Prepare output tensor: parse, validate, allocate GPU buffer (no H2D)
int hipdnn_ep_tensor_prepare_output(RuntimeState *state, span_t *outputs,
                                    size_t index, size_t expected_rank,
                                    TensorBuffer *out_buffer) {
  // Validate arguments
  if (!state) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_output: null state\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!outputs) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_output: null outputs\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!out_buffer) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_output: null out_buffer\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  // Validate index bounds
  if (index >= outputs->count) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_output: index %zu out of bounds "
            "(count=%zu)\n",
            index, outputs->count);
    return HIPDNN_EP_ERR_INDEX_OUT_OF_BOUNDS;
  }

  // Extract tensor from span
  tensor_t *tensor = &outputs->data[index];

  // Validate tensor pointers
  if (!tensor->data) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_output: tensor[%zu].data is null\n",
            index);
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!tensor->shape) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_output: tensor[%zu].shape is null\n",
            index);
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  // Validate rank
  if (tensor->rank != expected_rank) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_output: rank mismatch (expected %zu, got "
            "%zu)\n",
            expected_rank, tensor->rank);
    return HIPDNN_EP_ERR_RANK_MISMATCH;
  }

  // Calculate buffer size
  size_t size_bytes = calculateTensorSize(tensor->shape, tensor->rank);
  if (size_bytes == 0) {
    return HIPDNN_EP_ERR_INVALID_DIMENSION;
  }

  // Phase 1: Allocate GPU buffer (Phase 2: reuse from pool)
  void *gpu_ptr = nullptr;
  if (hipMalloc(&gpu_ptr, size_bytes) != hipSuccess) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_output: failed to allocate %zu bytes\n",
            size_bytes);
    return HIPDNN_EP_ERR_GPU_ALLOC_FAILED;
  }

  // No H2D transfer for output tensors

  // Populate output buffer
  out_buffer->gpu_ptr = gpu_ptr;
  out_buffer->host_ptr = tensor->data;
  out_buffer->shape_ptr = tensor->shape;
  out_buffer->rank = tensor->rank;
  out_buffer->size_bytes = size_bytes;
  out_buffer->is_pooled = false; // Phase 1: always allocated, not pooled

  return HIPDNN_EP_SUCCESS;
}

// Finalize output tensor: D2H transfer, sync, release buffer
int hipdnn_ep_tensor_finalize_output(RuntimeState *state,
                                     TensorBuffer *buffer) {
  if (!state) {
    fprintf(stderr, "hipdnn_ep_tensor_finalize_output: null state\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!buffer) {
    fprintf(stderr, "hipdnn_ep_tensor_finalize_output: null buffer\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  int result = HIPDNN_EP_SUCCESS;

  // D2H transfer
  if (hipMemcpyAsync(buffer->host_ptr, buffer->gpu_ptr, buffer->size_bytes,
                     hipMemcpyDeviceToHost,
                     static_cast<hipStream_t>(state->stream)) != hipSuccess) {
    fprintf(stderr, "hipdnn_ep_tensor_finalize_output: D2H transfer failed\n");
    result = HIPDNN_EP_ERR_D2H_TRANSFER_FAILED;
    // Continue to cleanup even on error (best-effort)
  }

  // Stream sync
  if (hipStreamSynchronize(static_cast<hipStream_t>(state->stream)) !=
      hipSuccess) {
    fprintf(stderr, "hipdnn_ep_tensor_finalize_output: stream sync failed\n");
    if (result == HIPDNN_EP_SUCCESS) {
      result = HIPDNN_EP_ERR_STREAM_SYNC_FAILED;
    }
    // Continue to cleanup even on error (best-effort)
  }

  // Phase 1: Free buffer (Phase 2: return to pool or keep if hoisted)
  if (!buffer->is_pooled && buffer->gpu_ptr) {
    hipFree(buffer->gpu_ptr);
    buffer->gpu_ptr = nullptr;
  }

  return result;
}

// Release input tensor buffer (no D2H transfer needed)
void hipdnn_ep_tensor_free_input(RuntimeState *state, TensorBuffer *buffer) {
  if (!buffer) {
    fprintf(stderr, "hipdnn_ep_tensor_free_input: null buffer\n");
    return;
  }

  // Phase 1: Free buffer (Phase 2: return to pool or keep if hoisted)
  if (!buffer->is_pooled && buffer->gpu_ptr) {
    hipFree(buffer->gpu_ptr);
    buffer->gpu_ptr = nullptr;
  }
}
