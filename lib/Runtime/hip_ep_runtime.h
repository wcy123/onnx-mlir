/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_EP_RUNTIME_H
#define HIP_EP_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for runtime state
typedef struct RuntimeState RuntimeState;

// Runtime state management functions
// These provide high-level initialization and cleanup that can be called
// directly from generated interface functions, reducing MLIR pass complexity

// Initialize runtime state (creates stream, MIOpen handle, hipBLAS handle)
// Returns allocated RuntimeState pointer via out_state
// Return codes:
//   0 = success
//   1 = allocation failed
//   2 = stream creation failed
//   3 = MIOpen creation failed
//   4 = set stream failed
//   5 = hipBLAS creation failed
int runtime_state_init(RuntimeState **out_state);

// Cleanup runtime state (destroys handles, frees memory)
// Best-effort cleanup - continues even if individual operations fail
// Returns 0 always (best-effort)
int runtime_state_cleanup(RuntimeState *state);

// Inference data structures for runtime helpers

// Represents a tensor with host data and shape information
typedef struct {
  void *data;     // Host data pointer
  int64_t *shape; // Array of dimension sizes
  size_t rank;    // Number of dimensions
} tensor_t;

// Represents a span of tensors (inputs or outputs)
typedef struct {
  tensor_t *data; // Array of tensors
  size_t count;   // Number of tensors
} span_t;

// Prepared inference data (returned by runtime_prepare_inference)
typedef struct {
  void **input_gpu_buffers;  // Array of GPU buffer pointers for inputs
  void **output_gpu_buffers; // Array of GPU buffer pointers for outputs
  int64_t *input_sizes;      // Array of input buffer sizes in bytes
  int64_t *output_sizes;     // Array of output buffer sizes in bytes
  size_t input_count;
  size_t output_count;
} InferenceData;

// Prepare inference: allocate GPU buffers, H2D copy for inputs
// Returns allocated InferenceData structure with GPU buffers
// Return codes:
//   0 = success
//   1 = invalid parameters
//   2 = memory allocation failed
//   3 = GPU allocation failed
//   4 = H2D copy failed
int runtime_prepare_inference(RuntimeState *state, span_t *inputs,
                              span_t *outputs, InferenceData **out_data);

// Cleanup inference: D2H copy for outputs, free GPU buffers
// Best-effort cleanup - continues even if operations fail
// Returns 0 always
int runtime_cleanup_inference(RuntimeState *state, InferenceData *data,
                              span_t *outputs);

// Constant management functions
// Called by initialize_constants during inference_init
// Uploads constant data from DLL .data section to GPU memory
int hip_upload_constant(RuntimeState *state, int64_t index, const void *data,
                        int64_t size);

// Retrieves GPU pointer for a constant by index
// Returns NULL on error
void *hip_get_constant(RuntimeState *state, int64_t index);

// Releases GPU memory for a constant
// Called by release_constants during inference_cleanup
int hip_release_constant(RuntimeState *state, int64_t index);

// MIOpen convolution forward operation
// Full wrapper with descriptor creation, algorithm finding, workspace
// management Parameters match generated LLVM IR from @main_internal
int miopenConvolutionForward(void *handle,      // MIOpen handle
                             void *stream,      // HIP stream
                             const void *input, // Input tensor GPU pointer
                             const int64_t *input_shape, // [N, C, H, W]
                             const void *weights, // Weights tensor GPU pointer
                             const int64_t *weights_shape, // [K, C, R, S]
                             void *output, // Output tensor GPU pointer
                             const int64_t *output_shape, // [N, K, H', W']
                             int64_t pad_h,               // Padding height
                             int64_t pad_w,               // Padding width
                             int64_t stride_h,            // Stride height
                             int64_t stride_w,            // Stride width
                             int64_t dilation_h,          // Dilation height
                             int64_t dilation_w);         // Dilation width

// hipBLASLt GEMM operation wrapper
// Called by generated IR for matrix multiplication operations
int hipblasLtGemmWrapper(void *handle, // hipBLASLt handle
                         void *stream, // HIP stream
                         int64_t m, int64_t n, int64_t k,
                         const void *alpha, // Scalar alpha
                         const void *A,     // Matrix A GPU pointer
                         const void *B,     // Matrix B GPU pointer
                         const void *beta,  // Scalar beta
                         void *C);          // Matrix C GPU pointer (in/out)

// HIP memory allocation wrapper with error handling
int hip_malloc_wrapper(void **ptr, int64_t size);

// HIP memory free wrapper with error handling
int hip_free_wrapper(void *ptr);

// HIP memory copy host-to-device wrapper
int hip_memcpy_h2d_async(void *dst, const void *src, int64_t size,
                         void *stream);

// HIP memory copy device-to-host wrapper
int hip_memcpy_d2h_async(void *dst, const void *src, int64_t size,
                         void *stream);

// HIP stream synchronization wrapper
int hip_stream_synchronize_wrapper(void *stream);

#ifdef __cplusplus
}
#endif

#endif // HIP_EP_RUNTIME_H
