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

//==============================================================================
// RuntimeState: Opaque Execution State
//==============================================================================
//
// RuntimeState encapsulates GPU execution resources (stream, library handles,
// model constants). Generated code treats it as opaque void*, runtime library
// owns the internal structure.
//
// Design rationale: Opaque pointer pattern allows runtime to evolve internal
// layout without breaking generated code.
//
// Lifecycle: init -> use -> cleanup (must call in this order)
// Thread safety: Not thread-safe (one inference per state at a time)
//==============================================================================

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

// Get GPU stream from state (for passing to HIP operations)
// Returns: hipStream_t cast to void* (NULL on error)
// Ownership: Caller does NOT own stream (destroyed in cleanup)
void *runtime_get_stream(RuntimeState *state);

//==============================================================================
// Inference API Types (for generated interface)
//==============================================================================

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

//==============================================================================
// Constant Management
//==============================================================================

// Upload constant to GPU and store at index
// Precondition: index assigned at compile-time (0, 1, 2, ...)
// Returns: 0=success, non-zero=error
int hip_upload_constant(RuntimeState *state, int64_t index, const void *data,
                        int64_t size);

// Get GPU pointer for constant at index
// Returns: GPU pointer (NULL if not uploaded or error)
// Ownership: Caller does NOT own pointer (freed in release_constant)
void *hip_get_constant(RuntimeState *state, int64_t index);

// Release GPU memory for constant at index
// Returns: 0=success, non-zero=error
int hip_release_constant(RuntimeState *state, int64_t index);

//==============================================================================
// Library Operations (MIOpen, hipBLAS)
//==============================================================================

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

//==============================================================================
// Low-Level HIP Wrappers
//==============================================================================

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
