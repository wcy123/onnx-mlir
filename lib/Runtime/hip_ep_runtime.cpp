/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hip_ep_runtime.h"

#ifndef BUILD_MOCK_RUNTIME
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <miopen/miopen.h>
#else
// Mock definitions when ROCm is not available
typedef void *hipStream_t;
typedef void *miopenHandle_t;
typedef void *hipblasLtHandle_t;
typedef int hipError_t;
typedef int miopenStatus_t;
typedef int hipblasStatus_t;
#define hipSuccess 0
#define miopenStatusSuccess 0
#define HIPBLAS_STATUS_SUCCESS 0
#define hipMemcpyHostToDevice 0
#define hipMemcpyDeviceToHost 0
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

// Internal runtime state structure
struct RuntimeState {
  hipStream_t stream;
  miopenHandle_t miopen_handle;
  hipblasLtHandle_t hipblas_handle;

  // Map from constant index to GPU pointer
  std::unordered_map<int64_t, void *> constants;
};

// Error checking macros
#ifdef BUILD_MOCK_RUNTIME
#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    (void)(cmd);                                                               \
  } while (0)
#define MIOPEN_CHECK(cmd)                                                      \
  do {                                                                         \
    (void)(cmd);                                                               \
  } while (0)
#define HIPBLAS_CHECK(cmd)                                                     \
  do {                                                                         \
    (void)(cmd);                                                               \
  } while (0)
#define hipGetErrorString(e) "mock_error"
#else
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

#define HIPBLAS_CHECK(cmd)                                                     \
  do {                                                                         \
    hipblasStatus_t status = (cmd);                                            \
    if (status != HIPBLAS_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "hipBLAS error at %s:%d: %d\n", __FILE__, __LINE__,      \
              status);                                                         \
      return -1;                                                               \
    }                                                                          \
  } while (0)
#endif

#ifdef BUILD_MOCK_RUNTIME
// Comprehensive mock implementations for all GPU functions
// Prints all operations for debugging and verification

// Mock HIP stream functions (non-static so test can link against them)
extern "C" hipError_t hipStreamCreate(hipStream_t *stream) {
  *stream = malloc(8); // Fake handle
  printf("[MOCK] hipStreamCreate() -> %p\n", *stream);
  return hipSuccess;
}

extern "C" hipError_t hipStreamDestroy(hipStream_t stream) {
  printf("[MOCK] hipStreamDestroy(%p)\n", stream);
  free(stream);
  return hipSuccess;
}

extern "C" hipError_t hipStreamSynchronize(hipStream_t stream) {
  printf("[MOCK] hipStreamSynchronize(%p)\n", stream);
  return hipSuccess;
}

// Mock HIP memory functions
static hipError_t hipMalloc(void **ptr, size_t size) {
  *ptr = malloc(size);
  printf("[MOCK] hipMalloc(%zu bytes) -> %p\n", size, *ptr);
  return *ptr ? hipSuccess : -1;
}

static hipError_t hipFree(void *ptr) {
  printf("[MOCK] hipFree(%p)\n", ptr);
  free(ptr);
  return hipSuccess;
}

static hipError_t hipMemcpyAsync(void *dst, const void *src, size_t size,
                                 int kind, hipStream_t stream) {
  const char *kind_str = (kind == hipMemcpyHostToDevice)   ? "H2D"
                         : (kind == hipMemcpyDeviceToHost) ? "D2H"
                                                           : "D2D";
  printf("[MOCK] hipMemcpyAsync(dst=%p, src=%p, size=%zu, %s, stream=%p)\n",
         dst, src, size, kind_str, stream);
  memcpy(dst, src, size);
  return hipSuccess;
}

// Mock MIOpen types and constants
typedef void *miopenTensorDescriptor_t;
typedef void *miopenConvolutionDescriptor_t;
typedef enum { miopenFloat = 0 } miopenDataType_t;
typedef enum { miopenConvolution = 0 } miopenConvolutionMode_t;
typedef enum { miopenConvolutionFwdAlgoGEMM = 0 } miopenConvFwdAlgorithm_t;

// Mock MIOpen handle functions (non-static so test can link against them)
extern "C" miopenStatus_t miopenCreate(miopenHandle_t *handle) {
  *handle = malloc(8); // Fake handle
  printf("[MOCK] miopenCreate() -> %p\n", *handle);
  return miopenStatusSuccess;
}

extern "C" miopenStatus_t miopenDestroy(miopenHandle_t handle) {
  printf("[MOCK] miopenDestroy(%p)\n", handle);
  free(handle);
  return miopenStatusSuccess;
}

extern "C" miopenStatus_t miopenSetStream(miopenHandle_t handle,
                                          hipStream_t stream) {
  printf("[MOCK] miopenSetStream(handle=%p, stream=%p)\n", handle, stream);
  return miopenStatusSuccess;
}

// Mock MIOpen tensor descriptor functions
static miopenStatus_t
miopenCreateTensorDescriptor(miopenTensorDescriptor_t *desc) {
  *desc = malloc(8); // Fake descriptor
  return miopenStatusSuccess;
}

static miopenStatus_t
miopenDestroyTensorDescriptor(miopenTensorDescriptor_t desc) {
  free(desc);
  return miopenStatusSuccess;
}

static miopenStatus_t miopenSet4dTensorDescriptor(miopenTensorDescriptor_t desc,
                                                  miopenDataType_t dataType,
                                                  int n, int c, int h, int w) {
  (void)desc;
  (void)dataType;
  // Print tensor shape for verification
  printf("[MOCK]   Tensor descriptor set: [%d, %d, %d, %d]\n", n, c, h, w);
  return miopenStatusSuccess;
}

// Mock MIOpen convolution descriptor functions
static miopenStatus_t
miopenCreateConvolutionDescriptor(miopenConvolutionDescriptor_t *desc) {
  *desc = malloc(8); // Fake descriptor
  return miopenStatusSuccess;
}

static miopenStatus_t
miopenDestroyConvolutionDescriptor(miopenConvolutionDescriptor_t desc) {
  free(desc);
  return miopenStatusSuccess;
}

static miopenStatus_t miopenInitConvolutionDescriptor(
    miopenConvolutionDescriptor_t desc, miopenConvolutionMode_t mode, int pad_h,
    int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w) {
  (void)desc;
  (void)mode;
  printf("[MOCK]   Convolution params: pad=[%d,%d], stride=[%d,%d], "
         "dilation=[%d,%d]\n",
         pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
  return miopenStatusSuccess;
}

// Mock MIOpen convolution algorithm finding
static miopenStatus_t miopenFindConvolutionForwardAlgorithm(
    miopenHandle_t handle, miopenTensorDescriptor_t input_desc,
    const void *input, miopenTensorDescriptor_t weights_desc,
    const void *weights, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t output_desc, const void *output,
    int requestAlgoCount, miopenConvFwdAlgorithm_t *algo,
    int *returnedAlgoCount, void *workspace, size_t workspaceSize,
    bool exhaustiveSearch) {
  (void)handle;
  (void)input_desc;
  (void)input;
  (void)weights_desc;
  (void)weights;
  (void)conv_desc;
  (void)output_desc;
  (void)output;
  (void)requestAlgoCount;
  (void)returnedAlgoCount;
  (void)workspace;
  (void)workspaceSize;
  (void)exhaustiveSearch;

  printf("[MOCK]   Finding convolution algorithm...\n");
  if (algo)
    *algo = miopenConvolutionFwdAlgoGEMM;
  return miopenStatusSuccess;
}

static miopenStatus_t miopenConvolutionForwardGetWorkSpaceSize(
    miopenHandle_t handle, miopenTensorDescriptor_t weights_desc,
    miopenTensorDescriptor_t input_desc,
    miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t output_desc, size_t *workspaceSize) {
  (void)handle;
  (void)weights_desc;
  (void)input_desc;
  (void)conv_desc;
  (void)output_desc;
  *workspaceSize = 0; // No workspace needed in mock
  return miopenStatusSuccess;
}

static miopenStatus_t miopenConvolutionForward(
    miopenHandle_t handle, const void *alpha,
    miopenTensorDescriptor_t input_desc, const void *input,
    miopenTensorDescriptor_t weights_desc, const void *weights,
    miopenConvolutionDescriptor_t conv_desc, miopenConvFwdAlgorithm_t algo,
    const void *beta, miopenTensorDescriptor_t output_desc, void *output,
    void *workspace, size_t workspaceSize) {
  (void)handle;
  (void)alpha;
  (void)input_desc;
  (void)input;
  (void)weights_desc;
  (void)weights;
  (void)conv_desc;
  (void)algo;
  (void)beta;
  (void)output_desc;
  (void)output;
  (void)workspace;
  (void)workspaceSize;

  printf("[MOCK]   Executing convolution forward pass\n");
  return miopenStatusSuccess;
}

// Mock hipBLASLt types and constants
typedef void *hipblasLtMatrixLayout_t;
typedef void *hipblasLtMatmulDesc_t;
typedef enum { HIPBLAS_R_32F = 0 } hipblasDatatype_t;
typedef enum { HIPBLAS_COMPUTE_32F = 0 } hipblasComputeType_t;

// Mock hipBLASLt handle functions (non-static so test can link against them)
extern "C" hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t *handle) {
  *handle = malloc(8); // Fake handle
  printf("[MOCK] hipblasLtCreate() -> %p\n", *handle);
  return HIPBLAS_STATUS_SUCCESS;
}

extern "C" hipblasStatus_t hipblasLtDestroy(hipblasLtHandle_t handle) {
  printf("[MOCK] hipblasLtDestroy(%p)\n", handle);
  free(handle);
  return HIPBLAS_STATUS_SUCCESS;
}

// Mock hipBLASLt matrix layout functions
static hipblasStatus_t
hipblasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t *layout,
                            hipblasDatatype_t type, uint64_t rows,
                            uint64_t cols, int64_t ld) {
  (void)type;
  (void)ld;
  *layout = malloc(8); // Fake layout
  printf("[MOCK]   Matrix layout: [%llu x %llu]\n", (unsigned long long)rows,
         (unsigned long long)cols);
  return HIPBLAS_STATUS_SUCCESS;
}

static hipblasStatus_t
hipblasLtMatrixLayoutDestroy(hipblasLtMatrixLayout_t layout) {
  free(layout);
  return HIPBLAS_STATUS_SUCCESS;
}

// Mock hipBLASLt matmul descriptor functions
static hipblasStatus_t
hipblasLtMatmulDescCreate(hipblasLtMatmulDesc_t *desc,
                          hipblasComputeType_t computeType,
                          hipblasDatatype_t dataType) {
  (void)computeType;
  (void)dataType;
  *desc = malloc(8); // Fake descriptor
  return HIPBLAS_STATUS_SUCCESS;
}

static hipblasStatus_t hipblasLtMatmulDescDestroy(hipblasLtMatmulDesc_t desc) {
  free(desc);
  return HIPBLAS_STATUS_SUCCESS;
}

// Mock hipBLASLt matmul function
static hipblasStatus_t
hipblasLtMatmul(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmul_desc,
                const void *alpha, const void *A, hipblasLtMatrixLayout_t matA,
                const void *B, hipblasLtMatrixLayout_t matB, const void *beta,
                const void *C, hipblasLtMatrixLayout_t matC, void *D,
                hipblasLtMatrixLayout_t matD, const void *algo, void *workspace,
                size_t workspaceSize, hipStream_t stream) {
  (void)handle;
  (void)matmul_desc;
  (void)alpha;
  (void)A;
  (void)matA;
  (void)B;
  (void)matB;
  (void)beta;
  (void)C;
  (void)matC;
  (void)D;
  (void)matD;
  (void)algo;
  (void)workspace;
  (void)workspaceSize;
  (void)stream;

  printf("[MOCK]   Executing GEMM operation\n");
  return HIPBLAS_STATUS_SUCCESS;
}

#endif // BUILD_MOCK_RUNTIME

// Runtime state management implementation

int runtime_state_init(RuntimeState **out_state) {
  if (!out_state) {
    fprintf(stderr, "Invalid output parameter to runtime_state_init\n");
    return 1;
  }

  // Allocate context struct (32 bytes)
  RuntimeState *state = (RuntimeState *)malloc(sizeof(RuntimeState));
  if (!state) {
    fprintf(stderr, "Failed to allocate runtime state\n");
    return 1; // Allocation failed
  }

  // Initialize all fields to null for safe cleanup
  state->stream = nullptr;
  state->miopen_handle = nullptr;
  state->hipblas_handle = nullptr;

  // Create HIP stream
  if (hipStreamCreate(&state->stream) != hipSuccess) {
    fprintf(stderr, "Failed to create HIP stream\n");
    free(state);
    return 2; // Stream creation failed
  }

  // Create MIOpen handle
  if (miopenCreate(&state->miopen_handle) != miopenStatusSuccess) {
    fprintf(stderr, "Failed to create MIOpen handle\n");
    hipStreamDestroy(state->stream);
    free(state);
    return 3; // MIOpen creation failed
  }

  // Set stream for MIOpen handle
  if (miopenSetStream(state->miopen_handle, state->stream) !=
      miopenStatusSuccess) {
    fprintf(stderr, "Failed to set MIOpen stream\n");
    miopenDestroy(state->miopen_handle);
    hipStreamDestroy(state->stream);
    free(state);
    return 4; // Set stream failed
  }

  // Create hipBLASLt handle
  if (hipblasLtCreate(&state->hipblas_handle) != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "Failed to create hipBLASLt handle\n");
    miopenDestroy(state->miopen_handle);
    hipStreamDestroy(state->stream);
    free(state);
    return 5; // hipBLAS creation failed
  }

  // Success - return initialized state
  *out_state = state;
  return 0;
}

int runtime_state_cleanup(RuntimeState *state) {
  if (!state) {
    fprintf(stderr, "Invalid runtime state in cleanup\n");
    return 0; // Best-effort - don't fail
  }

  // Best-effort cleanup - continue even if operations fail
  // Cleanup in reverse order of initialization (LIFO)

  // Synchronize stream to ensure all GPU operations complete
  if (state->stream) {
    hipStreamSynchronize(state->stream);
  }

  // Destroy hipBLASLt handle
  if (state->hipblas_handle) {
    hipblasLtDestroy(state->hipblas_handle);
  }

  // Destroy MIOpen handle
  if (state->miopen_handle) {
    miopenDestroy(state->miopen_handle);
  }

  // Destroy HIP stream
  if (state->stream) {
    hipStreamDestroy(state->stream);
  }

  // Free the context struct itself
  free(state);

  return 0; // Best-effort cleanup always returns success
}

// Inference data management implementation

int runtime_prepare_inference(RuntimeState *state, span_t *inputs,
                              span_t *outputs, InferenceData **out_data) {
  if (!state || !inputs || !outputs || !out_data) {
    fprintf(stderr, "Invalid parameters to runtime_prepare_inference\n");
    return 1;
  }

  // Allocate InferenceData structure
  InferenceData *data = (InferenceData *)malloc(sizeof(InferenceData));
  if (!data) {
    fprintf(stderr, "Failed to allocate InferenceData\n");
    return 2;
  }

  // Initialize fields
  data->input_count = inputs->count;
  data->output_count = outputs->count;
  data->input_gpu_buffers = nullptr;
  data->output_gpu_buffers = nullptr;
  data->input_sizes = nullptr;
  data->output_sizes = nullptr;

  // Allocate arrays for tracking
  data->input_gpu_buffers = (void **)malloc(inputs->count * sizeof(void *));
  data->output_gpu_buffers = (void **)malloc(outputs->count * sizeof(void *));
  data->input_sizes = (int64_t *)malloc(inputs->count * sizeof(int64_t));
  data->output_sizes = (int64_t *)malloc(outputs->count * sizeof(int64_t));

  if (!data->input_gpu_buffers || !data->output_gpu_buffers ||
      !data->input_sizes || !data->output_sizes) {
    fprintf(stderr, "Failed to allocate tracking arrays\n");
    free(data->input_gpu_buffers);
    free(data->output_gpu_buffers);
    free(data->input_sizes);
    free(data->output_sizes);
    free(data);
    return 2;
  }

  // Initialize all GPU buffer pointers to null for safe cleanup
  for (size_t i = 0; i < inputs->count; i++) {
    data->input_gpu_buffers[i] = nullptr;
  }
  for (size_t i = 0; i < outputs->count; i++) {
    data->output_gpu_buffers[i] = nullptr;
  }

  // Process input tensors
  for (size_t i = 0; i < inputs->count; i++) {
    tensor_t *tensor = &inputs->data[i];

    // Calculate tensor size: product of all dimensions * sizeof(float)
    int64_t num_elements = 1;
    for (size_t d = 0; d < tensor->rank; d++) {
      num_elements *= tensor->shape[d];
    }
    int64_t size_bytes = num_elements * sizeof(float);
    data->input_sizes[i] = size_bytes;

    // Allocate GPU buffer
    if (hip_malloc_wrapper(&data->input_gpu_buffers[i], size_bytes) != 0) {
      fprintf(stderr, "Failed to allocate GPU buffer for input %zu\n", i);
      // Cleanup allocated buffers so far
      runtime_cleanup_inference(state, data, outputs);
      return 3;
    }

    // Copy H2D
    if (hip_memcpy_h2d_async(data->input_gpu_buffers[i], tensor->data,
                             size_bytes, state->stream) != 0) {
      fprintf(stderr, "Failed H2D copy for input %zu\n", i);
      runtime_cleanup_inference(state, data, outputs);
      return 4;
    }
  }

  // Process output tensors (allocate GPU buffers only, no H2D)
  for (size_t i = 0; i < outputs->count; i++) {
    tensor_t *tensor = &outputs->data[i];

    // Calculate tensor size
    int64_t num_elements = 1;
    for (size_t d = 0; d < tensor->rank; d++) {
      num_elements *= tensor->shape[d];
    }
    int64_t size_bytes = num_elements * sizeof(float);
    data->output_sizes[i] = size_bytes;

    // Allocate GPU buffer
    if (hip_malloc_wrapper(&data->output_gpu_buffers[i], size_bytes) != 0) {
      fprintf(stderr, "Failed to allocate GPU buffer for output %zu\n", i);
      runtime_cleanup_inference(state, data, outputs);
      return 3;
    }
  }

  // Success
  *out_data = data;
  return 0;
}

int runtime_cleanup_inference(RuntimeState *state, InferenceData *data,
                              span_t *outputs) {
  if (!data) {
    return 0; // Nothing to cleanup
  }

  // Copy outputs D2H (best-effort)
  if (state && outputs && data->output_gpu_buffers && data->output_sizes) {
    for (size_t i = 0; i < data->output_count; i++) {
      if (data->output_gpu_buffers[i] && i < outputs->count) {
        tensor_t *tensor = &outputs->data[i];
        hip_memcpy_d2h_async(tensor->data, data->output_gpu_buffers[i],
                             data->output_sizes[i], state->stream);
      }
    }

    // Synchronize stream to ensure D2H completes
    if (state->stream) {
      hip_stream_synchronize_wrapper(state->stream);
    }
  }

  // Free GPU buffers (best-effort)
  if (data->input_gpu_buffers) {
    for (size_t i = 0; i < data->input_count; i++) {
      if (data->input_gpu_buffers[i]) {
        hip_free_wrapper(data->input_gpu_buffers[i]);
      }
    }
    free(data->input_gpu_buffers);
  }

  if (data->output_gpu_buffers) {
    for (size_t i = 0; i < data->output_count; i++) {
      if (data->output_gpu_buffers[i]) {
        hip_free_wrapper(data->output_gpu_buffers[i]);
      }
    }
    free(data->output_gpu_buffers);
  }

  // Free tracking arrays
  free(data->input_sizes);
  free(data->output_sizes);

  // Free InferenceData structure
  free(data);

  return 0;
}

// Constant management implementation
int hip_upload_constant(RuntimeState *state, int64_t index, const void *data,
                        int64_t size) {
  if (!state || !data || size <= 0) {
    fprintf(stderr, "Invalid arguments to hip_upload_constant\n");
    return -1;
  }

  // Check if constant already exists (shouldn't happen, but defensive)
  if (state->constants.find(index) != state->constants.end()) {
    fprintf(stderr, "Constant %lld already uploaded\n", (long long)index);
    return -1;
  }

  // Allocate GPU memory
  void *gpu_ptr = nullptr;
  HIP_CHECK(hipMalloc(&gpu_ptr, size));

  // Copy data from DLL .data section to GPU
  HIP_CHECK(hipMemcpyAsync(gpu_ptr, data, size, hipMemcpyHostToDevice,
                           state->stream));

  // Store in map
  state->constants[index] = gpu_ptr;

  return 0;
}

void *hip_get_constant(RuntimeState *state, int64_t index) {
  if (!state) {
    fprintf(stderr, "Invalid runtime state\n");
    return nullptr;
  }

  auto it = state->constants.find(index);
  if (it == state->constants.end()) {
    fprintf(stderr, "Constant %lld not found\n", (long long)index);
    return nullptr;
  }

  return it->second;
}

int hip_release_constant(RuntimeState *state, int64_t index) {
  if (!state) {
    fprintf(stderr, "Invalid runtime state\n");
    return -1;
  }

  auto it = state->constants.find(index);
  if (it == state->constants.end()) {
    fprintf(stderr, "Constant %lld not found\n", (long long)index);
    return -1;
  }

  // Free GPU memory
  HIP_CHECK(hipFree(it->second));

  // Remove from map
  state->constants.erase(it);

  return 0;
}

// MIOpen convolution forward implementation
int miopenConvolutionForward(void *handle, void *stream, const void *input,
                             const int64_t *input_shape, const void *weights,
                             const int64_t *weights_shape, void *output,
                             const int64_t *output_shape, int64_t pad_h,
                             int64_t pad_w, int64_t stride_h, int64_t stride_w,
                             int64_t dilation_h, int64_t dilation_w) {
  if (!handle || !stream || !input || !weights || !output) {
    fprintf(stderr, "Invalid arguments to miopenConvolutionForward\n");
    return -1;
  }

#ifdef BUILD_MOCK_RUNTIME
  printf("[MOCK] miopenConvolutionForward(\n");
  printf("[MOCK]   input_shape=[%lld,%lld,%lld,%lld],\n",
         (long long)input_shape[0], (long long)input_shape[1],
         (long long)input_shape[2], (long long)input_shape[3]);
  printf("[MOCK]   weights_shape=[%lld,%lld,%lld,%lld],\n",
         (long long)weights_shape[0], (long long)weights_shape[1],
         (long long)weights_shape[2], (long long)weights_shape[3]);
  printf("[MOCK]   output_shape=[%lld,%lld,%lld,%lld],\n",
         (long long)output_shape[0], (long long)output_shape[1],
         (long long)output_shape[2], (long long)output_shape[3]);
  printf(
      "[MOCK]   pad=[%lld,%lld], stride=[%lld,%lld], dilation=[%lld,%lld])\n",
      (long long)pad_h, (long long)pad_w, (long long)stride_h,
      (long long)stride_w, (long long)dilation_h, (long long)dilation_w);
#endif

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

// hipBLASLt GEMM wrapper implementation
int hipblasLtGemmWrapper(void *handle, void *stream, int64_t m, int64_t n,
                         int64_t k, const void *alpha, const void *A,
                         const void *B, const void *beta, void *C) {
  if (!handle || !stream || !alpha || !A || !B || !beta || !C) {
    fprintf(stderr, "Invalid arguments to hipblasLtGemmWrapper\n");
    return -1;
  }

#ifdef BUILD_MOCK_RUNTIME
  printf("[MOCK] hipblasLtGemmWrapper(M=%lld, N=%lld, K=%lld)\n", (long long)m,
         (long long)n, (long long)k);
#endif

  hipblasLtHandle_t hipblas_handle = static_cast<hipblasLtHandle_t>(handle);
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  // Create matrix descriptors (assuming float32, column-major)
  hipblasLtMatrixLayout_t matA, matB, matC;
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matA, HIPBLAS_R_32F, m, k, m));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matB, HIPBLAS_R_32F, k, n, k));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matC, HIPBLAS_R_32F, m, n, m));

  // Create operation descriptor
  hipblasLtMatmulDesc_t matmul_desc;
  HIPBLAS_CHECK(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F,
                                          HIPBLAS_R_32F));

  // Perform GEMM
  HIPBLAS_CHECK(hipblasLtMatmul(hipblas_handle, matmul_desc, alpha, A, matA, B,
                                matB, beta, C, matC, C, matC,
                                nullptr, // algo
                                nullptr, // workspace
                                0,       // workspaceSize
                                hip_stream));

  // Cleanup
  hipblasLtMatrixLayoutDestroy(matA);
  hipblasLtMatrixLayoutDestroy(matB);
  hipblasLtMatrixLayoutDestroy(matC);
  hipblasLtMatmulDescDestroy(matmul_desc);

  return 0;
}

// HIP memory wrappers
int hip_malloc_wrapper(void **ptr, int64_t size) {
  HIP_CHECK(hipMalloc(ptr, size));
  return 0;
}

int hip_free_wrapper(void *ptr) {
  HIP_CHECK(hipFree(ptr));
  return 0;
}

int hip_memcpy_h2d_async(void *dst, const void *src, int64_t size,
                         void *stream) {
  HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice,
                           static_cast<hipStream_t>(stream)));
  return 0;
}

int hip_memcpy_d2h_async(void *dst, const void *src, int64_t size,
                         void *stream) {
  HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost,
                           static_cast<hipStream_t>(stream)));
  return 0;
}

int hip_stream_synchronize_wrapper(void *stream) {
  HIP_CHECK(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
  return 0;
}
