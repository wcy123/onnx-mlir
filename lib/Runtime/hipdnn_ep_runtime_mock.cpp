/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hipdnn_ep_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

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

// Mock HIP memory functions (non-static for cross-module linking)
hipError_t hipMalloc(void **ptr, size_t size) {
  *ptr = malloc(size);
  printf("[MOCK] hipMalloc(%zu bytes) -> %p\n", size, *ptr);
  return *ptr ? hipSuccess : -1;
}

hipError_t hipFree(void *ptr) {
  printf("[MOCK] hipFree(%p)\n", ptr);
  free(ptr);
  return hipSuccess;
}

hipError_t hipMemcpyAsync(void *dst, const void *src, size_t size, int kind,
                          hipStream_t stream) {
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

// Mock error checking macros
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

// Mock wrapper implementations (called from generated MLIR code)

int wrap_miopenConvolutionForward(
    void *handle, void *stream, const void *input, const int64_t *input_shape,
    const void *weights, const int64_t *weights_shape, void *output,
    const int64_t *output_shape, int64_t pad_h, int64_t pad_w, int64_t stride_h,
    int64_t stride_w, int64_t dilation_h, int64_t dilation_w) {
  if (!handle || !stream || !input || !weights || !output) {
    fprintf(stderr, "Invalid arguments to wrap_miopenConvolutionForward\n");
    return -1;
  }

  printf("[MOCK] wrap_miopenConvolutionForward(\n");
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

int wrap_hipblasLtGemm(void *handle, void *stream, int64_t m, int64_t n,
                       int64_t k, const void *alpha, const void *A,
                       const void *B, const void *beta, void *C) {
  if (!handle || !stream || !alpha || !A || !B || !beta || !C) {
    fprintf(stderr, "Invalid arguments to wrap_hipblasLtGemm\n");
    return -1;
  }

  printf("[MOCK] wrap_hipblasLtGemm(M=%lld, N=%lld, K=%lld)\n", (long long)m,
         (long long)n, (long long)k);

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

int wrap_hipMalloc(void **ptr, int64_t size) {
  HIP_CHECK(hipMalloc(ptr, size));
  return 0;
}

int wrap_hipFree(void *ptr) {
  HIP_CHECK(hipFree(ptr));
  return 0;
}

int wrap_hipMemcpyH2D(void *dst, const void *src, int64_t size, void *stream) {
  HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice,
                           static_cast<hipStream_t>(stream)));
  return 0;
}

int wrap_hipMemcpyD2H(void *dst, const void *src, int64_t size, void *stream) {
  HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost,
                           static_cast<hipStream_t>(stream)));
  return 0;
}

int wrap_hipStreamSynchronize(void *stream) {
  HIP_CHECK(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
  return 0;
}
