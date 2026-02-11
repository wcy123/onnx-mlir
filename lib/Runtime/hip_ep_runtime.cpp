#include "hip_ep_runtime.h"

#ifndef BUILD_MOCK_RUNTIME
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <hipblaslt/hipblaslt.h>
#else
// Mock definitions when ROCm is not available
typedef void* hipStream_t;
typedef void* miopenHandle_t;
typedef void* hipblasLtHandle_t;
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
#include <vector>
#include <unordered_map>

// Internal runtime state structure
struct RuntimeState {
    hipStream_t stream;
    miopenHandle_t miopen_handle;
    hipblasLtHandle_t hipblas_handle;

    // Map from constant index to GPU pointer
    std::unordered_map<int64_t, void*> constants;
};

// Error checking macros
#ifdef BUILD_MOCK_RUNTIME
#define HIP_CHECK(cmd) do { (void)(cmd); } while(0)
#define MIOPEN_CHECK(cmd) do { (void)(cmd); } while(0)
#define HIPBLAS_CHECK(cmd) do { (void)(cmd); } while(0)
#define hipGetErrorString(e) "mock_error"
#else
#define HIP_CHECK(cmd) do { \
    hipError_t error = (cmd); \
    if (error != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(error)); \
        return -1; \
    } \
} while(0)

#define MIOPEN_CHECK(cmd) do { \
    miopenStatus_t status = (cmd); \
    if (status != miopenStatusSuccess) { \
        fprintf(stderr, "MIOpen error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        return -1; \
    } \
} while(0)

#define HIPBLAS_CHECK(cmd) do { \
    hipblasStatus_t status = (cmd); \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "hipBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        return -1; \
    } \
} while(0)
#endif

#ifdef BUILD_MOCK_RUNTIME
// Mock HIP functions when ROCm is not available
static hipError_t hipMalloc(void** ptr, size_t size) {
    *ptr = malloc(size);
    return *ptr ? hipSuccess : -1;
}

static hipError_t hipFree(void* ptr) {
    free(ptr);
    return hipSuccess;
}

static hipError_t hipMemcpyAsync(void* dst, const void* src, size_t size, int kind, hipStream_t stream) {
    (void)kind; (void)stream;
    memcpy(dst, src, size);
    return hipSuccess;
}
#endif

// Constant management implementation
int hip_upload_constant(RuntimeState* state, int64_t index, const void* data, int64_t size) {
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
    void* gpu_ptr = nullptr;
    HIP_CHECK(hipMalloc(&gpu_ptr, size));

    // Copy data from DLL .data section to GPU
    HIP_CHECK(hipMemcpyAsync(gpu_ptr, data, size, hipMemcpyHostToDevice, state->stream));

    // Store in map
    state->constants[index] = gpu_ptr;

    return 0;
}

void* hip_get_constant(RuntimeState* state, int64_t index) {
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

int hip_release_constant(RuntimeState* state, int64_t index) {
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
int miopenConvolutionForward(
    void* handle,
    void* stream,
    const void* input,
    const int64_t* input_shape,
    const void* weights,
    const int64_t* weights_shape,
    void* output,
    const int64_t* output_shape,
    int64_t pad_h,
    int64_t pad_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t dilation_h,
    int64_t dilation_w)
{
    if (!handle || !stream || !input || !weights || !output) {
        fprintf(stderr, "Invalid arguments to miopenConvolutionForward\n");
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
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(
        input_desc, miopenFloat,
        input_shape[0], input_shape[1], input_shape[2], input_shape[3]));

    MIOPEN_CHECK(miopenSet4dTensorDescriptor(
        weights_desc, miopenFloat,
        weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3]));

    MIOPEN_CHECK(miopenSet4dTensorDescriptor(
        output_desc, miopenFloat,
        output_shape[0], output_shape[1], output_shape[2], output_shape[3]));

    // Create convolution descriptor
    miopenConvolutionDescriptor_t conv_desc;
    MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&conv_desc));
    MIOPEN_CHECK(miopenInitConvolutionDescriptor(
        conv_desc,
        miopenConvolution,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w));

    // Find best algorithm
    miopenConvFwdAlgorithm_t algo;
    MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
        miopen_handle,
        input_desc, input,
        weights_desc, weights,
        conv_desc,
        output_desc, output,
        1,  // requestAlgoCount
        &algo,
        nullptr,  // returnedAlgoCount
        nullptr,  // workspace (nullptr to query size)
        0,  // workspaceSize
        false));  // exhaustiveSearch

    // Get workspace size
    size_t workspace_size = 0;
    MIOPEN_CHECK(miopenConvolutionForwardGetWorkSpaceSize(
        miopen_handle,
        weights_desc,
        input_desc,
        conv_desc,
        output_desc,
        &workspace_size));

    // Allocate workspace
    void* workspace = nullptr;
    if (workspace_size > 0) {
        HIP_CHECK(hipMalloc(&workspace, workspace_size));
    }

    // Perform convolution
    float alpha = 1.0f;
    float beta = 0.0f;
    MIOPEN_CHECK(miopenConvolutionForward(
        miopen_handle,
        &alpha,
        input_desc, input,
        weights_desc, weights,
        conv_desc,
        algo,
        &beta,
        output_desc, output,
        workspace, workspace_size));

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
int hipblasLtGemmWrapper(
    void* handle,
    void* stream,
    int64_t m, int64_t n, int64_t k,
    const void* alpha,
    const void* A,
    const void* B,
    const void* beta,
    void* C)
{
    if (!handle || !stream || !alpha || !A || !B || !beta || !C) {
        fprintf(stderr, "Invalid arguments to hipblasLtGemmWrapper\n");
        return -1;
    }

    hipblasLtHandle_t hipblas_handle = static_cast<hipblasLtHandle_t>(handle);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    // Create matrix descriptors (assuming float32, column-major)
    hipblasLtMatrixLayout_t matA, matB, matC;
    HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matA, HIPBLAS_R_32F, m, k, m));
    HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matB, HIPBLAS_R_32F, k, n, k));
    HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matC, HIPBLAS_R_32F, m, n, m));

    // Create operation descriptor
    hipblasLtMatmulDesc_t matmul_desc;
    HIPBLAS_CHECK(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIPBLAS_R_32F));

    // Perform GEMM
    HIPBLAS_CHECK(hipblasLtMatmul(
        hipblas_handle,
        matmul_desc,
        alpha,
        A, matA,
        B, matB,
        beta,
        C, matC,
        C, matC,
        nullptr,  // algo
        nullptr,  // workspace
        0,        // workspaceSize
        hip_stream));

    // Cleanup
    hipblasLtMatrixLayoutDestroy(matA);
    hipblasLtMatrixLayoutDestroy(matB);
    hipblasLtMatrixLayoutDestroy(matC);
    hipblasLtMatmulDescDestroy(matmul_desc);

    return 0;
}

// HIP memory wrappers
int hip_malloc_wrapper(void** ptr, int64_t size) {
    HIP_CHECK(hipMalloc(ptr, size));
    return 0;
}

int hip_free_wrapper(void* ptr) {
    HIP_CHECK(hipFree(ptr));
    return 0;
}

int hip_memcpy_h2d_async(void* dst, const void* src, int64_t size, void* stream) {
    HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice,
                             static_cast<hipStream_t>(stream)));
    return 0;
}

int hip_memcpy_d2h_async(void* dst, const void* src, int64_t size, void* stream) {
    HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost,
                             static_cast<hipStream_t>(stream)));
    return 0;
}

int hip_stream_synchronize_wrapper(void* stream) {
    HIP_CHECK(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
    return 0;
}
