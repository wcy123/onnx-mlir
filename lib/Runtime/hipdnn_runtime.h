#ifndef HIPDNN_RUNTIME_H
#define HIPDNN_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for runtime state
typedef struct RuntimeState RuntimeState;

// Constant management functions
// Called by initialize_constants during inference_init
// Uploads constant data from DLL .data section to GPU memory
int hip_upload_constant(RuntimeState* state, int64_t index, const void* data, int64_t size);

// Retrieves GPU pointer for a constant by index
// Returns NULL on error
void* hip_get_constant(RuntimeState* state, int64_t index);

// Releases GPU memory for a constant
// Called by release_constants during inference_cleanup
int hip_release_constant(RuntimeState* state, int64_t index);

// MIOpen convolution forward operation
// Full wrapper with descriptor creation, algorithm finding, workspace management
// Parameters match generated LLVM IR from @main_internal
int miopenConvolutionForward(
    void* handle,           // MIOpen handle
    void* stream,           // HIP stream
    const void* input,      // Input tensor GPU pointer
    const int64_t* input_shape,   // [N, C, H, W]
    const void* weights,    // Weights tensor GPU pointer
    const int64_t* weights_shape, // [K, C, R, S]
    void* output,           // Output tensor GPU pointer
    const int64_t* output_shape,  // [N, K, H', W']
    int64_t pad_h,          // Padding height
    int64_t pad_w,          // Padding width
    int64_t stride_h,       // Stride height
    int64_t stride_w,       // Stride width
    int64_t dilation_h,     // Dilation height
    int64_t dilation_w);    // Dilation width

// hipBLASLt GEMM operation wrapper
// Called by generated IR for matrix multiplication operations
int hipblasLtGemmWrapper(
    void* handle,           // hipBLASLt handle
    void* stream,           // HIP stream
    int64_t m, int64_t n, int64_t k,
    const void* alpha,      // Scalar alpha
    const void* A,          // Matrix A GPU pointer
    const void* B,          // Matrix B GPU pointer
    const void* beta,       // Scalar beta
    void* C);               // Matrix C GPU pointer (in/out)

// HIP memory allocation wrapper with error handling
int hip_malloc_wrapper(void** ptr, int64_t size);

// HIP memory free wrapper with error handling
int hip_free_wrapper(void* ptr);

// HIP memory copy host-to-device wrapper
int hip_memcpy_h2d_async(void* dst, const void* src, int64_t size, void* stream);

// HIP memory copy device-to-host wrapper
int hip_memcpy_d2h_async(void* dst, const void* src, int64_t size, void* stream);

// HIP stream synchronization wrapper
int hip_stream_synchronize_wrapper(void* stream);

#ifdef __cplusplus
}
#endif

#endif // HIPDNN_RUNTIME_H
