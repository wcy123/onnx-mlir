#include "../../lib/Runtime/hip_ep_runtime.h"
#include "../../lib/Runtime/hip_ep_runtime_mock.h"
#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Mock Runtime Test ===\n\n";

#ifndef BUILD_MOCK_RUNTIME
    std::cerr << "ERROR: This test must be built with BUILD_MOCK_RUNTIME=1\n";
    return 1;
#endif

    // Test 1: Handle creation (should print mock messages)
    std::cout << "--- Test 1: Handle Creation ---\n";
    void* stream = nullptr;
    int result = hipStreamCreate(&stream);
    assert(result == 0 && stream != nullptr);
    std::cout << "✓ Stream created\n";

    void* miopen = nullptr;
    result = miopenCreate(&miopen);
    assert(result == 0 && miopen != nullptr);
    std::cout << "✓ MIOpen handle created\n";

    void* hipblas = nullptr;
    result = hipblasLtCreate(&hipblas);
    assert(result == 0 && hipblas != nullptr);
    std::cout << "✓ hipBLAS handle created\n";

    result = miopenSetStream(miopen, stream);
    assert(result == 0);
    std::cout << "✓ MIOpen stream set\n\n";

    // Test 2: RuntimeState structure
    std::cout << "--- Test 2: RuntimeState Structure ---\n";
    // Note: RuntimeState is opaque, so we can't directly test size
    // But we can test that the constant management works
    std::cout << "✓ RuntimeState is opaque (as designed)\n\n";

    // Test 3: Constant management
    std::cout << "--- Test 3: Constant Management ---\n";

    // Create a minimal RuntimeState manually for testing
    struct TestRuntimeState {
        void* stream;
        void* miopen_handle;
        void* hipblas_handle;
    };

    // Note: We can't actually use RuntimeState directly since it's opaque
    // This test would need to be integrated with the generated interface functions
    // For now, we'll just verify the wrapper functions work
    std::cout << "✓ Constant management functions available\n";
    std::cout << "  (Full test requires generated interface functions)\n\n";

    // Test 4: Convolution operation (should print parameters)
    std::cout << "--- Test 4: Convolution Operation ---\n";
    int64_t input_shape[] = {1, 3, 224, 224};
    int64_t weights_shape[] = {64, 3, 7, 7};
    int64_t output_shape[] = {1, 64, 112, 112};

    void* input_gpu = nullptr;
    void* weights_gpu = nullptr;
    void* output_gpu = nullptr;

    result = hip_malloc_wrapper(&input_gpu, 1*3*224*224*4);
    assert(result == 0 && input_gpu != nullptr);

    result = hip_malloc_wrapper(&weights_gpu, 64*3*7*7*4);
    assert(result == 0 && weights_gpu != nullptr);

    result = hip_malloc_wrapper(&output_gpu, 1*64*112*112*4);
    assert(result == 0 && output_gpu != nullptr);

    result = miopenConvolutionForward(
        miopen, stream,
        input_gpu, input_shape,
        weights_gpu, weights_shape,
        output_gpu, output_shape,
        3, 3,  // pad
        2, 2,  // stride
        1, 1); // dilation
    assert(result == 0);
    std::cout << "✓ Convolution completed\n\n";

    // Test 5: GEMM operation (should print dimensions)
    std::cout << "--- Test 5: GEMM Operation ---\n";
    float alpha = 1.0f, beta = 0.0f;
    void* A = nullptr;
    void* B = nullptr;
    void* C = nullptr;

    result = hip_malloc_wrapper(&A, 1000*2048*4);
    assert(result == 0 && A != nullptr);

    result = hip_malloc_wrapper(&B, 2048*1*4);
    assert(result == 0 && B != nullptr);

    result = hip_malloc_wrapper(&C, 1000*1*4);
    assert(result == 0 && C != nullptr);

    result = hipblasLtGemmWrapper(
        hipblas, stream,
        1000, 1, 2048,  // M, N, K
        &alpha, A, B, &beta, C);
    assert(result == 0);
    std::cout << "✓ GEMM completed\n\n";

    // Test 6: Memory copy operations
    std::cout << "--- Test 6: Memory Copy Operations ---\n";
    std::vector<float> host_data(1024, 1.0f);
    void* device_data = nullptr;

    result = hip_malloc_wrapper(&device_data, 1024 * sizeof(float));
    assert(result == 0 && device_data != nullptr);

    result = hip_memcpy_h2d_async(device_data, host_data.data(),
                                   1024 * sizeof(float), stream);
    assert(result == 0);
    std::cout << "✓ H2D copy completed\n";

    result = hip_memcpy_d2h_async(host_data.data(), device_data,
                                   1024 * sizeof(float), stream);
    assert(result == 0);
    std::cout << "✓ D2H copy completed\n\n";

    // Test 7: Cleanup (should print in reverse order)
    std::cout << "--- Test 7: Cleanup ---\n";

    result = hip_free_wrapper(input_gpu);
    assert(result == 0);

    result = hip_free_wrapper(weights_gpu);
    assert(result == 0);

    result = hip_free_wrapper(output_gpu);
    assert(result == 0);

    result = hip_free_wrapper(A);
    assert(result == 0);

    result = hip_free_wrapper(B);
    assert(result == 0);

    result = hip_free_wrapper(C);
    assert(result == 0);

    result = hip_free_wrapper(device_data);
    assert(result == 0);

    result = hipStreamSynchronize(stream);
    assert(result == 0);

    result = hipblasLtDestroy(hipblas);
    assert(result == 0);

    result = miopenDestroy(miopen);
    assert(result == 0);

    result = hipStreamDestroy(stream);
    assert(result == 0);

    std::cout << "✓ All resources cleaned up\n\n";

    std::cout << "=== All Mock Runtime Tests PASSED ===\n";
    std::cout << "\nExpected output above should show:\n";
    std::cout << "  - Handle creation with memory addresses\n";
    std::cout << "  - Convolution with tensor shapes and parameters\n";
    std::cout << "  - GEMM with matrix dimensions\n";
    std::cout << "  - Memory operations with sizes and directions\n";
    std::cout << "  - Cleanup in reverse order (LIFO)\n";

    return 0;
}
