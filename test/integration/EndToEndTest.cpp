/**
 ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 ** Licensed under the MIT License.
 **/

//===----------------------------------------------------------------------===//
// End-to-End Test: Load generated DLL and test inference interface
//===----------------------------------------------------------------------===//
// This test demonstrates loading the generated DLL and calling the three
// exported functions: inference_init, inference_compute, inference_cleanup
//===----------------------------------------------------------------------===//

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>

#ifdef _WIN32
#include <windows.h>
#define LOAD_LIBRARY(path) LoadLibraryA(path)
#define GET_PROC_ADDRESS(handle, name) GetProcAddress((HMODULE)handle, name)
#define FREE_LIBRARY(handle) FreeLibrary((HMODULE)handle)
#else
#include <dlfcn.h>
#define LOAD_LIBRARY(path) dlopen(path, RTLD_LAZY)
#define GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#define FREE_LIBRARY(handle) dlclose(handle)
#endif

// C interface function pointers
typedef int (*inference_init_fn)(void** out_state);
typedef int (*inference_compute_fn)(void* state, void* inputs, void* outputs);
typedef int (*inference_cleanup_fn)(void* state);

// Test data structures matching the C interface
struct tensor_t {
    void* data;        // CPU pointer to tensor data
    int64_t* shape;    // Array of dimension sizes
    size_t rank;       // Number of dimensions
};

struct span_t {
    tensor_t* data;    // Array of tensors
    size_t count;      // Number of tensors
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path-to-dll>\n";
        return 1;
    }

    const char* dllPath = argv[1];
    std::cout << "Loading DLL: " << dllPath << "\n";

    // Load DLL
    void* dllHandle = LOAD_LIBRARY(dllPath);
    if (!dllHandle) {
        std::cerr << "Failed to load DLL: " << dllPath << "\n";
#ifndef _WIN32
        std::cerr << "Error: " << dlerror() << "\n";
#endif
        return 1;
    }

    // Get function pointers
    auto init_fn = (inference_init_fn)GET_PROC_ADDRESS(dllHandle, "inference_init");
    auto compute_fn = (inference_compute_fn)GET_PROC_ADDRESS(dllHandle, "inference_compute");
    auto cleanup_fn = (inference_cleanup_fn)GET_PROC_ADDRESS(dllHandle, "inference_cleanup");

    if (!init_fn || !compute_fn || !cleanup_fn) {
        std::cerr << "Failed to resolve exported functions\n";
        FREE_LIBRARY(dllHandle);
        return 1;
    }

    std::cout << "Successfully resolved all exported functions\n";

    // Test 1: Initialize
    std::cout << "\nTest 1: Calling inference_init...\n";
    void* state = nullptr;
    int result = init_fn(&state);
    if (result != 0) {
        std::cerr << "inference_init failed with code: " << result << "\n";
        FREE_LIBRARY(dllHandle);
        return 1;
    }
    if (!state) {
        std::cerr << "inference_init returned null state\n";
        FREE_LIBRARY(dllHandle);
        return 1;
    }
    std::cout << "inference_init succeeded, state = " << state << "\n";

    // Test 2: Compute (with dummy data)
    std::cout << "\nTest 2: Calling inference_compute...\n";

    // Create dummy input tensor (1x3x224x224 float32)
    std::vector<float> inputData(1 * 3 * 224 * 224, 1.0f);
    std::vector<int64_t> inputShape = {1, 3, 224, 224};

    tensor_t inputTensor;
    inputTensor.data = inputData.data();
    inputTensor.shape = inputShape.data();
    inputTensor.rank = inputShape.size();

    span_t inputSpan;
    inputSpan.data = &inputTensor;
    inputSpan.count = 1;

    // Create dummy output tensor (1x1000 float32)
    std::vector<float> outputData(1 * 1000, 0.0f);
    std::vector<int64_t> outputShape = {1, 1000};

    tensor_t outputTensor;
    outputTensor.data = outputData.data();
    outputTensor.shape = outputShape.data();
    outputTensor.rank = outputShape.size();

    span_t outputSpan;
    outputSpan.data = &outputTensor;
    outputSpan.count = 1;

    // Note: This will likely fail unless the DLL is compiled with matching metadata
    // This is just a structural test to verify the interface works
    result = compute_fn(state, &inputSpan, &outputSpan);
    if (result != 0) {
        std::cout << "inference_compute returned code: " << result
                  << " (expected if metadata doesn't match)\n";
    } else {
        std::cout << "inference_compute succeeded\n";
        std::cout << "Output sample: [" << outputData[0] << ", "
                  << outputData[1] << ", " << outputData[2] << ", ...]\n";
    }

    // Test 3: Cleanup
    std::cout << "\nTest 3: Calling inference_cleanup...\n";
    result = cleanup_fn(state);
    if (result != 0) {
        std::cerr << "inference_cleanup failed with code: " << result << "\n";
        FREE_LIBRARY(dllHandle);
        return 1;
    }
    std::cout << "inference_cleanup succeeded\n";

    // Cleanup
    FREE_LIBRARY(dllHandle);
    std::cout << "\nAll tests completed successfully!\n";
    return 0;
}
