// Accuracy validation test for test_identity.dll
// Tests that output matches expected values for identity operation

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#define LOAD_LIBRARY(name) LoadLibraryA(name)
#define GET_FUNCTION(lib, name) GetProcAddress((HMODULE)(lib), name)
#define FREE_LIBRARY(lib) FreeLibrary((HMODULE)(lib))
#else
#include <dlfcn.h>
#define LOAD_LIBRARY(name) dlopen(name, RTLD_NOW)
#define GET_FUNCTION(lib, name) dlsym(lib, name)
#define FREE_LIBRARY(lib) dlclose(lib)
#endif

// Type definitions
struct tensor_t {
    void* data;
    int64_t* shape;
    size_t rank;
};

struct span_t {
    tensor_t* data;
    size_t count;
};

typedef int (*inference_init_t)(void**);
typedef int (*inference_compute_t)(void*, span_t*, span_t*);
typedef int (*inference_cleanup_t)(void*);

// Test parameters
const double TOLERANCE = 1e-5;
const size_t ROWS = 10;
const size_t COLS = 10;
const size_t TOTAL_ELEMENTS = ROWS * COLS;

// Helper: Check if two floats are approximately equal
bool approx_equal(float a, float b, double tolerance) {
    return std::abs(a - b) < tolerance;
}

// Helper: Generate test data
void generate_test_data(std::vector<float>& data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<float>(i) * 0.1f; // 0.0, 0.1, 0.2, ...
    }
}

// Helper: Validate output against expected
bool validate_output(const std::vector<float>& output,
                    const std::vector<float>& expected,
                    double tolerance) {
    if (output.size() != expected.size()) {
        std::cerr << "Size mismatch: got " << output.size()
                  << ", expected " << expected.size() << std::endl;
        return false;
    }

    size_t mismatches = 0;
    double max_error = 0.0;

    for (size_t i = 0; i < output.size(); i++) {
        double error = std::abs(output[i] - expected[i]);
        if (error > tolerance) {
            if (mismatches < 10) {  // Print first 10 mismatches
                std::cerr << "Mismatch at index " << i << ": got "
                          << output[i] << ", expected " << expected[i]
                          << ", error " << error << std::endl;
            }
            mismatches++;
        }
        if (error > max_error) {
            max_error = error;
        }
    }

    if (mismatches > 0) {
        std::cerr << "Total mismatches: " << mismatches << " / " << output.size() << std::endl;
        std::cerr << "Max error: " << max_error << std::endl;
        return false;
    }

    std::cout << "✓ All " << output.size() << " elements match (max error: "
              << max_error << ")" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Accuracy Validation Test ===" << std::endl << std::endl;

    // Load DLL
    std::cout << "--- Step 1: Loading DLL ---" << std::endl;
    void* dll = LOAD_LIBRARY("test_identity.dll");
    if (!dll) {
        std::cerr << "✗ Failed to load test_identity.dll" << std::endl;
        return 1;
    }
    std::cout << "✓ DLL loaded successfully" << std::endl << std::endl;

    // Resolve functions
    std::cout << "--- Step 2: Resolving Functions ---" << std::endl;
    auto init_func = (inference_init_t)GET_FUNCTION(dll, "inference_init");
    auto compute_func = (inference_compute_t)GET_FUNCTION(dll, "inference_compute");
    auto cleanup_func = (inference_cleanup_t)GET_FUNCTION(dll, "inference_cleanup");

    if (!init_func || !compute_func || !cleanup_func) {
        std::cerr << "✗ Failed to resolve functions" << std::endl;
        FREE_LIBRARY(dll);
        return 1;
    }
    std::cout << "✓ All functions resolved" << std::endl << std::endl;

    // Initialize
    std::cout << "--- Step 3: Initializing Runtime ---" << std::endl;
    void* state = nullptr;
    int result = init_func(&state);
    if (result != 0 || !state) {
        std::cerr << "✗ inference_init failed with code " << result << std::endl;
        FREE_LIBRARY(dll);
        return 1;
    }
    std::cout << "✓ Runtime initialized" << std::endl << std::endl;

    // Prepare test data
    std::cout << "--- Step 4: Preparing Test Data ---" << std::endl;

    // Input: 10x10 matrix with values 0.0, 0.1, 0.2, ..., 9.9
    std::vector<float> input_data(TOTAL_ELEMENTS);
    generate_test_data(input_data, TOTAL_ELEMENTS);

    // Expected output: same as input (identity operation)
    std::vector<float> expected_output = input_data;

    // Output buffer (will be filled by inference)
    std::vector<float> output_data(TOTAL_ELEMENTS, -999.0f); // Initialize with sentinel

    // Create tensor metadata
    int64_t input_shape[] = {static_cast<int64_t>(ROWS), static_cast<int64_t>(COLS)};
    int64_t output_shape[] = {static_cast<int64_t>(ROWS), static_cast<int64_t>(COLS)};

    tensor_t input_tensor;
    input_tensor.data = input_data.data();
    input_tensor.shape = input_shape;
    input_tensor.rank = 2;

    tensor_t output_tensor;
    output_tensor.data = output_data.data();
    output_tensor.shape = output_shape;
    output_tensor.rank = 2;

    span_t inputs;
    inputs.data = &input_tensor;
    inputs.count = 1;

    span_t outputs;
    outputs.data = &output_tensor;
    outputs.count = 1;

    std::cout << "✓ Test data prepared" << std::endl;
    std::cout << "  Input: [" << ROWS << ", " << COLS << "] = " << TOTAL_ELEMENTS << " elements" << std::endl;
    std::cout << "  Sample values: [" << input_data[0] << ", " << input_data[1]
              << ", ..., " << input_data[TOTAL_ELEMENTS-1] << "]" << std::endl << std::endl;

    // Run inference
    std::cout << "--- Step 5: Running Inference ---" << std::endl;
    result = compute_func(state, &inputs, &outputs);
    if (result != 0) {
        std::cerr << "✗ inference_compute failed with code " << result << std::endl;
        cleanup_func(state);
        FREE_LIBRARY(dll);
        return 1;
    }
    std::cout << "✓ Inference completed" << std::endl << std::endl;

    // Validate output
    std::cout << "--- Step 6: Validating Output ---" << std::endl;
    std::cout << "Expected: Identity operation (output = input)" << std::endl;
    std::cout << "Tolerance: " << TOLERANCE << std::endl;

    bool valid = validate_output(output_data, expected_output, TOLERANCE);

    if (!valid) {
        std::cerr << std::endl << "✗ VALIDATION FAILED" << std::endl;
        cleanup_func(state);
        FREE_LIBRARY(dll);
        return 1;
    }

    std::cout << std::endl << "✓ VALIDATION PASSED" << std::endl << std::endl;

    // Cleanup
    std::cout << "--- Step 7: Cleaning Up ---" << std::endl;
    result = cleanup_func(state);
    if (result != 0) {
        std::cerr << "Warning: inference_cleanup returned " << result << std::endl;
    }
    std::cout << "✓ Runtime cleaned up" << std::endl << std::endl;

    FREE_LIBRARY(dll);
    std::cout << "✓ DLL unloaded" << std::endl << std::endl;

    // Summary
    std::cout << "=== ACCURACY TEST PASSED ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Test Summary:" << std::endl;
    std::cout << "  Model: test_identity.dll (identity/no-op operation)" << std::endl;
    std::cout << "  Input shape: [" << ROWS << ", " << COLS << "]" << std::endl;
    std::cout << "  Total elements: " << TOTAL_ELEMENTS << std::endl;
    std::cout << "  Tolerance: " << TOLERANCE << std::endl;
    std::cout << "  Result: All elements match expected values ✓" << std::endl;

    return 0;
}
