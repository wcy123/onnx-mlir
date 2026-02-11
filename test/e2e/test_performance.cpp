// Performance benchmarking test for test_identity.dll
// Measures latency and throughput of the inference pipeline

#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>

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

// Benchmark configuration
const size_t WARMUP_ITERATIONS = 10;
const size_t BENCHMARK_ITERATIONS = 100;
const size_t ROWS = 10;
const size_t COLS = 10;
const size_t TOTAL_ELEMENTS = ROWS * COLS;

// Helper: Calculate statistics
struct Stats {
    double mean;
    double std_dev;
    double min;
    double max;
    double median;
    double p95;
    double p99;
};

Stats calculate_stats(std::vector<double>& times) {
    Stats stats;

    // Sort for percentiles
    std::sort(times.begin(), times.end());

    // Mean
    stats.mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    // Standard deviation
    double sq_sum = 0.0;
    for (double t : times) {
        sq_sum += (t - stats.mean) * (t - stats.mean);
    }
    stats.std_dev = std::sqrt(sq_sum / times.size());

    // Min/max
    stats.min = times.front();
    stats.max = times.back();

    // Percentiles
    stats.median = times[times.size() / 2];
    stats.p95 = times[static_cast<size_t>(times.size() * 0.95)];
    stats.p99 = static_cast<size_t>(times.size() * 0.99) < times.size() ?
                times[static_cast<size_t>(times.size() * 0.99)] :
                times.back();

    return stats;
}

int main() {
    std::cout << "=== Performance Benchmarking Test ===" << std::endl << std::endl;

    // Load DLL
    std::cout << "--- Configuration ---" << std::endl;
    std::cout << "Warmup iterations: " << WARMUP_ITERATIONS << std::endl;
    std::cout << "Benchmark iterations: " << BENCHMARK_ITERATIONS << std::endl;
    std::cout << "Input shape: [" << ROWS << ", " << COLS << "]" << std::endl;
    std::cout << "Total elements: " << TOTAL_ELEMENTS << std::endl << std::endl;

    std::cout << "--- Loading DLL ---" << std::endl;
    void* dll = LOAD_LIBRARY("test_identity.dll");
    if (!dll) {
        std::cerr << "✗ Failed to load test_identity.dll" << std::endl;
        return 1;
    }
    std::cout << "✓ DLL loaded" << std::endl << std::endl;

    // Resolve functions
    auto init_func = (inference_init_t)GET_FUNCTION(dll, "inference_init");
    auto compute_func = (inference_compute_t)GET_FUNCTION(dll, "inference_compute");
    auto cleanup_func = (inference_cleanup_t)GET_FUNCTION(dll, "inference_cleanup");

    if (!init_func || !compute_func || !cleanup_func) {
        std::cerr << "✗ Failed to resolve functions" << std::endl;
        FREE_LIBRARY(dll);
        return 1;
    }
    std::cout << "✓ Functions resolved" << std::endl << std::endl;

    // Initialize runtime
    std::cout << "--- Initializing Runtime ---" << std::endl;
    void* state = nullptr;
    int result = init_func(&state);
    if (result != 0 || !state) {
        std::cerr << "✗ initialization failed" << std::endl;
        FREE_LIBRARY(dll);
        return 1;
    }
    std::cout << "✓ Runtime initialized" << std::endl << std::endl;

    // Prepare test data
    std::vector<float> input_data(TOTAL_ELEMENTS);
    std::vector<float> output_data(TOTAL_ELEMENTS);

    for (size_t i = 0; i < TOTAL_ELEMENTS; i++) {
        input_data[i] = static_cast<float>(i) * 0.1f;
    }

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

    // Warmup phase
    std::cout << "--- Warmup Phase ---" << std::endl;
    for (size_t i = 0; i < WARMUP_ITERATIONS; i++) {
        result = compute_func(state, &inputs, &outputs);
        if (result != 0) {
            std::cerr << "✗ Warmup iteration " << i << " failed" << std::endl;
            cleanup_func(state);
            FREE_LIBRARY(dll);
            return 1;
        }
    }
    std::cout << "✓ Completed " << WARMUP_ITERATIONS << " warmup iterations" << std::endl << std::endl;

    // Benchmark phase
    std::cout << "--- Benchmark Phase ---" << std::endl;
    std::vector<double> latencies;
    latencies.reserve(BENCHMARK_ITERATIONS);

    auto total_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < BENCHMARK_ITERATIONS; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        result = compute_func(state, &inputs, &outputs);

        auto end = std::chrono::high_resolution_clock::now();

        if (result != 0) {
            std::cerr << "✗ Benchmark iteration " << i << " failed" << std::endl;
            cleanup_func(state);
            FREE_LIBRARY(dll);
            return 1;
        }

        double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(latency_ms);

        // Progress indicator every 20 iterations
        if ((i + 1) % 20 == 0) {
            std::cout << "  Completed " << (i + 1) << " / " << BENCHMARK_ITERATIONS << " iterations" << std::endl;
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time_s = std::chrono::duration<double>(total_end - total_start).count();

    std::cout << "✓ Completed " << BENCHMARK_ITERATIONS << " benchmark iterations" << std::endl << std::endl;

    // Calculate statistics
    Stats stats = calculate_stats(latencies);

    // Display results
    std::cout << "=== Performance Results ===" << std::endl << std::endl;

    std::cout << "Latency Statistics (ms per inference):" << std::endl;
    std::cout << "  Mean:     " << stats.mean << " ms" << std::endl;
    std::cout << "  Std Dev:  " << stats.std_dev << " ms" << std::endl;
    std::cout << "  Min:      " << stats.min << " ms" << std::endl;
    std::cout << "  Median:   " << stats.median << " ms" << std::endl;
    std::cout << "  P95:      " << stats.p95 << " ms" << std::endl;
    std::cout << "  P99:      " << stats.p99 << " ms" << std::endl;
    std::cout << "  Max:      " << stats.max << " ms" << std::endl << std::endl;

    std::cout << "Throughput Statistics:" << std::endl;
    std::cout << "  Total time:        " << total_time_s << " s" << std::endl;
    std::cout << "  Total inferences:  " << BENCHMARK_ITERATIONS << std::endl;
    std::cout << "  Throughput:        " << (BENCHMARK_ITERATIONS / total_time_s) << " inferences/sec" << std::endl;
    std::cout << "  Avg latency:       " << (total_time_s * 1000 / BENCHMARK_ITERATIONS) << " ms/inference" << std::endl << std::endl;

    // Cleanup
    std::cout << "--- Cleanup ---" << std::endl;
    result = cleanup_func(state);
    if (result != 0) {
        std::cerr << "Warning: cleanup returned " << result << std::endl;
    }
    FREE_LIBRARY(dll);
    std::cout << "✓ Cleanup complete" << std::endl << std::endl;

    // Summary
    std::cout << "=== BENCHMARK COMPLETE ===" << std::endl << std::endl;

    std::cout << "Summary:" << std::endl;
    std::cout << "  Model: test_identity.dll" << std::endl;
    std::cout << "  Input: [" << ROWS << ", " << COLS << "] = " << TOTAL_ELEMENTS << " elements" << std::endl;
    std::cout << "  Iterations: " << BENCHMARK_ITERATIONS << " (after " << WARMUP_ITERATIONS << " warmup)" << std::endl;
    std::cout << "  Mean latency: " << stats.mean << " ms" << std::endl;
    std::cout << "  Throughput: " << (BENCHMARK_ITERATIONS / total_time_s) << " inferences/sec" << std::endl << std::endl;

    std::cout << "Note: Performance measured using mock runtime (memory operations only)." << std::endl;
    std::cout << "      Real GPU performance will differ significantly." << std::endl;

    return 0;
}
