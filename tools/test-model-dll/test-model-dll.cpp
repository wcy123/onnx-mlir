/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * test-model-dll: End-to-end test tool for compiled model DLLs
 *
 * Usage: test-model-dll <path-to-dll> [options]
 *
 * Options:
 *   --verbose, -v     : Verbose output (show tensor values and timing)
 *   --validate        : Validate output (check for NaN, inf)
 *   --iterations N    : Run N inference iterations (default: 1)
 *   --help, -h        : Show usage
 *
 * Tests:
 * 1. DLL loads successfully
 * 2. All 3 interface functions resolve (init/compute/cleanup)
 * 3. init/compute/cleanup sequence executes without crashes
 * 4. Optional: Output validation (NaN/Inf checks)
 * 5. Optional: Performance measurement
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef _WIN32
#define NOMINMAX // Prevent Windows.h from defining min/max macros
#include <windows.h>
#define LOAD_LIB(path) LoadLibraryA(path)
#define GET_PROC(h, name) GetProcAddress((HMODULE)(h), name)
#define FREE_LIB(h) FreeLibrary((HMODULE)(h))
#define LIB_HANDLE HMODULE
#else
#include <dlfcn.h>
#define LOAD_LIB(path) dlopen(path, RTLD_NOW)
#define GET_PROC(h, name) dlsym(h, name)
#define FREE_LIB(h) dlclose(h)
#define LIB_HANDLE void *
#endif

// Interface types (must match lib/Runtime/hip_ep_runtime.h)
struct tensor_t {
  void *data;
  int64_t *shape;
  size_t rank;
};

struct span_t {
  tensor_t *data;
  size_t count;
};

typedef int (*init_fn)(void **);
typedef int (*compute_fn)(void *, span_t *, span_t *);
typedef int (*cleanup_fn)(void *);

// Configuration
struct Config {
  const char *dllPath = nullptr;
  bool verbose = false;
  bool validate = false;
  int iterations = 1;
  bool showHelp = false;
};

// Parse command-line arguments
Config parseArgs(int argc, char **argv) {
  Config config;

  if (argc < 2) {
    config.showHelp = true;
    return config;
  }

  // Check for --help first (can be anywhere in arguments)
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      config.showHelp = true;
      return config;
    }
  }

  config.dllPath = argv[1];

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
      config.verbose = true;
    } else if (strcmp(argv[i], "--validate") == 0) {
      config.validate = true;
    } else if (strcmp(argv[i], "--iterations") == 0) {
      if (i + 1 < argc) {
        config.iterations = atoi(argv[++i]);
        if (config.iterations < 1)
          config.iterations = 1;
      }
    } else {
      std::cerr << "Unknown option: " << argv[i] << "\n";
      config.showHelp = true;
    }
  }

  return config;
}

// Show usage
void showUsage(const char *progName) {
  std::cout << "Usage: " << progName << " <dll-path> [options]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --verbose, -v     : Verbose output (show tensor details and "
               "timing)\n";
  std::cout << "  --validate        : Validate output (check for NaN, inf)\n";
  std::cout
      << "  --iterations N    : Run N inference iterations (default: 1)\n";
  std::cout << "  --help, -h        : Show this help\n\n";
  std::cout << "Examples:\n";
  std::cout << "  " << progName << " model.dll\n";
  std::cout << "  " << progName << " model.dll --verbose --validate\n";
  std::cout << "  " << progName << " model.dll --iterations 100\n";
}

// Generate test input data
std::vector<float> generateTestInput(const std::vector<int64_t> &shape) {
  size_t totalElements = 1;
  for (auto dim : shape) {
    totalElements *= dim;
  }

  std::vector<float> data(totalElements);
  // Fill with normalized sequential pattern [0, 1]
  for (size_t i = 0; i < totalElements; i++) {
    data[i] = static_cast<float>(i % 256) / 255.0f;
  }
  return data;
}

// Calculate total elements from shape
size_t getTensorSize(const std::vector<int64_t> &shape) {
  size_t size = 1;
  for (auto dim : shape) {
    size *= dim;
  }
  return size;
}

// Validate output for NaN/Inf
bool validateOutput(const std::vector<float> &output, bool verbose) {
  bool valid = true;
  size_t nanCount = 0;
  size_t infCount = 0;
  float minVal = output[0];
  float maxVal = output[0];

  for (float val : output) {
    if (std::isnan(val)) {
      nanCount++;
      valid = false;
    } else if (std::isinf(val)) {
      infCount++;
      valid = false;
    } else {
      minVal = std::min(minVal, val);
      maxVal = std::max(maxVal, val);
    }
  }

  if (!valid) {
    std::cerr << "ERROR: Invalid output values detected\n";
    if (nanCount > 0)
      std::cerr << "  NaN count: " << nanCount << "\n";
    if (infCount > 0)
      std::cerr << "  Inf count: " << infCount << "\n";
    return false;
  }

  if (verbose) {
    std::cout << "  Output range: [" << minVal << ", " << maxVal << "]\n";
  }

  return true;
}

// Decode error codes
const char *decodeErrorCode(int rc) {
  switch (rc) {
  case 0:
    return "Success";
  case -1:
    return "Index out of bounds";
  case -2:
    return "Rank mismatch";
  case -100:
    return "GPU allocation failed";
  case -101:
    return "GPU memory copy failed";
  case -102:
    return "GPU kernel launch failed";
  default:
    return "Unknown error";
  }
}

// Print tensor info
void printTensorInfo(const char *name, const std::vector<int64_t> &shape,
                     size_t elements) {
  std::cout << name << " tensor: [";
  for (size_t i = 0; i < shape.size(); i++) {
    if (i > 0)
      std::cout << ", ";
    std::cout << shape[i];
  }
  std::cout << "] (" << elements << " elements)\n";
}

int main(int argc, char **argv) {
  Config config = parseArgs(argc, argv);

  if (config.showHelp) {
    showUsage(argv[0]);
    return config.dllPath ? 0 : 1;
  }

  // Print header
  std::cout << "=== Model DLL Test ===\n";
  std::cout << "DLL: " << config.dllPath << "\n";
  if (config.verbose)
    std::cout << "Mode: Verbose";
  if (config.validate)
    std::cout << (config.verbose ? " validation\n" : "Mode: Validation\n");
  if (config.iterations > 1)
    std::cout << "Iterations: " << config.iterations << "\n";
  std::cout << "\n";

  // Load DLL
  std::cout << "--- Loading DLL ---\n";
  LIB_HANDLE lib = LOAD_LIB(config.dllPath);
  if (!lib) {
    std::cerr << "ERROR: Cannot load DLL\n";
#ifndef _WIN32
    std::cerr << "Error: " << dlerror() << "\n";
#else
    std::cerr << "Error code: " << GetLastError() << "\n";
#endif
    return 1;
  }
  std::cout << "\u2713 DLL loaded successfully\n\n";

  // Resolve functions
  std::cout << "--- Resolving Exports ---\n";
  auto init = (init_fn)GET_PROC(lib, "inference_init");
  auto compute = (compute_fn)GET_PROC(lib, "inference_compute");
  auto cleanup = (cleanup_fn)GET_PROC(lib, "inference_cleanup");

  if (!init) {
    std::cerr << "ERROR: Failed to find 'inference_init' export\n";
    std::cerr << "Make sure DLL was compiled with --generate-interface pass\n";
    FREE_LIB(lib);
    return 1;
  }
  std::cout << "\u2713 Found inference_init\n";

  if (!compute) {
    std::cerr << "ERROR: Failed to find 'inference_compute' export\n";
    FREE_LIB(lib);
    return 1;
  }
  std::cout << "\u2713 Found inference_compute\n";

  if (!cleanup) {
    std::cerr << "ERROR: Failed to find 'inference_cleanup' export\n";
    FREE_LIB(lib);
    return 1;
  }
  std::cout << "\u2713 Found inference_cleanup\n\n";

  // Initialize
  std::cout << "--- Running inference_init ---\n";
  void *state = nullptr;
  int rc = init(&state);
  if (rc != 0 || !state) {
    std::cerr << "ERROR: inference_init failed\n";
    std::cerr << "  Return code: " << rc << " (" << decodeErrorCode(rc)
              << ")\n";
    std::cerr << "  State pointer: " << state << "\n";
    FREE_LIB(lib);
    return 1;
  }
  std::cout << "\u2713 State initialized\n";
  if (config.verbose) {
    std::cout << "  State address: " << state << "\n";
  }
  std::cout << "\n";

  // Prepare test data
  std::cout << "--- Preparing Test Data ---\n";

  // Test shapes for demo_two_layer_conv model
  // Input: tensor<1x3x224x224xf32> = 150528 elements
  // Output: tensor<1x64x112x112xf32> = 802816 elements
  std::vector<int64_t> inputShape = {1, 3, 224, 224};
  std::vector<int64_t> outputShape = {1, 64, 112, 112};

  size_t inputElements = getTensorSize(inputShape);
  size_t outputElements = getTensorSize(outputShape);

  if (config.verbose) {
    printTensorInfo("Input", inputShape, inputElements);
    printTensorInfo("Output", outputShape, outputElements);
  }

  std::cout << "Allocating input data (" << inputElements * sizeof(float)
            << " bytes = " << (inputElements * sizeof(float) / 1024)
            << " KB)...\n";
  std::vector<float> inputData;
  try {
    inputData = generateTestInput(inputShape);
  } catch (const std::exception &e) {
    std::cerr << "ERROR: Failed to allocate input data: " << e.what() << "\n";
    cleanup(state);
    FREE_LIB(lib);
    return 1;
  }

  std::cout << "Allocating output data (" << outputElements * sizeof(float)
            << " bytes = " << (outputElements * sizeof(float) / 1024)
            << " KB)...\n";
  std::vector<float> outputData;
  try {
    outputData.resize(outputElements, 0.0f);
  } catch (const std::exception &e) {
    std::cerr << "ERROR: Failed to allocate output data: " << e.what() << "\n";
    cleanup(state);
    FREE_LIB(lib);
    return 1;
  }
  std::cout << "Data allocated successfully\n";

  tensor_t inputTensor = {inputData.data(), inputShape.data(),
                          inputShape.size()};
  tensor_t outputTensor = {outputData.data(), outputShape.data(),
                           outputShape.size()};

  span_t inputs = {&inputTensor, 1};
  span_t outputs = {&outputTensor, 1};

  if (config.verbose && inputElements <= 16) {
    std::cout << "Input values (first 16): ";
    for (size_t i = 0; i < std::min(inputElements, size_t(16)); i++) {
      std::cout << std::fixed << std::setprecision(3) << inputData[i] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // Run inference
  std::cout << "--- Running inference_compute ---\n";

  std::vector<long long> durations;
  bool computeSuccess = true;

  for (int iter = 0; iter < config.iterations; iter++) {
    if (config.iterations > 1 && config.verbose) {
      std::cout << "Iteration " << (iter + 1) << "/" << config.iterations
                << "...\n";
    }

    auto start = std::chrono::high_resolution_clock::now();
    rc = compute(state, &inputs, &outputs);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    durations.push_back(duration.count());

    if (rc != 0) {
      std::cerr << "ERROR: inference_compute returned " << rc << "\n";
      std::cerr << "  Reason: " << decodeErrorCode(rc) << "\n";
      computeSuccess = false;
      break;
    }
  }

  if (!computeSuccess) {
    cleanup(state);
    FREE_LIB(lib);
    return rc;
  }

  std::cout << "\u2713 Compute succeeded\n";

  // Show timing
  if (config.verbose || config.iterations > 1) {
    if (config.iterations == 1) {
      std::cout << "Inference time: " << durations[0] << " \u03bcs\n";
    } else {
      long long total = 0;
      long long minTime = durations[0];
      long long maxTime = durations[0];
      for (auto d : durations) {
        total += d;
        minTime = std::min(minTime, d);
        maxTime = std::max(maxTime, d);
      }
      long long avg = total / durations.size();
      std::cout << "Timing (" << config.iterations << " iterations):\n";
      std::cout << "  Average: " << avg << " \u03bcs\n";
      std::cout << "  Min: " << minTime << " \u03bcs\n";
      std::cout << "  Max: " << maxTime << " \u03bcs\n";
    }
  }
  std::cout << "\n";

  // Validate output
  if (config.validate) {
    std::cout << "--- Validating Output ---\n";
    if (!validateOutput(outputData, config.verbose)) {
      std::cerr << "ERROR: Validation failed\n";
      cleanup(state);
      FREE_LIB(lib);
      return 1;
    }
    std::cout << "\u2713 No NaN/Inf values detected\n";
    std::cout << "\n";
  }

  // Show output sample
  if (config.verbose && outputElements <= 16) {
    std::cout << "Output values (first 16): ";
    for (size_t i = 0; i < std::min(outputElements, size_t(16)); i++) {
      std::cout << std::fixed << std::setprecision(3) << outputData[i] << " ";
    }
    std::cout << "\n\n";
  }

  // Cleanup
  std::cout << "--- Running inference_cleanup ---\n";
  rc = cleanup(state);
  if (rc != 0) {
    std::cerr << "ERROR: inference_cleanup returned " << rc << "\n";
    std::cerr << "  Reason: " << decodeErrorCode(rc) << "\n";
    FREE_LIB(lib);
    return rc;
  }
  std::cout << "\u2713 Cleanup succeeded\n\n";

  FREE_LIB(lib);
  std::cout << "=== Test PASSED ===\n";
  return 0;
}
