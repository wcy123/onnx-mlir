/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * DLL Tester: Minimal tool to load and test generated inference DLLs
 *
 * Usage: dll_tester <path-to-dll>
 *
 * Tests:
 * 1. DLL loads successfully
 * 2. All 3 interface functions resolve
 * 3. init/compute/cleanup sequence executes without crashes
 */

#include <cstdint>
#include <iostream>
#include <vector>

#ifdef _WIN32
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

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <dll-path>\n";
    return 1;
  }

  const char *dllPath = argv[1];
  std::cout << "Testing DLL: " << dllPath << "\n";

  // Load DLL
  LIB_HANDLE lib = LOAD_LIB(dllPath);
  if (!lib) {
    std::cerr << "FAIL: Cannot load DLL\n";
#ifndef _WIN32
    std::cerr << "Error: " << dlerror() << "\n";
#endif
    return 1;
  }
  std::cout << "OK: DLL loaded\n";

  // Resolve functions
  auto init = (init_fn)GET_PROC(lib, "inference_init");
  auto compute = (compute_fn)GET_PROC(lib, "inference_compute");
  auto cleanup = (cleanup_fn)GET_PROC(lib, "inference_cleanup");

  if (!init || !compute || !cleanup) {
    std::cerr << "FAIL: Cannot resolve functions\n";
    std::cerr << "  init=" << (void *)init << " compute=" << (void *)compute
              << " cleanup=" << (void *)cleanup << "\n";
    FREE_LIB(lib);
    return 1;
  }
  std::cout << "OK: Functions resolved\n";

  // Test init
  void *state = nullptr;
  int rc = init(&state);
  if (rc != 0 || !state) {
    std::cerr << "FAIL: init returned " << rc << ", state=" << state << "\n";
    FREE_LIB(lib);
    return 1;
  }
  std::cout << "OK: init succeeded, state=" << state << "\n";

  // Test compute with minimal data
  std::vector<float> inputData(16, 1.0f);
  std::vector<float> outputData(16, 0.0f);
  int64_t shape[] = {4, 4};

  tensor_t inputTensor = {inputData.data(), shape, 2};
  tensor_t outputTensor = {outputData.data(), shape, 2};

  span_t inputs = {&inputTensor, 1};
  span_t outputs = {&outputTensor, 1};

  rc = compute(state, &inputs, &outputs);
  if (rc != 0) {
    std::cerr << "FAIL: compute returned " << rc << "\n";
    cleanup(state);
    FREE_LIB(lib);
    return 1;
  }
  std::cout << "OK: compute succeeded\n";

  // Test cleanup
  rc = cleanup(state);
  if (rc != 0) {
    std::cerr << "FAIL: cleanup returned " << rc << "\n";
    FREE_LIB(lib);
    return 1;
  }
  std::cout << "OK: cleanup succeeded\n";

  FREE_LIB(lib);
  std::cout << "PASS: All tests passed\n";
  return 0;
}
