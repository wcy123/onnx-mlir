/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../../lib/Runtime/hipdnn_ep_runtime.h"
#include "../../lib/Runtime/hip_ep_runtime_mock.h"
#include <cassert>
#include <cstdio>
#include <iostream>

int main() {
  std::cout << "=== Runtime State Init/Cleanup Test ===\n\n";

#ifndef BUILD_MOCK_RUNTIME
  std::cerr << "ERROR: This test must be built with BUILD_MOCK_RUNTIME=1\n";
  return 1;
#endif

  // Test 1: Basic initialization and cleanup
  std::cout << "--- Test 1: Basic Init/Cleanup ---\n";
  RuntimeState *state = nullptr;
  int result = hipdnn_ep_state_init(&state);
  assert(result == 0);
  assert(state != nullptr);
  std::cout << "✓ State initialized successfully\n";

  result = hipdnn_ep_state_cleanup(state);
  assert(result == 0);
  std::cout << "✓ State cleaned up successfully\n\n";

  // Test 2: Multiple init/cleanup cycles
  std::cout << "--- Test 2: Multiple Init/Cleanup Cycles ---\n";
  for (int i = 0; i < 3; i++) {
    RuntimeState *state2 = nullptr;
    result = hipdnn_ep_state_init(&state2);
    assert(result == 0);
    assert(state2 != nullptr);

    result = hipdnn_ep_state_cleanup(state2);
    assert(result == 0);
  }
  std::cout << "✓ Multiple cycles succeeded\n\n";

  // Test 3: Null parameter handling
  std::cout << "--- Test 3: Null Parameter Handling ---\n";
  result = hipdnn_ep_state_init(nullptr);
  assert(result == 1); // Should return allocation failed
  std::cout << "✓ Null parameter handled correctly\n";

  result = hipdnn_ep_state_cleanup(nullptr);
  assert(result == 0); // Best-effort cleanup doesn't fail
  std::cout << "✓ Null cleanup handled correctly\n\n";

  std::cout << "=== All Runtime State Tests PASSED ===\n";
  std::cout << "\nExpected output above should show:\n";
  std::cout << "  - hipStreamCreate, miopenCreate, miopenSetStream, "
               "hipblasLtCreate\n";
  std::cout << "  - hipStreamSynchronize, hipblasLtDestroy, miopenDestroy, "
               "hipStreamDestroy\n";
  std::cout << "  - Cleanup in reverse order of initialization (LIFO)\n";

  return 0;
}
