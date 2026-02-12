/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hipdnn_ep_runtime.h"

#ifndef BUILD_MOCK_RUNTIME
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <miopen/miopen.h>
#else
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

// Forward declarations for mock functions (defined in hipdnn_ep_runtime_mock.cpp)
extern "C" hipError_t hipStreamCreate(hipStream_t *stream);
extern "C" hipError_t hipStreamDestroy(hipStream_t stream);
extern "C" hipError_t hipStreamSynchronize(hipStream_t stream);
hipError_t hipFree(void *ptr);
extern "C" miopenStatus_t miopenCreate(miopenHandle_t *handle);
extern "C" miopenStatus_t miopenDestroy(miopenHandle_t handle);
extern "C" miopenStatus_t miopenSetStream(miopenHandle_t handle, hipStream_t stream);
extern "C" hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t *handle);
extern "C" hipblasStatus_t hipblasLtDestroy(hipblasLtHandle_t handle);
#endif

#include <cstdio>
#include <cstdlib>

// Internal runtime state structure
struct RuntimeState {
  hipStream_t stream;
  miopenHandle_t miopen_handle;
  hipblasLtHandle_t hipblas_handle;

  // Array of GPU pointers for constants (size known at compile time)
  void **gpu_constants;
  size_t num_constants;
};

// Runtime state management implementation

int hipdnn_ep_state_init(RuntimeState **out_state, size_t num_constants) {
  if (!out_state) {
    fprintf(stderr, "Invalid output parameter to hipdnn_ep_state_init\n");
    return 1;
  }

  // Allocate context struct
  RuntimeState *state = (RuntimeState *)malloc(sizeof(RuntimeState));
  if (!state) {
    fprintf(stderr, "Failed to allocate runtime state\n");
    return 1; // Allocation failed
  }

  // Initialize all fields to null for safe cleanup
  state->stream = nullptr;
  state->miopen_handle = nullptr;
  state->hipblas_handle = nullptr;
  state->gpu_constants = nullptr;
  state->num_constants = num_constants;

  // Allocate constants array (initialized to NULL)
  if (num_constants > 0) {
    state->gpu_constants = (void **)calloc(num_constants, sizeof(void *));
    if (!state->gpu_constants) {
      fprintf(stderr, "Failed to allocate constants array\n");
      free(state);
      return 1; // Allocation failed
    }
  }

  // Create HIP stream
  if (hipStreamCreate(&state->stream) != hipSuccess) {
    fprintf(stderr, "Failed to create HIP stream\n");
    free(state->gpu_constants);
    free(state);
    return 2; // Stream creation failed
  }

  // Create MIOpen handle
  if (miopenCreate(&state->miopen_handle) != miopenStatusSuccess) {
    fprintf(stderr, "Failed to create MIOpen handle\n");
    hipStreamDestroy(state->stream);
    free(state->gpu_constants);
    free(state);
    return 3; // MIOpen creation failed
  }

  // Set stream for MIOpen handle
  if (miopenSetStream(state->miopen_handle, state->stream) !=
      miopenStatusSuccess) {
    fprintf(stderr, "Failed to set MIOpen stream\n");
    miopenDestroy(state->miopen_handle);
    hipStreamDestroy(state->stream);
    free(state->gpu_constants);
    free(state);
    return 4; // Set stream failed
  }

  // Create hipBLASLt handle
  if (hipblasLtCreate(&state->hipblas_handle) != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "Failed to create hipBLASLt handle\n");
    miopenDestroy(state->miopen_handle);
    hipStreamDestroy(state->stream);
    free(state->gpu_constants);
    free(state);
    return 5; // hipBLAS creation failed
  }

  // Success - return initialized state
  *out_state = state;
  return 0;
}

int hipdnn_ep_state_cleanup(RuntimeState *state) {
  if (!state) {
    fprintf(stderr, "Invalid runtime state in cleanup\n");
    return 0; // Best-effort - don't fail
  }

  // Best-effort cleanup - continue even if operations fail
  // Cleanup in reverse order of initialization (LIFO)

  // Synchronize stream to ensure all GPU operations complete
  if (state->stream) {
    hipStreamSynchronize(state->stream);
  }

  // Free all constants (best-effort)
  if (state->gpu_constants) {
    for (size_t i = 0; i < state->num_constants; i++) {
      if (state->gpu_constants[i]) {
        hipFree(state->gpu_constants[i]);
      }
    }
    free(state->gpu_constants);
  }

  // Destroy hipBLASLt handle
  if (state->hipblas_handle) {
    hipblasLtDestroy(state->hipblas_handle);
  }

  // Destroy MIOpen handle
  if (state->miopen_handle) {
    miopenDestroy(state->miopen_handle);
  }

  // Destroy HIP stream
  if (state->stream) {
    hipStreamDestroy(state->stream);
  }

  // Free the context struct itself
  free(state);

  return 0; // Best-effort cleanup always returns success
}

void *hipdnn_ep_get_stream(RuntimeState *state) {
  return state ? static_cast<void *>(state->stream) : nullptr;
}

//==============================================================================
// Backward Compatibility Wrappers (for old test MLIR)
//==============================================================================
// These wrappers support test MLIR that uses old function names.
// Production code should use the hipdnn_ep_* API directly.

extern "C" {

// Legacy wrapper: runtime_state_init -> hipdnn_ep_state_init
int runtime_state_init(void **out_state) {
  // Legacy interface assumes 0 constants (old test MLIR doesn't use constants)
  return hipdnn_ep_state_init(reinterpret_cast<RuntimeState **>(out_state), 0);
}

// Legacy wrapper: runtime_state_cleanup -> hipdnn_ep_state_cleanup
int runtime_state_cleanup(void *state) {
  return hipdnn_ep_state_cleanup(static_cast<RuntimeState *>(state));
}

// Legacy wrapper: runtime_prepare_inference
// Simple implementation: allocates temporary inference data structure
int runtime_prepare_inference(void *state, void *inputs_ptr, void *outputs_ptr,
                               void **out_data) {
  // For now, just return a dummy pointer
  // Real implementation would allocate InferenceData and prepare GPU buffers
  *out_data = malloc(8); // Dummy allocation
  return 0; // Success
}

// Legacy wrapper: runtime_cleanup_inference
// Simple implementation: frees temporary inference data
int runtime_cleanup_inference(void *state, void *data, void *outputs_ptr) {
  // Free the dummy allocation
  if (data) {
    free(data);
  }
  return 0; // Success
}

} // extern "C"
