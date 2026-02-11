/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_EP_RUNTIME_MOCK_H
#define HIP_EP_RUNTIME_MOCK_H

// Mock function declarations for testing without ROCm
// Only available when BUILD_MOCK_RUNTIME is defined

#ifdef BUILD_MOCK_RUNTIME

#ifdef __cplusplus
extern "C" {
#endif

// Mock handle creation functions
int hipStreamCreate(void **stream);
int hipStreamDestroy(void *stream);
int hipStreamSynchronize(void *stream);

int miopenCreate(void **handle);
int miopenDestroy(void *handle);
int miopenSetStream(void *handle, void *stream);

int hipblasLtCreate(void **handle);
int hipblasLtDestroy(void *handle);

#ifdef __cplusplus
}
#endif

#endif // BUILD_MOCK_RUNTIME

#endif // HIP_EP_RUNTIME_MOCK_H
