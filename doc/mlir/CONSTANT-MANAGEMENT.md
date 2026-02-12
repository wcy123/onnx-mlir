<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Constant Management Design

**Note:** This document has been consolidated into [../CONSTANT-HANDLING-DESIGN.md](../CONSTANT-HANDLING-DESIGN.md). Please refer to that document for the complete and authoritative constant handling design and implementation details.

**Related:** [../MLIR-COMPILATION-OVERVIEW.md](../MLIR-COMPILATION-OVERVIEW.md)

---

## Overview

This document describes how model constants (weights, biases) are handled in the MLIR compilation pipeline:
1. Extraction to globals (compile-time)
2. Upload to GPU (runtime, in inference_init)
3. Retrieval in computation (runtime, in @main)
4. Cleanup (runtime, in inference_cleanup)

---

## Constant Lifecycle

```
Compile Time                     Runtime
┌──────────────────────┐        ┌─────────────────────┐
│ ONNX Model           │        │ DLL Loaded          │
│ - Weights in proto   │        │                     │
└──────────┬───────────┘        │                     │
           │                     │                     │
           ▼                     │                     │
┌──────────────────────┐        │                     │
│ OnnxToHip Pass       │        │                     │
│ - Extract constants  │        │                     │
│ - Create llvm.globals│        │                     │
└──────────┬───────────┘        │                     │
           │                     │                     │
           ▼                     │                     │
┌──────────────────────┐        │                     │
│ llvm.mlir.global     │        │                     │
│ @constant_0          │───────►│ .data section       │
│ @constant_1          │        │ (embedded in DLL)   │
│ ...                  │        │                     │
└──────────────────────┘        └──────┬──────────────┘
                                       │
                                       ▼
                         ┌─────────────────────────────┐
                         │ inference_init()            │
                         ├─────────────────────────────┤
                         │ 1. Get constant addresses   │
                         │    llvm.mlir.addressof      │
                         │                             │
                         │ 2. Allocate GPU memory      │
                         │    hipMalloc()              │
                         │                             │
                         │ 3. Copy to GPU              │
                         │    hipMemcpy()              │
                         │                             │
                         │ 4. Store GPU pointers       │
                         │    context.gpu_constants[i] │
                         └──────┬──────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────────┐
                  │ GPU Memory                       │
                  │ - Constant 0 at ptr[0]          │
                  │ - Constant 1 at ptr[1]          │
                  │ - ...                            │
                  └─────────┬───────────────────────┘
                            │
                            ▼
              ┌────────────────────────────────────┐
              │ inference_compute()                │
              │ calls @main                        │
              ├────────────────────────────────────┤
              │ Get constant from context:         │
              │ %ptr = context.gpu_constants[i]    │
              │ Use in operations                  │
              └────────────────────────────────────┘
                            │
                            ▼
              ┌────────────────────────────────────┐
              │ inference_cleanup()                │
              ├────────────────────────────────────┤
              │ 1. Free GPU memory                 │
              │    hipFree(context.gpu_constants[i])│
              │ 2. Free array                       │
              │    free(context.gpu_constants)      │
              └────────────────────────────────────┘
```

---

## Compile-Time: Constant Extraction

### ONNX Constant Nodes

In ONNX models, constants appear as:
```protobuf
node {
  name: "conv1_weight"
  op_type: "Constant"
  attribute {
    name: "value"
    type: TENSOR
    t { dims: [64, 3, 3, 3]  data_type: FLOAT  raw_data: "..." }
  }
}
```

### Extraction to LLVM Globals

**OnnxToHip pass converts to:**

```mlir
llvm.mlir.global constant @constant_0(
  dense<[[[[1.0, 2.0, ...]]]]> : tensor<64x3x3x3xf32>
) : !llvm.array<1728 x f32>  // 64 × 3 × 3 × 3 = 1728 elements

llvm.mlir.global constant @constant_1(
  dense<[0.5, 0.5, ...]> : tensor<64xf32>
) : !llvm.array<64 x f32>
```

**Why globals?**
- Embedded in DLL `.data` section at compile time
- No parsing overhead at runtime
- Efficient memory access (locality)

---

## Runtime: Upload to GPU (inference_init)

### Helper Functions

**get_constant_count()** - Generated by OnnxToHip
```mlir
llvm.func @get_constant_count() -> i64 {
  %count = llvm.mlir.constant(4 : i64) : i64  // Count known at compile time
  llvm.return %count : i64
}
```

**initialize_constants(state)** - Generated by OnnxToHip (uses runtime functions)
```mlir
llvm.func @initialize_constants(%state: !llvm.ptr) -> i32 {
  // For each constant:
  // 1. Get address of global (embedded in DLL)
  %constant_0_addr = llvm.mlir.addressof @constant_0 : !llvm.ptr

  // 2. Calculate size (elements × element_size)
  %size = llvm.mlir.constant(6912 : i64) : i64  // 1728 × 4 bytes

  // 3. Upload via runtime function (OPAQUE - no GEP on state)
  %index_0 = llvm.mlir.constant(0 : i64) : i64
  llvm.call @hip_upload_constant(%state, %index_0, %constant_0_addr, %size)
    : (!llvm.ptr, i64, !llvm.ptr, i64) -> i32

  // Repeat for all constants...

  %c0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %c0 : i32
}
```

**Runtime implementation (in hip_ep_runtime.cpp)**:
```c
extern "C" int hip_upload_constant(void* state, int64_t index, void* cpu_data, int64_t size) {
  RuntimeState* runtime_state = static_cast<RuntimeState*>(state);

  // Allocate GPU memory
  void* gpu_ptr;
  hipError_t err = hipMalloc(&gpu_ptr, size);
  if (err != hipSuccess) return -1;

  // Copy CPU → GPU
  err = hipMemcpy(gpu_ptr, cpu_data, size, hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    hipFree(gpu_ptr);
    return -1;
  }

  // Store in runtime state (runtime knows struct layout)
  runtime_state->gpu_constants[index] = gpu_ptr;
  return 0;
}
```

**Key points**:
- ✅ Generated code calls `hip_upload_constant` (no GEP on state)
- ✅ Runtime function knows RuntimeState layout and manages storage
- ✅ Clean separation: generated code provides data, runtime manages state
- ✅ ABI stability: runtime can change struct layout without regenerating DLLs

### Called from inference_init

```mlir
llvm.func @inference_init(%out_state: !llvm.ptr<!llvm.ptr>) -> i32 {
  // 1. Allocate context
  %context = llvm.call @malloc(%context_size) : (i64) -> !llvm.ptr

  // 2. Create GPU handles (stream, miopen, hipblas)
  // ...

  // 3. Allocate gpu_constants array
  %count = llvm.call @get_constant_count() : () -> i64
  %ptr_size = llvm.mlir.constant(8 : i64) : i64  // sizeof(void*)
  %array_size = llvm.mul %count, %ptr_size : i64
  %gpu_constants = llvm.call @malloc(%array_size) : (i64) -> !llvm.ptr

  // Store array pointer in context
  %gpu_constants_field = llvm.getelementptr %context[0, 3] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %gpu_constants, %gpu_constants_field : !llvm.ptr

  // 4. Upload constants to GPU
  %ret = llvm.call @initialize_constants(%context) : (!llvm.ptr) -> i32

  // 5. Return context
  llvm.store %context, %out_state : !llvm.ptr
  llvm.return %ret : i32
}
```

**Preconditions for initialize_constants:**
- ✅ Context allocated
- ✅ Handles created (stream, miopen, hipblas)
- ✅ gpu_constants array allocated
- ✅ Array is uninitialized

**Postconditions:**
- ✅ All constants uploaded to GPU
- ✅ gpu_constants[i] contains GPU pointer for constant i
- ✅ Returns 0 on success

---

## Runtime: Retrieval in Computation (@main)

### Helper Function (provided by runtime library)

**CRITICAL**: This function is provided by the **runtime library**, NOT generated code.

```c
// Runtime implementation (in hip_ep_runtime.cpp)
extern "C" void* hip_get_constant(void* state, int64_t index) {
  RuntimeState* runtime_state = static_cast<RuntimeState*>(state);
  return runtime_state->gpu_constants[index];
}
```

**Why runtime-provided?**
- RuntimeState is OPAQUE to generated code
- Only runtime knows the struct layout
- Generated code calls this function (no GEP operations)
- Clean separation: runtime owns state, generated code calls functions

### Usage in @main

```mlir
llvm.func @main(%state: !llvm.ptr, %inputs: !llvm.ptr, %outputs: !llvm.ptr) -> i32 {
  // Get weights for first convolution via runtime function (OPAQUE - no GEP)
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %weights1_gpu = llvm.call @hip_get_constant(%state, %c0) : (!llvm.ptr, i64) -> !llvm.ptr

  // Cast to correct address space
  %weights1_gpu_as1 = llvm.addrspacecast %weights1_gpu : !llvm.ptr to !llvm.ptr<1>

  // Build memref struct for weights
  %weights1_struct = [build memref struct with weights1_gpu_as1 as data pointer]

  // Get bias for first convolution via runtime function (OPAQUE - no GEP)
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %bias1_gpu = llvm.call @hip_get_constant(%state, %c1) : (!llvm.ptr, i64) -> !llvm.ptr
  %bias1_struct = [build memref struct]

  // Get stream via runtime function (OPAQUE - no GEP)
  %stream = llvm.call @runtime_get_stream(%state) : (!llvm.ptr) -> !llvm.ptr

  // Call wrapper with constants
  %ret = llvm.call @hip_conv_wrapper(%stream, %input, %weights1_struct, %bias1_struct, ...)

  // ... more operations

  %c0_i32 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %c0_i32 : i32
}
```

**Key points**:
- ✅ RuntimeState is **opaque** - accessed via `hip_get_constant` and `runtime_get_stream`
- ✅ **No GEP operations** on state pointer
- ✅ Clean separation: runtime provides functions, generated code calls them
- ✅ ABI stability: runtime can change struct layout without breaking generated code

---

## Runtime: Cleanup (inference_cleanup)

### Helper Function (generated code - uses runtime functions)

```mlir
llvm.func @release_constants(%state: !llvm.ptr) -> i32 {
  // Release each constant via runtime function (OPAQUE - no GEP)
  %index_0 = llvm.mlir.constant(0 : i64) : i64
  llvm.call @hip_release_constant(%state, %index_0)
    : (!llvm.ptr, i64) -> i32

  %index_1 = llvm.mlir.constant(1 : i64) : i64
  llvm.call @hip_release_constant(%state, %index_1)
    : (!llvm.ptr, i64) -> i32

  // ... repeat for all constants

  %c0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %c0 : i32
}
```

**Runtime implementation (in hip_ep_runtime.cpp)**:
```c
extern "C" int hip_release_constant(void* state, int64_t index) {
  RuntimeState* runtime_state = static_cast<RuntimeState*>(state);

  // Get GPU pointer from runtime state
  void* gpu_ptr = runtime_state->gpu_constants[index];

  // Free GPU memory
  hipError_t err = hipFree(gpu_ptr);
  if (err != hipSuccess) return -1;

  // Clear pointer in runtime state
  runtime_state->gpu_constants[index] = nullptr;
  return 0;
}
```

**Key points**:
- ✅ Generated code calls `hip_release_constant` (no GEP on state)
- ✅ Runtime function manages GPU memory and updates state
- ✅ Clean abstraction: generated code provides index, runtime handles internals
- ✅ ABI stability: runtime struct layout can change without affecting generated code

### Called from inference_cleanup

**NOTE**: `inference_cleanup` is implemented in the **runtime library**, not generated code.

```c
// Runtime implementation (in hip_ep_runtime.cpp)
extern "C" int inference_cleanup(void* state) {
  RuntimeState* runtime_state = static_cast<RuntimeState*>(state);

  // 1. Free GPU constants (calls generated helper)
  release_constants(state);

  // 2. Destroy GPU handles
  hipStreamDestroy(runtime_state->stream);
  miopenDestroy(runtime_state->miopenHandle);
  hipblasLtDestroy(runtime_state->hipblasHandle);

  // 3. Free gpu_constants array
  delete[] runtime_state->gpu_constants;

  // 4. Free runtime state
  delete runtime_state;

  return 0;
}
```

**Key points**:
- ✅ `inference_cleanup` is runtime-provided (knows RuntimeState layout)
- ✅ Calls generated `release_constants` helper (collaboration)
- ✅ RuntimeState is OPAQUE - only runtime accesses fields directly
- ✅ Clean lifecycle management

---

## Key Design Points

1. **Compile-time extraction**: Weights become llvm.mlir.global (DLL .data section)
2. **Runtime upload**: GPU memory allocated once in inference_init
3. **Fast retrieval**: Simple array lookup in @main
4. **Clean separation**: Constant management isolated in helper functions
5. **No disk I/O**: Everything in memory (DLL embedded in EPContext)

---

## Memory Layout

```
DLL .data section (compile-time)
┌────────────────────────────────┐
│ @constant_0: [1.0, 2.0, ...]   │  ← CPU memory, read-only
│ @constant_1: [0.5, 0.5, ...]   │
│ ...                             │
└────────────────────────────────┘

GPU memory (runtime)
┌────────────────────────────────┐
│ GPU alloc 0: [1.0, 2.0, ...]   │  ← Copied from @constant_0
│ GPU alloc 1: [0.5, 0.5, ...]   │  ← Copied from @constant_1
│ ...                             │
└────────────────────────────────┘

Context struct (runtime)
┌────────────────────────────────┐
│ stream: hipStream_t             │
│ miopenHandle: miopenHandle_t    │
│ hipblasHandle: hipblasLtHandle_t│
│ gpu_constants: ──┐              │
└──────────────────│──────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ void* array          │
        ├──────────────────────┤
        │ [0]: ptr to GPU alloc 0  │  ← Points to weights on GPU
        │ [1]: ptr to GPU alloc 1  │  ← Points to bias on GPU
        │ ...                  │
        └──────────────────────┘
```

---

## Related Documents

- [MODULE-STRUCTURE.md](MODULE-STRUCTURE.md) - Where constant functions fit in module
- [INTERFACE-DESIGN.md](INTERFACE-DESIGN.md) - Constant management function contracts
- [LOWERING-PIPELINE.md](LOWERING-PIPELINE.md) - How constants flow through pipeline
- [../STATE-AND-CONTEXT.md](../STATE-AND-CONTEXT.md) - Context struct layout with gpu_constants
