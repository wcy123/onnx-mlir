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

**initialize_constants(context)** - Generated by OnnxToHip
```mlir
llvm.func @initialize_constants(%context: !llvm.ptr) -> i32 {
  // For each constant:
  // 1. Get address of global
  %constant_0_addr = llvm.mlir.addressof @constant_0 : !llvm.ptr

  // 2. Calculate size (elements × element_size)
  %size = llvm.mlir.constant(6912 : i64) : i64  // 1728 × 4 bytes

  // 3. Allocate GPU memory
  %gpu_ptr_ptr = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr
  llvm.call @hipMalloc(%gpu_ptr_ptr, %size) : (!llvm.ptr, i64) -> i32
  %gpu_ptr = llvm.load %gpu_ptr_ptr : !llvm.ptr

  // 4. Copy CPU → GPU
  %kind = llvm.mlir.constant(1 : i32) : i32  // hipMemcpyHostToDevice
  llvm.call @hipMemcpy(%gpu_ptr, %constant_0_addr, %size, %kind)
    : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32

  // 5. Store GPU pointer in context
  %gpu_constants_ptr = llvm.getelementptr %context[0, 3] : (!llvm.ptr) -> !llvm.ptr
  %gpu_constants = llvm.load %gpu_constants_ptr : !llvm.ptr
  %slot_0 = llvm.getelementptr %gpu_constants[0] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %gpu_ptr, %slot_0 : !llvm.ptr

  // Repeat for all constants...

  %c0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %c0 : i32
}
```

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

### Helper Function (generated by HipToLLVM)

```mlir
llvm.func @hip_get_constant(%context: !llvm.ptr, %index: i64) -> !llvm.ptr {
  // 1. Get gpu_constants array pointer
  %gpu_constants_ptr = llvm.getelementptr %context[0, 3] : (!llvm.ptr) -> !llvm.ptr
  %gpu_constants = llvm.load %gpu_constants_ptr : !llvm.ptr

  // 2. Get GPU pointer for this constant
  %slot = llvm.getelementptr %gpu_constants[%index] : (!llvm.ptr, i64) -> !llvm.ptr
  %gpu_ptr = llvm.load %slot : !llvm.ptr

  llvm.return %gpu_ptr : !llvm.ptr
}
```

### Usage in @main

```mlir
llvm.func @main(%context: !llvm.ptr, %inputs: !llvm.ptr, %outputs: !llvm.ptr) -> i32 {
  // Get weights for first convolution
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %weights1_gpu = llvm.call @hip_get_constant(%context, %c0) : (!llvm.ptr, i64) -> !llvm.ptr

  // Cast to correct address space
  %weights1_gpu_as1 = llvm.addrspacecast %weights1_gpu : !llvm.ptr to !llvm.ptr<1>

  // Build memref struct for weights
  %weights1_struct = [build memref struct with weights1_gpu_as1 as data pointer]

  // Get bias for first convolution
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %bias1_gpu = llvm.call @hip_get_constant(%context, %c1) : (!llvm.ptr, i64) -> !llvm.ptr
  %bias1_struct = [build memref struct]

  // Call wrapper with constants
  %ret = llvm.call @hip_conv_wrapper(%context, %input, %weights1_struct, %bias1_struct, ...)

  // ... more operations

  %c0_i32 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %c0_i32 : i32
}
```

---

## Runtime: Cleanup (inference_cleanup)

### Helper Function

```mlir
llvm.func @release_constants(%context: !llvm.ptr) -> i32 {
  // Get gpu_constants array
  %gpu_constants_ptr = llvm.getelementptr %context[0, 3] : (!llvm.ptr) -> !llvm.ptr
  %gpu_constants = llvm.load %gpu_constants_ptr : !llvm.ptr

  // Free each constant
  %slot_0 = llvm.getelementptr %gpu_constants[0] : (!llvm.ptr) -> !llvm.ptr
  %gpu_ptr_0 = llvm.load %slot_0 : !llvm.ptr
  llvm.call @hipFree(%gpu_ptr_0) : (!llvm.ptr) -> i32

  %slot_1 = llvm.getelementptr %gpu_constants[1] : (!llvm.ptr) -> !llvm.ptr
  %gpu_ptr_1 = llvm.load %slot_1 : !llvm.ptr
  llvm.call @hipFree(%gpu_ptr_1) : (!llvm.ptr) -> i32

  // ... repeat for all constants

  %c0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %c0 : i32
}
```

### Called from inference_cleanup

```mlir
llvm.func @inference_cleanup(%state: !llvm.ptr) -> i32 {
  // 1. Free GPU constants
  llvm.call @release_constants(%state) : (!llvm.ptr) -> i32

  // 2. Destroy GPU handles
  // ...

  // 3. Free gpu_constants array
  %gpu_constants_ptr = llvm.getelementptr %state[0, 3] : (!llvm.ptr) -> !llvm.ptr
  %gpu_constants = llvm.load %gpu_constants_ptr : !llvm.ptr
  llvm.call @free(%gpu_constants) : (!llvm.ptr) -> ()

  // 4. Free context
  llvm.call @free(%state) : (!llvm.ptr) -> ()

  %c0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %c0 : i32
}
```

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
