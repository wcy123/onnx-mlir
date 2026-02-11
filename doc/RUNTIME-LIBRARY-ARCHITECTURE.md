# Runtime Library Architecture Documentation

## Overview

This document explains how the `lib/Runtime` static library works in the ONNX HipDNN Execution Provider, including static linking mechanics and deployment architecture.

## What is lib/Runtime?

The Runtime library is a **static library** that provides GPU state management, constant (model weights) management, and wrapper functions for ROCm operations. It is compiled once as part of the toolchain build and then linked into every generated model DLL.

### Files
- `lib/Runtime/hip_ep_runtime.h` (78 lines) - C-ABI header declaring interface
- `lib/Runtime/hip_ep_runtime.cpp` (343 lines) - Implementation
- `lib/Runtime/CMakeLists.txt` (59 lines) - Build configuration

### Current Functionality (343 lines total)

#### 1. Constant Management (~80 lines)
- `hip_upload_constant()` - Upload model weights from DLL .data section to GPU memory
- `hip_get_constant()` - Retrieve GPU pointer for a constant by index
- `hip_release_constant()` - Free GPU memory for constants
- Uses `std::unordered_map<int64_t, void*>` to track uploaded GPU pointers

#### 2. GPU State Management (~50 lines)
- `RuntimeState` struct contains:
  - HIP stream handle
  - MIOpen handle for convolution operations
  - hipBLASLt handle for matrix operations
  - GPU constant pointers map

#### 3. GPU Operation Wrappers (~150 lines)
- `miopenConvolutionForward()` - Full MIOpen convolution wrapper
  - Creates tensor/convolution descriptors
  - Finds optimal algorithm
  - Allocates workspace memory
  - Performs forward pass
- `hipblasLtGemmWrapper()` - Matrix multiplication wrapper
  - Creates matrix layout descriptors
  - Performs GEMM operation

#### 4. Memory Management Wrappers (~60 lines)
- `hip_malloc_wrapper()` - GPU memory allocation
- `hip_free_wrapper()` - GPU memory deallocation
- `hip_memcpy_h2d_async()` - Host-to-device async copy
- `hip_memcpy_d2h_async()` - Device-to-host async copy
- `hip_stream_synchronize_wrapper()` - Stream synchronization

## How Static Linking Works

### Build Process

#### Step 1: Compile Runtime (One-Time)
```
During toolchain build:

lib/Runtime/hip_ep_runtime.cpp
    ↓ [C++ Compiler]
hip_ep_runtime.o (object file)
    ↓ [Archiver/Librarian]
libHipDnnRuntime.a (static library archive)
```

This creates a static library file containing compiled machine code for all Runtime functions.

#### Step 2: Generate Model Code (Per Model)
```
ONNX Model (.onnx)
    ↓ [MLIR Compiler Pipeline]
MLIR (OnnxDialect → HipDialect)
    ↓ [HipToLLVM + GenerateInterfacePass]
LLVM IR with Runtime function calls:
  - Calls to hip_upload_constant()
  - Calls to hip_get_constant()
  - Calls to miopenConvolutionForward()
  - Calls to hipblasLtGemmWrapper()
  - etc.
    ↓ [LLVM Backend]
model.o (object file with generated code)
```

The generated LLVM IR contains **calls** to Runtime functions but not their implementations.

#### Step 3: Link to Create DLL (Per Model)
```
Linker inputs:
  1. model.o (generated inference code)
  2. libHipDnnRuntime.a (runtime implementations)

Linking process:
  - Linker sees model.o calls hip_upload_constant()
  - Linker extracts hip_upload_constant() from libHipDnnRuntime.a
  - Linker copies the function code into final DLL
  - Repeat for all referenced Runtime functions
  - Linker resolves external symbols (HIP, MIOpen, hipBLASLt)

Output:
  model.dll (single self-contained DLL)
```

### Final DLL Structure

```
model.dll (SINGLE FILE - self-contained)
┌─────────────────────────────────────────────────┐
│ CODE SECTION (.text)                            │
├─────────────────────────────────────────────────┤
│ Generated Functions:                            │
│   - inference_init()                            │
│   - inference_compute()                         │
│   - inference_cleanup()                         │
│   - @main() (computation kernel)                │
│   - initialize_constants()                      │
│   - release_constants()                         │
│                                                 │
│ Runtime Functions (statically linked from .a):  │
│   - hip_upload_constant()                       │
│   - hip_get_constant()                          │
│   - hip_release_constant()                      │
│   - miopenConvolutionForward()                  │
│   - hipblasLtGemmWrapper()                      │
│   - hip_malloc_wrapper()                        │
│   - hip_free_wrapper()                          │
│   - hip_memcpy_h2d_async()                      │
│   - hip_memcpy_d2h_async()                      │
│   - hip_stream_synchronize_wrapper()            │
├─────────────────────────────────────────────────┤
│ DATA SECTION (.data)                            │
├─────────────────────────────────────────────────┤
│   - Model weights (constant arrays)             │
│   - Global variables                            │
└─────────────────────────────────────────────────┘

IMPORT TABLE (external DLL dependencies):
  - amdhip64.dll (AMD HIP Runtime)
      → hipMalloc, hipFree, hipMemcpy, hipStreamCreate, etc.
  - MIOpen.dll (AMD MIOpen Library)
      → miopenCreate, miopenCreateTensorDescriptor, etc.
  - hipblaslt.dll (AMD hipBLASLt Library)
      → hipblasLtCreate, hipblasLtMatmul, etc.
```

## Deployment Architecture

### What You Deploy
**Single DLL per model** (e.g., `resnet50.dll`)
- Contains all inference code
- Contains all Runtime helper functions (statically linked)
- Contains model weights in .data section

### What Must Be Installed on Target System
**AMD ROCm Runtime DLLs** (provided by AMD ROCm installation):
- `amdhip64.dll` - HIP runtime
- `MIOpen.dll` - Convolution operations
- `hipblaslt.dll` - Matrix operations

These are **external dependencies**, not part of your DLL.

### Deployment Example
```
User's System:
├── C:\Program Files\AMD\ROCm\bin\
│   ├── amdhip64.dll (from ROCm install)
│   ├── MIOpen.dll (from ROCm install)
│   └── hipblaslt.dll (from ROCm install)
│
└── C:\MyApp\
    └── resnet50.dll (your generated model)
        ↳ Calls functions in ROCm DLLs
```

## Runtime Execution Flow

### Initialization (`inference_init`)
```
Application calls inference_init()
    ↓
1. Allocate RuntimeState struct
    - Create HIP stream (hipStreamCreate)
    - Create MIOpen handle (miopenCreate)
    - Create hipBLASLt handle (hipblasLtCreate)
    ↓
2. Call initialize_constants()
    - Get addresses of constants in DLL .data section
    - For each constant:
        → Allocate GPU memory (hip_malloc_wrapper → hipMalloc)
        → Upload to GPU (hip_upload_constant → hipMemcpyAsync)
        → Store GPU pointer in map
    ↓
3. Return RuntimeState pointer to application
```

### Inference (`inference_compute`)
```
Application calls inference_compute(state, inputs, outputs)
    ↓
1. Allocate GPU memory for input/output tensors
    - hip_malloc_wrapper() for each tensor
    ↓
2. Copy inputs to GPU
    - hip_memcpy_h2d_async() for each input
    ↓
3. Call @main() computation kernel
    - Retrieves constants via hip_get_constant()
    - Performs convolutions via miopenConvolutionForward()
    - Performs GEMM via hipblasLtGemmWrapper()
    ↓
4. Copy outputs from GPU
    - hip_memcpy_d2h_async() for each output
    ↓
5. Synchronize
    - hip_stream_synchronize_wrapper()
    ↓
6. Free temporary GPU memory
    - hip_free_wrapper() for tensors
```

### Cleanup (`inference_cleanup`)
```
Application calls inference_cleanup(state)
    ↓
1. Call release_constants()
    - For each constant in map:
        → Free GPU memory (hip_free_wrapper → hipFree)
    - Clear constant map
    ↓
2. Destroy GPU handles
    - Destroy hipBLASLt handle
    - Destroy MIOpen handle
    - Destroy HIP stream
    ↓
3. Free RuntimeState struct
```

## Static vs Dynamic Linking Comparison

### Static Linking (Current Approach) ✅
**Build Time:**
- Runtime compiled to libHipDnnRuntime.a
- Linker copies Runtime code INTO model.dll

**Deployment:**
- Single DLL per model
- Runtime code embedded in DLL

**Advantages:**
- ✅ Simple deployment (1 file per model)
- ✅ No version conflicts between models
- ✅ Each model can use different Runtime version
- ✅ Self-contained DLLs

**Disadvantages:**
- Code duplication (each DLL contains Runtime code)
- Slightly larger DLLs (~343 lines × models)

### Dynamic Linking (Alternative) ❌
**Build Time:**
- Runtime compiled to hip_ep_runtime.dll
- Model DLL links to runtime DLL at runtime

**Deployment:**
- Model DLL + runtime DLL (2 files)

**Advantages:**
- Shared Runtime code (smaller total size)
- Can update Runtime without recompiling models

**Disadvantages:**
- ❌ Must deploy 2 files per model
- ❌ Version conflicts if multiple models need different Runtime versions
- ❌ DLL dependency management complexity
- ❌ Runtime DLL must be in PATH or same directory

## Why Static Linking is Industry Standard

**Similar frameworks use static linking:**
- **TensorRT** (NVIDIA): libnvinfer_plugin.a statically linked
- **TVM**: Runtime modules statically linked per model
- **IREE**: HAL runtime statically linked
- **XLA**: Runtime services statically linked

**Reasons:**
1. Deployment simplicity (1 DLL = 1 model)
2. No DLL versioning hell
3. Self-contained artifacts
4. Better for distribution/packaging

## Optimization Opportunity

### Current Runtime Size: 343 lines

**Can be reduced to ~130 lines by:**

1. **Keep (130 lines):**
   - GPU State Management (~50 lines) - REQUIRED
     - RuntimeState struct, handle lifecycle
   - Constant Management (~80 lines) - REQUIRED
     - upload/get/release functions, GPU pointer map

2. **Remove (213 lines) - Generate directly in LLVM IR instead:**
   - Operation wrappers (~150 lines)
     - Generate direct calls to miopenConvolutionForward
     - Generate direct calls to hipblasLtMatmul
     - Inline descriptor creation in generated IR
   - Memory wrappers (~60 lines)
     - Generate direct calls to hipMalloc/hipFree
     - Generate direct calls to hipMemcpyAsync
     - Generate direct calls to hipStreamSynchronize

**Benefits of optimization:**
- Smaller static library (130 vs 343 lines)
- Model-specific optimizations possible
- Direct ROCm API calls (no wrapper overhead)
- More flexible code generation

**What must stay:**
- State management (RuntimeState persistence between calls)
- Constant management (GPU pointer map must persist)

## Dependencies

### Build Dependencies
- CMake 3.18+
- C++17 compiler
- ROCm toolchain (optional with mock mode)

### Runtime Dependencies (External DLLs)
- **HIP Runtime** (`amdhip64.dll`) - GPU compute abstraction
- **MIOpen** (`MIOpen.dll`) - Convolution and neural network operations
- **hipBLASLt** (`hipblaslt.dll`) - BLAS operations for matrix multiplication

### Internal Dependencies
- LLVM/MLIR (for code generation)
- HipDialect (generates Runtime function calls)

## Critical Files

### Runtime Implementation
- `lib/Runtime/hip_ep_runtime.h`
- `lib/Runtime/hip_ep_runtime.cpp`
- `lib/Runtime/CMakeLists.txt`

### Code Generation Integration
- `lib/HipDialect/GenerateInterfacePass.cpp`
  - Generates `inference_init/compute/cleanup` functions
  - Generates calls to Runtime functions

- `lib/HipDialect/HipToLLVM.cpp`
  - Lowers HipDialect ops to LLVM IR
  - Generates Runtime function call IR

### Build System
- `CMakeLists.txt` (line 25)
  - `add_subdirectory(lib/Runtime)` - Builds Runtime first

- `level-1-pass-mlir-compiler/CMakeLists.txt` (line 34)
  - Links `HipDnnRuntime` to MLIR compiler

## Summary

The `lib/Runtime` static library is a **necessary and well-architected component** that:
- Provides essential GPU state and constant management
- Uses industry-standard static linking approach
- Results in single self-contained DLL per model
- Only depends on AMD ROCm DLLs (already installed with ROCm)
- Can be optimized to ~130 lines by generating operations directly

**Answer to original question: "Do we really need lib/Runtime?"**

**Yes**, but it can be minimized:
- GPU state management is REQUIRED (must persist between inference calls)
- Constant management is REQUIRED (must track uploaded GPU pointers)
- Operation wrappers can be ELIMINATED (generate direct ROCm calls instead)

The current approach is sound. The optimization opportunity is to reduce from 343 to ~130 lines by generating operation code directly rather than using wrappers.
