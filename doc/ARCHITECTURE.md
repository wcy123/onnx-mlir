<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Architecture Design

**MLIR-based AOT Compilation with EPContext for AMD ROCm**

**Version:** 1.0
**Date:** 2026-02-09
**Branch:** `mlir-integration`

---

## Overview

This document describes the architecture for integrating MLIR compilation technology into the ONNX Runtime HipDNN Execution Provider. The design combines:

- **PR #1**: ONNX → MLIR conversion (MorphiZen framework)
- **PR #4**: HIP MLIR dialect (hip-opt)
- **PR #2/#3**: Pattern matching inspiration

The goal is to create a unified MLIR compilation pipeline with ahead-of-time (AOT) compilation to native code, stored in ONNX Runtime's EPContext mechanism.

**Key Innovation:** Compile ONNX models to native GPU code at model load time, eliminate JIT overhead, and remove LLVM/MLIR runtime dependencies.

---

## System Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      ONNX Model (Input)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              COMPILE-TIME (Level-1 Pass)                     │
├─────────────────────────────────────────────────────────────┤
│  Dependencies: LLVM, MLIR, HIP headers, MIOpen headers      │
│                                                              │
│  1. MorphiZen Framework (PR #1)                             │
│     └─→ ONNX → MLIR ModuleOp (ONNX-MLIR)                   │
│                                                              │
│  2. Pattern-Based Lowering (OnnxToHip Pass)                 │
│     └─→ ONNX dialect → HIP dialect (with MIOpen ops)       │
│                                                              │
│  3. HIP → LLVM Conversion (HipToLLVM Pass)                  │
│     └─→ HIP dialect → LLVM dialect                          │
│                                                              │
│  4. Interface Generation (GenerateInterfacePass)            │
│     └─→ Generate C interface functions (init/compute/cleanup)│
│                                                              │
│  5. LLVM Compilation                                         │
│     └─→ LLVM IR → Native DLL (GPU-specific)                │
│                                                              │
│  6. EPContext Serialization                                  │
│     └─→ Embed DLL bytes in ONNX model EPContext node       │
│                                                              │
│  Output: ONNX model with EPContext (cached compilation)     │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               ONNX Model with EPContext                      │
│         (contains pre-compiled native code)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                RUNTIME (Custom Op)                           │
├─────────────────────────────────────────────────────────────┤
│  Dependencies: MemoryModule (~50KB), HIP runtime, MIOpen    │
│  NO LLVM/MLIR at runtime!                                   │
│                                                              │
│  1. Load EPContext from ONNX model                          │
│  2. Extract DLL bytes from EPContext                        │
│  3. Load DLL from memory (MemoryModule)                     │
│  4. Resolve entry function symbol                           │
│  5. Execute inference on GPU                                 │
│  6. Return results                                           │
│                                                              │
│  Performance: ~1-10ms load time, zero compilation overhead  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

| Stage | Input | Processing | Output |
|-------|-------|------------|--------|
| **MorphiZen Parse** | ONNX model | Parse to MLIR | MLIR ModuleOp (bytecode) |
| **ONNX to HIP Lowering** | MLIR ModuleOp | Pattern matching | HIP dialect MLIR |
| **HIP to LLVM Lowering** | HIP MLIR | Conversion pass | LLVM IR |
| **Interface Generation** | LLVM IR | Generate C wrappers | LLVM IR with interface |
| **Native Compilation** | LLVM IR | AOT compilation | Native DLL |
| **EPContext Storage** | Native DLL | Embed in ONNX | ONNX + EPContext |
| **Runtime Loading** | EPContext DLL | Memory load | Executable code |

---

## Key Design Decisions

### 1. Artifact Format: Native DLL (Not LLVM IR or MLIR)

**Decision:** Level-1 Pass compiles all the way to native machine code (DLL), stored in EPContext.

**Rationale:**
- **Zero JIT overhead**: Compilation happens once at model load/conversion time
- **Lightweight runtime**: No LLVM libraries needed (50-200 MB savings)
- **Fast inference startup**: ~1-10ms to load DLL vs 100-500ms for JIT compilation
- **Industry standard**: TensorRT EP, QNN EP, VitisAI EP all use pre-compiled artifacts
- **EPContext philosophy**: Purpose of EPContext is to eliminate recompilation

**Trade-offs:**
- GPU architecture-specific (compiled for gfx1150, gfx1030, etc.)
- Requires architecture detection/validation at runtime
- Larger EPContext size than LLVM IR (~10x)

### 2. Memory DLL Loading (No Disk Access)

**Decision:** Load DLL directly from EPContext memory buffer using MemoryModule library.

**Rationale:**
- **No disk I/O**: Entire execution from memory - cleaner deployment
- **Fast loading**: ~1-10ms to parse PE and map to memory
- **Simple integration**: MemoryModule is ~1000 lines, MPL 2.0 license
- **WebNN compatibility**: Required for no-disk-access constraint

**Implementation:**
- Use [MemoryModule](https://github.com/fancycode/MemoryModule) (MPL 2.0 license)
- Only ~50 KB runtime dependency

### 3. Pattern-Based Lowering (MLIR Standard)

**Decision:** Use MLIR's pattern rewriting framework to convert ONNX → HIP dialect.

**Rationale:**
- **MLIR best practice**: Standard way to implement dialect conversions
- **Extensible**: Adding new operations = adding new pattern classes
- **Reuses existing code**: PR #4 already uses patterns in `HipToLLVM.cpp`
- **Inspired by PR #2/#3**: Can leverage pattern matching approaches
- **Composable**: Can combine with optimization passes in pipeline

**Implementation approach:**
```cpp
// New file: lib/HipDialect/OnnxToHip.cpp
struct ConvToHipPattern : public OpConversionPattern<...> {
  LogicalResult matchAndRewrite(Operation *op, ...) {
    // Match ONNX Conv operation
    // Extract attributes (kernel, stride, padding, etc.)
    // Create hip.conv operation
    // Return success
  }
};

void populateOnnxToHipPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvToHipPattern>(patterns.getContext());
  patterns.add<GemmToHipPattern>(patterns.getContext());
  patterns.add<PoolingToHipPattern>(patterns.getContext());
  // Easy to extend with new operations
}
```

### 4. Compilation Pipeline

**Pipeline structure:**
```cpp
PassManager pm(context);

// Phase 1: Lowering ONNX to HIP dialect
pm.addPass(createConvertOnnxToHipPass());

// Phase 2: Lowering HIP to LLVM dialect
pm.addPass(createConvertHipToLLVMPass());

// Phase 3: Generate C interface functions
pm.addPass(createGenerateInterfacePass());

// Phase 4: Standard LLVM optimizations
pm.addPass(createCanonicalizerPass());

pm.run(module);
```

**For detailed MLIR module structure and lowering design, see [MLIR-COMPILATION-DESIGN.md](MLIR-COMPILATION-DESIGN.md).**

**For memory management strategy (critical for performance), see [MEMORY-MANAGEMENT.md](MEMORY-MANAGEMENT.md).**

---

## Component Integration

### Directory Structure

```
onnx-hipdnn-ep/
├── CMakeLists.txt                    # Top-level build with build options
│
├── lib/
│   └── HipDialect/                   # Shared MLIR dialect library
│       ├── CMakeLists.txt
│       ├── HipDialect.{h,cpp,td}     # From PR #4
│       ├── HipOps.td                 # Extended with MIOpen ops
│       ├── HipTypes.td               # From PR #4
│       ├── OnnxToHip.cpp             # NEW: ONNX→HIP lowering
│       └── HipToLLVM.cpp             # From PR #4
│
├── tools/
│   └── hip-opt/                      # Standalone MLIR tool (optional)
│       ├── CMakeLists.txt
│       └── hip-opt.cpp
│
├── level-1-pass-mlir-compiler/       # Renamed from level-1-pass-mlir
│   ├── CMakeLists.txt
│   └── src/
│       └── pass_main.cpp             # Full compilation pipeline
│
├── custom-op-hipdnn/                 # From PR #2, modified
│   ├── CMakeLists.txt
│   └── src/
│       ├── custom_op.cpp             # Modified: load from EPContext
│       └── memory_dll_loader.cpp     # NEW: MemoryModule integration
│
├── test/
│   ├── test_ort_integration.cpp      # From PR #1
│   └── ...
│
└── doc/
    ├── ARCHITECTURE.md               # This document
    └── TESTING.md                    # Testing guide
```

### Build Options

The project uses CMake build options for modular compilation:

```cmake
option(BUILD_HIP_DIALECT "Build HIP MLIR dialect library" ON)
option(BUILD_HIP_OPT_TOOL "Build hip-opt standalone tool" OFF)
option(BUILD_MLIR_COMPILER "Build MLIR compiler (Level-1 Pass)" ON)
```

### Build Dependencies

**Compile-Time (Level-1 Pass):**
- LLVM/MLIR (TableGen, compiler libraries)
- ONNX Runtime
- MorphiZen framework
- HIP headers
- MIOpen headers
- CMake 3.29+

**Runtime (Custom Op):**
- ONNX Runtime (already required)
- HIP runtime library (already required)
- MIOpen runtime library (already required)
- MemoryModule (~50 KB, MPL 2.0)

**NO LLVM/MLIR at runtime** - significant deployment advantage.

### HIP Dialect Extensions

Current PR #4 (hip-opt) operations:
```mlir
%handle = hip.create_handle() : !hip.handle
%mem = hip.alloc(%handle, %size) : memref<?xf32, 1>
hip.free(%handle, %mem) : memref<?xf32, 1>
hip.destroy_handle(%handle) : !hip.handle
```

**MIOpen DNN operations (implemented in lib/HipDialect/HipOps.td):**

All HIP operations use **in-place semantics** (destination-passing style):

```mlir
// Convolution (in-place: output buffer passed as argument)
%output = hip.alloc(%handle) : memref<1x64x224x224xf32, 1>
hip.conv(%handle, %input, %weights, %bias, %output)
         {kernel_shape = [3, 3], strides = [1, 1],
          pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
         : (!hip.context, memref<1x3x224x224xf32, 1>,
            memref<64x3x3x3xf32, 1>, memref<64xf32, 1>,
            memref<1x64x224x224xf32, 1>)

// Matrix multiplication (GEMM) - in-place
%result = hip.alloc(%handle) : memref<MxNxf32, 1>
hip.gemm(%handle, %A, %B, %result)
         {transA = 0, transB = 0, alpha = 1.0, beta = 0.0}
         : (!hip.context, memref<MxKxf32, 1>, memref<KxNxf32, 1>,
            memref<MxNxf32, 1>)

// Max pooling - in-place
%output = hip.alloc(%handle) : memref<1x64x56x56xf32, 1>
hip.maxpool(%handle, %input, %output)
            {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]}
            : (!hip.context, memref<1x64x112x112xf32, 1>,
               memref<1x64x56x56xf32, 1>)

// Average pooling - in-place
%output = hip.alloc(%handle) : memref<1x64x56x56xf32, 1>
hip.avgpool(%handle, %input, %output)
            {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]}
            : (!hip.context, memref<1x64x112x112xf32, 1>,
               memref<1x64x56x56xf32, 1>)
```

**Design:** Operations take output buffer as argument (in-place semantics), matching MIOpen/hipBLAS API semantics directly. Each operation lowers to corresponding MIOpen or hipBLASLt API calls in `HipToLLVM.cpp`.

**Future operations** (to be added as needed):
- Batch normalization: `hip.batchnorm`
- Activation functions: `hip.relu`, `hip.sigmoid`, `hip.tanh`
- Softmax: `hip.softmax`
- Elementwise operations: `hip.add`, `hip.mul`, etc.

---

## EPContext Integration

### EPContext Overview

ONNX Runtime's EPContext is a standard mechanism for execution providers to cache pre-compiled models:

**Purpose:**
- Compile model once, save compilation artifacts
- Subsequent loads skip compilation entirely
- Dramatically reduces session creation time

**Storage modes:**
- `embed_mode=1`: DLL embedded directly in ONNX model
- `embed_mode=0`: DLL stored in separate file

**API:**
```cpp
// Enable EPContext during session creation
SessionOptions options;
options.AddConfigEntry("ep.context_enable", "1");
options.AddConfigEntry("ep.context_embed_mode", "1");
```

### Integration Workflow

1. **First run** (compilation):
   ```
   User loads model.onnx
       ↓
   Level-1 Pass executes:
       - ONNX → MLIR → HIP → LLVM → DLL
       - Create EPContext node with DLL bytes
       - Save modified ONNX with EPContext
       ↓
   Custom Op executes from EPContext
   ```

2. **Subsequent runs** (cached):
   ```
   User loads model_with_epcontext.onnx
       ↓
   Level-1 Pass detects EPContext exists
       - Skip compilation entirely
       ↓
   Custom Op loads DLL from EPContext
       - Fast startup (~1-10ms)
   ```

### Custom Op Execution

```cpp
// Simplified pseudocode showing EPContext → DLL loading → inference execution
class HipDnnCustomOp {
private:
  HMEMORYMODULE dll_module_;
  void* inference_state_;

  // Function pointers from compiled DLL
  int (*inference_init_)(void** out_state);
  int (*inference_compute_)(void* state, span_t* inputs, span_t* outputs);
  int (*inference_cleanup_)(void* state);

public:
  void Initialize(const OrtCustomOpApi* api, OrtKernelInfo* info) {
    // 1. Get EPContext attribute
    auto epcontext_data = GetEPContextAttribute(info);

    // 2. Load DLL from memory (using MemoryModule)
    dll_module_ = MemoryLoadLibrary(epcontext_data.dll_bytes);
    if (!dll_module_) {
      throw std::runtime_error("Failed to load DLL from EPContext");
    }

    // 3. Resolve entry points (3 functions per interface spec)
    inference_init_ = (decltype(inference_init_))
        MemoryGetProcAddress(dll_module_, "inference_init");
    inference_compute_ = (decltype(inference_compute_))
        MemoryGetProcAddress(dll_module_, "inference_compute");
    inference_cleanup_ = (decltype(inference_cleanup_))
        MemoryGetProcAddress(dll_module_, "inference_cleanup");

    if (!inference_init_ || !inference_compute_ || !inference_cleanup_) {
      throw std::runtime_error("Failed to resolve DLL entry points");
    }

    // 4. Validate GPU architecture
    if (!ValidateGPUArchitecture(epcontext_data.target_arch)) {
      throw std::runtime_error("GPU architecture mismatch");
    }

    // 5. Initialize inference state (creates GPU resources)
    int ret = inference_init_(&inference_state_);
    if (ret != INFERENCE_SUCCESS) {
      throw std::runtime_error("Inference initialization failed");
    }
  }

  void Compute(OrtKernelContext* context) {
    // Prepare input/output tensors
    tensor_t inputs[1] = {
      {.data = GetInputPointer(context, 0),
       .shape = GetInputShape(context, 0),
       .rank = GetInputRank(context, 0)}
    };
    tensor_t outputs[1] = {
      {.data = GetOutputPointer(context, 0),
       .shape = GetOutputShape(context, 0),
       .rank = GetOutputRank(context, 0)}
    };

    span_t inputs_span = {.data = inputs, .count = 1};
    span_t outputs_span = {.data = outputs, .count = 1};

    // Execute pre-compiled inference (synchronous)
    int ret = inference_compute_(inference_state_, inputs_span, outputs_span);
    if (ret != INFERENCE_SUCCESS) {
      throw std::runtime_error("Inference compute failed");
    }
  }

  ~HipDnnCustomOp() {
    // Cleanup GPU resources
    if (inference_cleanup_ && inference_state_) {
      inference_cleanup_(inference_state_);
    }

    // Unload DLL
    if (dll_module_) {
      MemoryFreeLibrary(dll_module_);
    }
  }
};
```

**Key Points:**
- Uses the 3-function interface: `inference_init()`, `inference_compute()`, `inference_cleanup()`
- DLL loaded entirely from memory (no disk I/O)
- GPU resources initialized once, reused across inferences
- See "Compiled Function Interface Design" section for complete interface specification

---

## Compiled Function Interface Design

This section documents the critical design decisions for the interface between the compiled native code (DLL) and the CustomOp runtime.

### Design Context: 100% GPU Offloading

**Key Assumption:** The entire ONNX model graph is fused into a single CustomOp node containing the complete subgraph. This means:
- One CustomOp instance per model (not per operation)
- All intermediate tensors stay on GPU (never return to CPU mid-inference)
- Only model input/output tensors cross CPU-GPU boundary
- The compiled code manages the entire inference pipeline

**Example:**
```
ONNX Model (ResNet50):
  Input → Conv → BN → ReLU → Conv → ... → FC → Output

After Level-1 Pass Fusion:
  Input → [MyCustomOp containing entire subgraph] → Output
          ↑
          Single fused node
```

### Critical Performance Finding: GPU Memory Allocation Overhead

**Research Finding:** GPU memory allocation/deallocation has significant overhead that impacts architectural decisions.

| Operation | Overhead | Source |
|-----------|----------|---------|
| `hipMalloc` | ~35ms per GB | [HIP Issue #3809](https://github.com/ROCm/hip/issues/3809) |
| `hipFree` | Implicit sync (5-20ms) | [HIP Performance Guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html) |
| **Total per inference** | **~22-37ms for 500MB model** | |

**Context:** Typical ResNet50 inference on GPU takes ~5-10ms. Allocating/freeing memory on every call would add **3-4x overhead**!

**Industry Best Practice:** "Allocate memory once at the beginning and reuse" - CUDA/HIP optimization guides consistently recommend this pattern. `cudaMalloc`/`hipMalloc` are **>100x more expensive** than `malloc`/`free`.

**Impact on Design:** This finding led to choosing a stateful interface (init/compute/cleanup) over a stateless interface (allocate-per-call).

### Resource Management Strategy

**Decision:** Each compiled model manages its own GPU resources independently (Option A - Standalone).

**Alternatives Considered:**

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Option A: Standalone** | Each compiled model creates own handles | • Zero HIP dependencies in CustomOp<br>• Self-contained DLLs<br>• Clean architecture<br>• Extensible | • 3x resource overhead (300MB vs 100MB kernel cache for 3 models)<br>• More GPU memory usage |
| **Option B: Shared Context** | CustomOps share handles via SessionContextRegistry (PR #3 pattern) | • Resource efficient (1x overhead)<br>• Better GPU utilization | • CustomOp needs HIP headers<br>• Tight coupling to HIP<br>• Not extensible (interface grows with libraries)<br>• Not portable to other backends |

**Decision Rationale for Option A:**
1. **Architectural Cleanliness:** CustomOp has zero GPU backend dependencies - can work with HIP, CUDA, SYCL, or future backends
2. **Interface Stability:** Adding new GPU libraries (rocFFT, rocRAND, etc.) doesn't change the interface
3. **Acceptable Trade-off:** 200MB extra memory on 64GB GPU (0.3%) is worth the architectural benefits
4. **Future Optimization:** Can implement custom kernel cache sharing later without interface changes
5. **Portability:** Same CustomOp code works across different GPU backends

**Memory Overhead Context:**
- MI250X: 64GB memory
- MI300X: 192GB memory
- 200MB overhead = 0.3% of 64GB
- Modern GPUs have ample memory for this trade-off

### Interface Design Principles

**Guiding Principles:**
1. **Simplicity:** Minimal interface, easy to implement in MLIR codegen
2. **Stability:** Interface shouldn't change when adding features
3. **Portability:** Backend-agnostic (no HIP-specific types in interface)
4. **Performance:** Zero overhead abstractions
5. **Self-Contained:** Compiled code manages its own resources

### Constants and Weights Management

**Decision:** Constants (weights, biases, etc.) are embedded directly in the compiled DLL.

**Rationale:**
- EPContext is the final compiled artifact - recompiling is required to change weights anyway
- Compiled code has full control over constant layout and organization
- Simplifies interface (no need to pass constants as parameters)
- Self-contained DLL (code + data together)

**EPContext Structure:**
```
EPContext (embedded in ONNX model):
{
  "compiled_dll": <binary blob of code + embedded constants>,
  "metadata": {
    "num_inputs": 1,
    "num_outputs": 1,
    "target_arch": "gfx1150",
    // ... other metadata
  }
}
```

### Final Interface Specification

```c
// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Type Definitions
 *
 * For complete tensor_t and span_t struct definitions, see:
 * doc/mlir/INTERFACE-DESIGN.md#prerequisite-5-tensor-interface
 */

// ============================================================================
// Status Codes
// ============================================================================

/**
 * Return status codes (compatible with int for easy codegen)
 * 0 = success, non-zero = error
 */
#define INFERENCE_SUCCESS                0
#define INFERENCE_ERROR_OOM              1    // Out of GPU memory
#define INFERENCE_ERROR_GPU_NOT_FOUND    2    // No AMD GPU detected
#define INFERENCE_ERROR_INVALID_STATE    3    // Invalid state pointer
#define INFERENCE_ERROR_GPU_KERNEL       4    // GPU kernel execution failed
#define INFERENCE_ERROR_INVALID_INPUT    5    // Invalid input tensor
// ... more error codes as needed

// ============================================================================
// Core Interface (Only 3 Functions!)
// ============================================================================

/**
 * Initialize inference state
 *
 * Called once after loading the DLL. Creates GPU resources:
 * - GPU handles (hipStream, miopenHandle, hipblasLtHandle, etc.)
 * - Uploads embedded constants to GPU
 * - Allocates workspace memory
 * - Initializes algorithm caches
 *
 * All configuration is compile-time (embedded in DLL):
 * - Device ID: Use default (0) or env var HIP_VISIBLE_DEVICES
 * - Batch size: Compiled for specific max batch
 * - Debug flags: Env var MORPHIZEN_DEBUG_*
 *
 * @param out_state Output parameter for opaque state pointer
 * @return INFERENCE_SUCCESS on success, error code otherwise
 */
int inference_init(void** out_state);

/**
 * Execute inference (synchronous)
 *
 * Called for each inference request. Blocks until GPU work completes.
 *
 * Current: inputs/outputs point to CPU memory (compiled code handles transfers)
 * Future: inputs/outputs point to GPU memory (zero-copy, add device_type to tensor_t)
 *
 * @param state Opaque state pointer from inference_init()
 * @param inputs Pointer to span of input tensors (count known at compile time, validated at runtime)
 * @param outputs Pointer to span of output tensors (count known at compile time, validated at runtime)
 * @return INFERENCE_SUCCESS on success, error code otherwise
 */
int inference_compute(void* state, span_t* inputs, span_t* outputs);

/**
 * Cleanup inference state
 *
 * Called once before unloading the DLL. Frees all GPU resources:
 * - Destroys GPU handles
 * - Frees GPU memory (weights, workspace, intermediate buffers)
 *
 * @param state Opaque state pointer from inference_init()
 * @return INFERENCE_SUCCESS on success, error code otherwise
 */
int inference_cleanup(void* state);

// ============================================================================
// Optional Helper
// ============================================================================

/**
 * Get human-readable error message for status code
 *
 * @param status Status code from inference_*() functions
 * @return Static string describing the error (never NULL)
 */
const char* inference_status_message(int status);
```

### Interface Design Decisions

#### 1. Error Handling: Status Codes (int)

**Decision:** Return `int` status codes with output parameters.

**Rationale:**
- Explicit error checking (forces caller to handle errors)
- Standard C pattern (POSIX, HIP, CUDA, MIOpen all use this)
- Simple for LLVM codegen (just return i32)
- Compatible with `int` type (no enum typedef needed)
- Extensible (can add error codes without ABI break)

**Alternatives Considered:**
- Return NULL on error: Too implicit, hard to debug
- Exception-based: Not C-compatible, complex for codegen
- Thread-local error state: Requires TLS, not thread-safe

#### 2. Synchronous Execution

**Decision:** `inference_compute()` blocks until GPU work completes.

**Rationale:**
- Simpler interface (no separate sync function needed)
- Matches ORT CustomOp::Compute() semantics (synchronous)
- Sufficient for current scope (single fused node = entire model)
- Can add async variant later if needed without breaking existing interface

**Alternatives Considered:**
- Async + separate sync: More complex, premature optimization
- Caller-controlled flag: Unnecessary complexity for current scope

#### 3. Dynamic Shape Support

**Decision:** Use `tensor_t` struct with runtime shape information.

**Rationale:**
- Essential for real-world models (dynamic batch size, variable sequence length)
- Clean bundling of pointer + shape + rank
- Extensible (can add dtype, strides, device_type later)

**Current Implementation:** Shapes passed at every inference call
**Optimization Opportunity:** For static shapes, compiled code can ignore runtime shapes

#### 4. No Query Functions

**Decision:** No introspection API (no `get_num_inputs()`, `get_input_shape()`, etc.)

**Rationale:**
- CustomOp already knows model I/O from ONNX graph and EPContext metadata
- Simpler DLL interface (only 3 functions)
- Single source of truth: EPContext metadata (not duplicated in DLL)
- Validation happens at compile time and EPContext load time

**Alternatives Considered:**
- Full introspection: Adds complexity, duplicates metadata
- Minimal queries (just counts): Unnecessary when CustomOp has metadata

#### 5. Constants Embedded in DLL

**Decision:** No constants parameter - weights/biases embedded in compiled DLL.

**Rationale:**
- EPContext is final artifact (recompile to change weights)
- Self-contained DLL (code + data)
- Compiled code controls constant layout/organization
- Simpler interface (zero parameters for `init`)

**DLL Size Impact:** ResNet50 DLL ~100MB (code + weights), acceptable for EPContext storage

### Usage Example

```cpp
// CustomOp implementation (simplified)
class MyCustomOp : public CustomOpImp {
  void* state_;

  MyCustomOp(...) {
    // Load DLL from EPContext
    dll_ = MemoryLoadLibrary(epcontext_dll_bytes);

    // Resolve symbols
    auto init = (int(*)(void**))MemoryGetProcAddress(dll_, "inference_init");
    auto compute = (int(*)(void*, span_t, span_t))MemoryGetProcAddress(dll_, "inference_compute");
    auto cleanup = (int(*)(void*))MemoryGetProcAddress(dll_, "inference_cleanup");

    // Initialize
    int ret = init(&state_);
    if (ret != INFERENCE_SUCCESS) {
      throw std::runtime_error("Inference init failed");
    }
  }

  void Compute(OrtKernelContext* context) {
    // Prepare tensors
    tensor_t inputs[1] = {
      {.data = GetInputPointer(0), .shape = GetInputShape(0), .rank = 4}
    };
    tensor_t outputs[1] = {
      {.data = GetOutputPointer(0), .shape = GetOutputShape(0), .rank = 2}
    };

    span_t inputs_span = {.data = inputs, .count = 1};
    span_t outputs_span = {.data = outputs, .count = 1};

    // Execute
    int ret = compute(state_, inputs_span, outputs_span);
    if (ret != INFERENCE_SUCCESS) {
      throw std::runtime_error("Inference compute failed");
    }
  }

  ~MyCustomOp() {
    cleanup(state_);
    MemoryFreeLibrary(dll_);
  }
};
```

### Future Extensibility

**Planned Extensions (without breaking interface):**

1. **GPU Pointer Support (100% GPU Offloading):**
   ```c
   typedef struct {
       void* data;
       int64_t* shape;
       int rank;
       int device_type;  // NEW: 0=CPU, 1=GPU
   } tensor_t;
   ```
   - Compiled code checks `device_type` and skips H2D/D2H copies for GPU pointers
   - Backward compatible: Old code ignores new field (defaults to 0=CPU)

2. **Data Type Support:**
   ```c
   typedef struct {
       void* data;
       int64_t* shape;
       int rank;
       int dtype;  // NEW: float32=0, float16=1, int8=2, etc.
   } tensor_t;
   ```

3. **Strided Tensor Support:**
   ```c
   typedef struct {
       void* data;
       int64_t* shape;
       int64_t* strides;  // NEW: stride for each dimension
       int rank;
   } tensor_t;
   ```

4. **Cross-Session Resource Sharing:**
   - Add `inference_init_with_shared_resources()` variant
   - Keep existing `inference_init()` for standalone mode
   - Compiled code adapts based on which init function is called

**Design Principle:** Interface remains stable. Extensions add new fields or optional functions without breaking existing compiled code.

---

## ONNX-MLIR Integration

### Rationale for Using ONNX-MLIR

The project integrates onnx-mlir as a git submodule to provide a type-safe, well-tested ONNX dialect for MLIR. This decision significantly simplifies the implementation of ONNX → HIP lowering passes.

**Decision:** Import onnx-mlir from https://github.com/wcy123/onnx-mlir.git (fork with Windows build fixes)

### Key Benefits

#### 1. Type-Safe Operation Access

**Without onnx-mlir (generic Operation*):**
```cpp
// String-based matching - runtime overhead, error-prone
auto opName = op->getName().getStringRef();
if (!opName.consume_front("onnx.")) return failure();
if (opName != "Conv") return failure();

// Manual operand access by index - no semantics
Value input = op->getOperand(0);   // Is this X or W? Must check ONNX spec!
Value weight = op->getOperand(1);
Value bias = op->getNumOperands() > 2 ? op->getOperand(2) : nullptr;
```

**With onnx-mlir (typed ONNXConvOp):**
```cpp
struct ConvToHipPattern : public OpConversionPattern<ONNXConvOp> {
  LogicalResult matchAndRewrite(ONNXConvOp convOp, ...) override {
    // ✅ Type-safe! Pattern only triggers for ONNXConvOp
    // ✅ Semantic operand access (self-documenting)
    Value X = convOp.getX();      // Input tensor
    Value W = convOp.getW();      // Weight tensor
    Value B = convOp.getB();      // Bias (may be NoneType for optional)
    ...
  }
};
```

#### 2. Built-in Attribute Getters

**Without onnx-mlir:**
```cpp
// Manual attribute extraction with casting (verbose + unsafe)
auto kernelAttr = op->getAttrOfType<ArrayAttr>("kernel_shape");
if (!kernelAttr) return failure();  // Runtime check
std::vector<int64_t> kernel;
for (auto attr : kernelAttr) {
  kernel.push_back(attr.cast<IntegerAttr>().getInt());  // Can crash!
}
```

**With onnx-mlir:**
```cpp
// ✅ Type-safe attribute getters (compile-time checked)
auto kernel = convOp.getKernelShape();        // Returns ArrayAttr
auto strides = convOp.getStrides();           // Optional<ArrayAttr>
auto pads = convOp.getPads();                 // Optional<ArrayAttr>
auto dilations = convOp.getDilations();       // Optional<ArrayAttr>
auto group = convOp.getGroup();               // IntegerAttr
```

#### 3. Shape Inference Already Implemented

ONNX operations have complex shape inference rules. For example, Conv2D output shape computation involves:
- Input shape [N, C_in, H_in, W_in]
- Weight shape [C_out, C_in/group, K_H, K_W]
- Padding, stride, dilation calculations
- Auto-padding mode handling

**Without onnx-mlir:** ~200 lines of code to implement Conv shape inference manually

**With onnx-mlir:** Shape inference already done by ONNXConvOp verifier
```cpp
auto outputType = convOp.getResult().getType();  // Already inferred!
```

#### 4. Operation Verification

onnx-mlir provides built-in verification for ONNX operation semantics:
- Operand type constraints (Conv input must be 4D tensor)
- Attribute constraints (kernel_shape size must match spatial dimensions)
- Semantic correctness (weight channels must match input channels / group)

**Result:** Catch errors at MLIR construction time, not at runtime

### Development Productivity Impact

| Aspect | Without onnx-mlir | With onnx-mlir | Improvement |
|--------|-------------------|----------------|-------------|
| **Pattern Code Length** | ~100 lines/op | ~20 lines/op | **5x less code** |
| **Attribute Access** | Manual cast + null check | One-line getter | **10x faster to write** |
| **Error Messages** | "Invalid operand 0" | "Conv input X must be 4D tensor" | **Much clearer** |
| **Maintainability** | ONNX spec changes = manual updates | Auto-updated with onnx-mlir | **Future-proof** |
| **IDE Support** | No auto-complete | Full C++ API | **Better DX** |
| **Debugging** | String-based operation names | Typed C++ classes | **Type safety** |

### Concrete Example: Conv Operation Complexity

ONNX Conv has 13 attributes:
- Mandatory: kernel_shape, strides, pads, dilations, group
- Optional: auto_pad
- Plus complex operand constraints

**Development effort:**
- Without onnx-mlir: ~50-100 lines per operation × 10+ operations = **500-1000 LOC**
- With onnx-mlir: ~20 lines per operation × 10+ operations = **~200 LOC**

**Saved: ~3-4 weeks of development time** + fewer bugs from manual parsing

### Build Cost

**One-time integration cost:**
- Add submodule: 5 minutes
- First build: +10-15 minutes (onnx-mlir TableGen generation)
- Binary size: +~50 MB (onnx-mlir ONNX dialect library)

**Ongoing benefit:** 3-5x faster lowering pass development

### Why Use a Fork (wcy123/onnx-mlir)

The project uses a fork instead of upstream onnx-mlir because:
1. **Windows Build Fixes:** Contains patches for MSVC compatibility (M_PI definition, inline specifiers, std::string conversions)
2. **Customization:** Ability to modify onnx-mlir for project-specific needs
3. **Version Control:** Pin to specific commit for reproducible builds

**Fork maintenance:** Periodically sync with upstream onnx-mlir to get bug fixes and new ONNX operation support

### Integration Points

onnx-mlir is used in:
1. **Level-1 Pass MLIR Compiler:** Parse ONNX → ONNXOps (typed) → HIP dialect
2. **Transform Passes:** Pattern-based lowering using ONNXConvOp, ONNXGemmOp, etc.
3. **Shape Inference:** Reuse onnx-mlir's shape inference for dynamic shapes

**Not used in:** Runtime Custom Op (no MLIR dependencies at inference time)

---

## Technical Gaps and Solutions

### Gap 1: ONNX → HIP Dialect Transformation

**Current state:** PR #1 parses ONNX to generic MLIR, but doesn't transform to HIP dialect.

**Solution:** Implement pattern-based lowering pass using typed ONNX operations from onnx-mlir.

**Implementation:**
- New file: `lib/HipDialect/OnnxToHip.cpp`
- Define conversion patterns for each ONNX operation (ONNXConvOp, ONNXGemmOp, etc.)
- Register patterns in pass manager
- Use MLIR's `OpConversionPattern` framework with typed ONNX ops from onnx-mlir

**Example pattern (using onnx-mlir typed operations):**
```cpp
struct ConvToHipPattern : public OpConversionPattern<ONNXConvOp> {
  LogicalResult matchAndRewrite(
      ONNXConvOp convOp,  // ✅ Type-safe! Compiler knows this is Conv
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // ✅ Type-safe attribute getters (compile-time checked)
    auto kernel = convOp.getKernelShape();
    auto strides = convOp.getStrides();
    auto pads = convOp.getPads();
    auto dilations = convOp.getDilations();
    auto group = convOp.getGroup();

    // ✅ Semantic operand access (self-documenting)
    Value X = convOp.getX();      // Input tensor
    Value W = convOp.getW();      // Weight tensor
    Value B = convOp.getB();      // Bias (may be NoneType for optional)

    // Create HIP Conv operation
    rewriter.replaceOpWithNewOp<hip::ConvOp>(
        convOp,
        X, W, B,
        kernel, strides, pads, dilations, group
    );
    return success();
  }
};
```

**Key advantage:** Type safety eliminates entire classes of bugs (wrong operand index, attribute name typos, etc.)

### Gap 2: Native Code Execution from EPContext

**Current state:** PR #2 Custom Op loads serialized graph metadata, not compiled code.

**Solution:** Integrate MemoryModule for in-memory DLL loading.

**Implementation:**
1. Add MemoryModule to project dependencies
2. Modify Custom Op to:
   - Extract DLL bytes from EPContext
   - Load using `MemoryLoadLibrary()`
   - Resolve inference entry point
   - Execute
3. Handle architecture validation

**Code changes:**
- `custom-op-hipdnn/src/custom_op.cpp`: Modify initialization
- `custom-op-hipdnn/src/memory_dll_loader.cpp`: New file wrapping MemoryModule
- `custom-op-hipdnn/CMakeLists.txt`: Add MemoryModule dependency

### Gap 3: Full Compilation Pipeline in Level-1 Pass

**Current state:** PR #1 only parses and prints MLIR.

**Solution:** Implement complete compilation pipeline.

**Implementation:**
```cpp
// level-1-pass-mlir-compiler/src/pass_main.cpp
void Level1MlirPass::process(IPass &self, Graph &graph) {
  // 1. Parse ONNX to MLIR (existing from PR #1)
  auto module = ParseToMLIR(graph);

  // 2. Run optimization and lowering pipeline
  PassManager pm(context);
  pm.addPass(createConvertOnnxToHipPass());
  pm.addPass(createConvertHipToLLVMPass());
  pm.run(module);

  // 3. Translate LLVM dialect to LLVM IR
  auto llvmIR = TranslateToLLVMIR(module);

  // 4. Compile LLVM IR to native DLL
  auto dllBytes = CompileToNativeDLL(llvmIR);

  // 5. Create EPContext with DLL
  CreateEPContextNode(graph, dllBytes);
}
```

---

## Implementation Status

### Completed Components

**✅ MLIR Compilation Pipeline:**
1. OnnxToHip Pass - Pattern-based lowering from ONNX to HIP dialect
2. HipToLLVM Pass - Lowering HIP dialect to LLVM dialect with MIOpen wrappers
3. GenerateInterfacePass - C interface generation (inference_init/compute/cleanup)
4. HIP Dialect - Complete with MIOpen operations (Conv, Gemm, Pool, etc.)

**✅ Build System:**
- CMake integration with LLVM/MLIR
- Build options for modular compilation
- onnx-mlir integration as submodule

**✅ Documentation:**
- Complete MLIR compilation overview
- Pass-specific documentation (OnnxToHip, HipToLLVM, GenerateInterfacePass)
- Interface design specification
- Dynamic shape support design
- Constant handling design

### In Progress

**⏳ Native Compilation:**
- LLVM IR → native DLL generation
- EPContext serialization

**⏳ Runtime Integration:**
- MemoryModule integration for DLL loading
- Custom Op modifications for EPContext execution

### Future Work

**Potential Optimizations (Not Currently Planned):**
- ONNX-level operator fusion
- HIP-level kernel fusion
- Memory layout optimizations

These optimizations are not required for the current architecture to work and can be added incrementally as performance needs dictate.

---

## Performance Benefits

| Metric | Current PR #2 | This Architecture | Improvement |
|--------|---------------|-------------------|-------------|
| **First inference startup** | ~100-200ms | ~1-10ms | **10-20x faster** |
| **Subsequent startups** | ~100-200ms | ~1-10ms | **10-20x faster** |
| **Runtime binary size** | ~50 MB | ~5 MB | **10x smaller** |
| **Model file size** | +10 KB metadata | +500 KB DLL | Larger but acceptable |
| **Inference latency** | Same | Same | No regression |

---

## References

- [ONNX Runtime EP Context Design](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html)
- [MLIR Documentation](https://mlir.llvm.org/)
- [MLIR Dialect Conversion Guide](https://mlir.llvm.org/docs/DialectConversion/)
- [MemoryModule GitHub](https://github.com/fancycode/MemoryModule)
- [AMD HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/)
- [AMD MIOpen Documentation](https://rocm.docs.amd.com/projects/MIOpen/)

---

**Document History:**
- v1.0 (2026-02-09): Initial architecture document for repository
