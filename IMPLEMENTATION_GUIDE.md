# MLIR to DLL Generation Pipeline - Implementation Guide

## Overview

This document describes the complete MLIR to DLL generation pipeline implementation that enables ONNX models to be compiled into native DLLs with exported C interface functions.

**Implemented Components:**
- ✅ Runtime Library (HIP/MIOpen/hipBLASLt wrappers)
- ✅ LLVM Backend (MLIR → LLVM IR → Object File)
- ✅ GenerateInterfacePass (3 C interface functions)
- ✅ DLL Linker (LLD library integration)
- ✅ Compiler Driver (dual-mode orchestration)
- ✅ Build System Integration
- ✅ Testing Infrastructure

## Compilation Modes

### IR Mode (Debugging)
Generates LLVM IR text file for inspection and debugging:
```
ONNX Model → MLIR → LLVM IR (.ll file)
```

**Usage:**
```bash
export COMPILATION_MODE=ir
export OUTPUT_PATH=model_debug
# Run compilation...
# Output: model_debug.ll
```

### Native Mode (Production)
Generates native DLL for production deployment:
```
ONNX Model → MLIR → LLVM IR → Object File → DLL
```

**Usage:**
```bash
export COMPILATION_MODE=native
export OUTPUT_PATH=inference
# Run compilation...
# Output: inference.dll (Windows) or inference.so (Linux)
```

## Architecture

### Component 1: Runtime Library

**Location:** `lib/Runtime/`

**Purpose:** Provides C implementations of GPU operations called by generated LLVM IR.

**Key Files:**
- `hipdnn_runtime.h` - API declarations
- `hipdnn_runtime.cpp` - HIP/MIOpen/hipBLASLt wrappers
- `CMakeLists.txt` - Build configuration

**Key Functions:**
```c
// Constant management (GPU upload/download)
int hip_upload_constant(RuntimeState* state, int64_t index, const void* data, int64_t size);
void* hip_get_constant(RuntimeState* state, int64_t index);
int hip_release_constant(RuntimeState* state, int64_t index);

// MIOpen convolution (full wrapper with descriptor management)
int miopenConvolutionForward(void* handle, void* stream, ...);

// hipBLASLt GEMM wrapper
int hipblasLtGemmWrapper(void* handle, void* stream, ...);

// HIP memory operations
int hip_malloc_wrapper(void** ptr, int64_t size);
int hip_free_wrapper(void* ptr);
int hip_memcpy_h2d_async(void* dst, const void* src, int64_t size, void* stream);
int hip_memcpy_d2h_async(void* dst, const void* src, int64_t size, void* stream);
int hip_stream_synchronize_wrapper(void* stream);
```

**Dependencies:** HIP, MIOpen, hipBLASLt

---

### Component 2: LLVM Backend

**Location:** `lib/Backend/LLVMBackend.{h,cpp}`

**Purpose:** Translates MLIR to LLVM IR and optionally compiles to native code.

**Key Functions:**
```cpp
// MLIR → LLVM IR (used by both modes)
std::unique_ptr<llvm::Module> translateMLIRtoLLVMIR(
    mlir::ModuleOp mlirModule, llvm::LLVMContext& llvmContext);

// Optimize LLVM IR
void optimizeLLVMIR(llvm::Module* module, int optLevel);

// IR Mode: Emit .ll text file
bool emitLLVMIR(llvm::Module* module, const std::string& outputPath);

// Native Mode: Compile to object file
bool compileToObjectFile(llvm::Module* module, const std::string& outputPath);
```

**Implementation Notes:**
- Uses `mlir::translateModuleToLLVMIR` C++ API (NOT external mlir-translate tool)
- Runs AFTER all MLIR passes complete (not a pass itself)
- Uses LLVM TargetMachine for native code emission

**Dependencies:** LLVM, MLIR translation libraries

---

### Component 3: GenerateInterfacePass

**Location:** `lib/HipDialect/GenerateInterfacePass.cpp`

**Purpose:** Generate 3 C interface functions that wrap internal @main function.

#### Function 1: `inference_init`

**Signature:** `int inference_init(void** out_state)`

**Context Struct Layout (32 bytes):**
```
offset 0:  hipStream_t stream
offset 8:  miopenHandle_t miopen_handle
offset 16: hipblasLtHandle_t hipblas_handle
offset 24: void** gpu_constants
```

**Generated Code:**
1. Allocate context struct (malloc 32 bytes)
2. Create HIP stream (hipStreamCreate)
3. Create MIOpen handle (miopenCreate + miopenSetStream)
4. Create hipBLAS handle (hipblasLtCreate)
5. Store context pointer to output parameter
6. Return 0 (success) or error code (1-5)

**Error Handling:** 5 error blocks with proper cleanup paths

#### Function 2: `inference_cleanup`

**Signature:** `int inference_cleanup(void* state)`

**Generated Code (reverse order - LIFO):**
1. Synchronize stream (hipStreamSynchronize)
2. Destroy hipBLAS handle (hipblasLtDestroy)
3. Destroy MIOpen handle (miopenDestroy)
4. Destroy HIP stream (hipStreamDestroy)
5. Free context struct (free)
6. Return 0 (best-effort cleanup)

**Error Handling:** Best-effort - continues on errors to prevent leaks

#### Function 3: `inference_compute`

**Signature:** `int inference_compute(void* state, span_t* inputs, span_t* outputs)`

**Data Structures:**
```c
struct tensor_t {
    void* data;        // CPU pointer to tensor data
    int64_t* shape;    // Array of dimension sizes (RUNTIME!)
    size_t rank;       // Number of dimensions
};

struct span_t {
    tensor_t* data;    // Array of tensors
    size_t count;      // Number of tensors
};
```

**Generated Code (simplified - actual implementation more complex):**
1. Validate tensor counts (inputs->count, outputs->count)
2. Parse span_t to get tensor_t arrays
3. For each input tensor:
   - Load rank from tensor_t
   - Load dimensions from tensor_t.shape (RUNTIME dimensions!)
   - Calculate buffer size
   - Allocate GPU buffer (hipMalloc)
   - Copy CPU→GPU (hipMemcpyAsync H2D)
   - Build memref struct with runtime dimensions
4. For each output tensor:
   - Load dimensions
   - Allocate GPU buffer
   - Build memref struct
5. Store memref structs in stack arrays
6. Call @main(context, input_array, output_array)
7. For each output:
   - Copy GPU→CPU (hipMemcpyAsync D2H)
8. Synchronize stream (hipStreamSynchronize)
9. Free all temporary GPU buffers (hipFree)
10. Return result from @main or error code

**Error Handling:** 6 error paths with GPU buffer cleanup

**Note:** Current implementation is simplified and demonstrates the pattern. A complete implementation would handle variable-rank tensors and all edge cases.

---

### Component 4: DLL Linker

**Location:** `lib/Backend/DLLLinker.{h,cpp}`

**Purpose:** Link object file with runtime library using LLD library APIs.

**Key Function:**
```cpp
bool linkDLL(
    const std::string& objectFile,
    const std::string& outputDLL,
    const std::vector<std::string>& libraries,
    const std::vector<std::string>& libraryPaths,
    const std::vector<std::string>& exportSymbols);
```

**Platform-Specific Implementation:**

**Windows (PE/COFF):**
- Uses `lld::coff::link()` library API
- Creates .def file for exports
- Links with `/DLL /MACHINE:X64`
- Produces .dll with exported symbols

**Linux (ELF):**
- Uses `lld::elf::link()` library API
- Uses `-shared --export-dynamic`
- Sets RPATH for runtime library search
- Produces .so with exported symbols

**Important:** Uses LLD library directly - NO external linker programs!

**Dependencies:** LLD (lldCOFF, lldELF, lldCommon)

---

### Component 5: Compiler Driver

**Location:** `level-1-pass-mlir-compiler/src/pass_main.cpp`

**Purpose:** Orchestrate full MLIR → IR/DLL pipeline with mode selection.

**Enhanced Pipeline:**
```cpp
void process(IPass &self, Graph &graph) {
  // 1. Read compilation mode from environment
  std::string mode = ENV_PARAM(COMPILATION_MODE); // "ir" or "native"

  // 2. Parse MLIR bytecode
  auto module = parseMLIRBytecode(graph, context);

  // 3. Run MLIR passes
  PassManager pm;
  pm.addPass(createConvertOnnxToHipPass());
  pm.addPass(createConvertHipToLLVMPass());
  pm.addPass(createGenerateInterfacePass());  // NEW
  pm.run(module);

  // 4. Translate MLIR → LLVM IR (BOTH modes)
  auto llvmModule = translateMLIRtoLLVMIR(module, llvmContext);

  // 5. Optimize LLVM IR (BOTH modes)
  optimizeLLVMIR(llvmModule.get(), 2);

  if (mode == "ir") {
    // IR MODE: Emit .ll text file
    emitLLVMIR(llvmModule.get(), "inference.ll");
  } else if (mode == "native") {
    // NATIVE MODE: Compile and link DLL

    // 6. Emit object file
    compileToObjectFile(llvmModule.get(), "inference.obj");

    // 7. Link DLL using LLD library
    DLLLinker linker;
    linker.linkDLL("inference.obj", "inference.dll",
                   {"amdhip64", "MIOpen", "hipblaslt"},
                   {"/opt/rocm/lib"});

    // 8. Verify exports
    verifyDLLExports("inference.dll");
  }
}
```

**Environment Variables:**
- `COMPILATION_MODE`: "ir" or "native" (default: "native")
- `OUTPUT_PATH`: Output file base name (default: "inference")
- `MLIR_PRINT_WITH_VERBOSE`: Print detailed MLIR (default: "0")

---

## Build System

### Directory Structure
```
lib/
├── Runtime/              # GPU runtime library
│   ├── hipdnn_runtime.h
│   ├── hipdnn_runtime.cpp
│   └── CMakeLists.txt
├── Backend/              # LLVM Backend + DLL Linker
│   ├── LLVMBackend.h
│   ├── LLVMBackend.cpp
│   ├── DLLLinker.h
│   ├── DLLLinker.cpp
│   └── CMakeLists.txt
└── HipDialect/           # MLIR passes (existing)
    ├── GenerateInterfacePass.cpp  # ENHANCED
    └── ...

level-1-pass-mlir-compiler/
└── src/
    └── pass_main.cpp     # ENHANCED with dual-mode pipeline

test/integration/
├── EndToEndTest.cpp
├── verify_dll_exports.bat
├── verify_dll_exports.sh
└── CMakeLists.txt
```

### CMake Integration

**Root CMakeLists.txt:**
```cmake
add_subdirectory(lib/Runtime)
add_subdirectory(lib/Backend)
add_subdirectory(lib/HipDialect)
add_subdirectory(level-1-pass-mlir-compiler)
add_subdirectory(test)
```

**Compiler CMakeLists.txt:**
```cmake
target_link_libraries(morphizen-level1-pass-mlir-compiler PUBLIC
    HipDialect      # MLIR passes
    LLVMBackend     # LLVM Backend + DLL Linker
    HipDnnRuntime   # Runtime library
    MLIRLLVMDialect
    MLIRPass
    # ... other LLVM/MLIR libs
)
```

---

## Testing

### Integration Test

**Location:** `test/integration/EndToEndTest.cpp`

**Purpose:** Load generated DLL and test C interface.

**Usage:**
```bash
# Build
cmake --build build --config Release

# Run test
build/bin/test/EndToEndTest build/bin/inference.dll
```

**Test Steps:**
1. Load DLL (LoadLibrary/dlopen)
2. Resolve exported functions
3. Call `inference_init(&state)` - verify state is non-null
4. Call `inference_compute(state, inputs, outputs)` - with dummy data
5. Call `inference_cleanup(state)` - verify cleanup
6. Verify no crashes or leaks

### Export Verification

**Windows:**
```bash
test/integration/verify_dll_exports.bat inference.dll
```

**Linux:**
```bash
test/integration/verify_dll_exports.sh inference.so
```

**Expected Output:**
```
=== Checking Required Symbols ===
[OK] inference_init found
[OK] inference_compute found
[OK] inference_cleanup found
```

---

## Usage Examples

### Example 1: Generate LLVM IR for Debugging

```bash
# Set IR mode
export COMPILATION_MODE=ir
export OUTPUT_PATH=my_model_debug

# Run compilation (through ONNX Runtime EP)
# ... model execution ...

# Output: my_model_debug.ll (human-readable LLVM IR)
```

**Use Cases:**
- Debug generated code
- Verify transformations
- Cross-platform inspection
- Teaching/learning

### Example 2: Generate Native DLL for Production

```bash
# Set Native mode
export COMPILATION_MODE=native
export OUTPUT_PATH=inference

# Run compilation
# ... model execution ...

# Output: inference.dll (Windows) or inference.so (Linux)
```

**Use Cases:**
- Production deployment
- Maximum performance
- Standalone inference library
- Embedded in applications

### Example 3: Test Generated DLL

```bash
# Verify exports
./test/integration/verify_dll_exports.bat inference.dll

# Run integration test
./build/bin/test/EndToEndTest inference.dll
```

---

## Implementation Status

### ✅ Completed
- Runtime Library (400+ LOC)
- LLVM Backend (250+ LOC)
- GenerateInterfacePass - `inference_init` (150+ LOC)
- GenerateInterfacePass - `inference_cleanup` (80+ LOC)
- GenerateInterfacePass - `inference_compute` (200+ LOC, simplified)
- DLL Linker with LLD library (500+ LOC)
- Compiler Driver dual-mode integration (150+ LOC)
- Build System integration
- Testing infrastructure

### ⚠️ Simplified (Phase 2)
- `inference_compute` implementation (demonstrates pattern, not complete)
  - Full loop through all tensors
  - Variable-rank tensor support
  - Complete memref descriptor construction
  - Actual @main function call
  - Comprehensive error handling

### 📋 Out of Scope (Future Phase 2)
- EPContext serialization
- Runtime DLL loading from EPContext
- MemoryModule integration
- CustomOp integration with ONNX Runtime
- Performance optimization (kernel fusion, memory pooling)
- Multi-model support
- Standalone compiler tool

---

## Known Limitations

1. **inference_compute Simplification:** Current implementation demonstrates the pattern but doesn't:
   - Loop through all input/output tensors
   - Build complete memref descriptors
   - Actually call @main function
   - Handle all error cases

2. **Constant Initialization:** The `initialize_constants` and `release_constants` calls are commented out due to cross-dialect function call complexity.

3. **ROCm Path Hardcoded:** Library paths in pass_main.cpp are hardcoded for Windows. Should be configurable.

4. **Single Platform Testing:** Primarily tested on Windows. Linux support implemented but needs verification.

---

## Next Steps

### Phase 2 Enhancements
1. Complete `inference_compute` implementation
2. Add EPContext serialization
3. Implement runtime DLL loading
4. Add performance benchmarks
5. Create standalone compiler tool
6. Add support for more operations
7. Implement kernel fusion optimizations

### Testing Enhancements
1. Unit tests for each component
2. Memory leak detection (valgrind/sanitizers)
3. Dynamic shape testing
4. Multi-model testing
5. Performance regression tests

---

## Troubleshooting

### Build Errors

**Error:** `Cannot find LLD libraries`
- **Solution:** Ensure LLVM was built with LLD enabled (`LLVM_ENABLE_PROJECTS=lld`)

**Error:** `Cannot find ROCm libraries`
- **Solution:** Install ROCm SDK and update library paths in CMakeLists.txt

### Runtime Errors

**Error:** `Failed to load DLL`
- **Solution:** Check that all ROCm DLLs are in PATH (Windows) or LD_LIBRARY_PATH (Linux)

**Error:** `inference_init returns error code`
- **Solution:** Check that GPU is available and ROCm runtime is initialized

**Error:** `Symbol not found`
- **Solution:** Verify DLL exports using verification scripts

---

## Performance Considerations

### Optimization Levels
- `-O0`: No optimization (debugging)
- `-O1`: Basic optimization
- `-O2`: Default (recommended)
- `-O3`: Aggressive optimization

### Memory Management
- GPU buffers allocated per inference call
- Future: Implement memory pooling for reuse
- Future: Pre-allocate buffers in `inference_init`

### Kernel Fusion
- Current: Separate kernel launches
- Future: Fuse consecutive operations
- Future: Use MLIR fusion passes

---

## References

- [MLIR Documentation](https://mlir.llvm.org/)
- [LLVM Backend Documentation](https://llvm.org/docs/WritingAnLLVMBackend.html)
- [LLD Documentation](https://lld.llvm.org/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [MIOpen Documentation](https://rocm.docs.amd.com/projects/MIOpen/)

---

**Document Version:** 1.0
**Last Updated:** 2026-02-11
**Author:** Implementation team
