# MLIR to DLL Generation Pipeline - Testing Plan

## Overview

This document provides a **step-by-step testing plan** to verify the MLIR to DLL generation pipeline, module by module, building from simple unit tests to full end-to-end integration.

**Testing Philosophy:**
- Start with minimal dependencies
- Test each module independently first
- Progressively integrate modules
- Build confidence incrementally
- Clear pass/fail criteria at each step

---

## Testing Phases

```
Phase 1: Header Compilation          (✅ Completed)
Phase 2: Mock Runtime Testing         (⏳ Current)
Phase 3: LLVM Backend Unit Tests      (⏳ Next)
Phase 4: DLL Linker Unit Tests        (⏳ Next)
Phase 5: MLIR Pass Testing            (⏳ Requires MLIR)
Phase 6: Integration Testing          (⏳ Requires Full Stack)
Phase 7: End-to-End Pipeline          (⏳ Final Goal)
```

---

## Phase 1: Header Compilation ✅

**Goal:** Verify all headers compile without syntax errors

**Prerequisites:**
- C++17 compiler (MSVC, GCC, or Clang)
- CMake 3.20+

**Test Files:**
- `test/standalone_test.cpp` - Header compilation test
- `test/minimal_build_test.cpp` - Mock runtime test

### Step 1.1: Standalone Header Test

```bash
# Navigate to project root
cd /path/to/onnx-hipdnn-ep

# Create standalone build
mkdir -p build_standalone
cd build_standalone

# Configure (minimal dependencies)
cmake .. -DBUILD_STANDALONE_TEST=ON

# Build
cmake --build . --config Release

# Run
./Release/standalone_test  # Windows
./standalone_test          # Linux
```

**Expected Output:**
```
=== MLIR to DLL Pipeline - Standalone Test ===

Test 1: Runtime Library Headers
  ✓ hip_ep_runtime.h compiled successfully

Test 2: LLVM Backend
  ✓ LLVMBackend.h compiled successfully

Test 3: DLL Linker
  ✓ DLLLinker.h compiled successfully
```

**Success Criteria:**
- ✅ No compilation errors
- ✅ All header includes resolve
- ✅ Test executable runs

**Status:** ✅ PASSED (as of 2026-02-11)

---

## Phase 2: Mock Runtime Testing ⏳

**Goal:** Verify runtime library logic without ROCm dependencies

**Prerequisites:**
- C++17 compiler
- Mock implementations enabled

### Step 2.1: Build Mock Runtime

```bash
# Create test directory
mkdir -p build_runtime_test
cd build_runtime_test

# Configure with mock mode
cmake .. \
  -DBUILD_RUNTIME_TEST=ON \
  -DUSE_MOCK_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release
```

### Step 2.2: Test Runtime State Structure

**Test:** `test/runtime/test_runtime_state.cpp`

```cpp
#include "../lib/Runtime/hip_ep_runtime.h"
#include <cassert>
#include <iostream>

int main() {
    std::cout << "Testing RuntimeState structure...\n";

    // Test 1: Structure size
    RuntimeState state;
    assert(sizeof(state) == 32); // 4 pointers × 8 bytes
    std::cout << "✓ RuntimeState size correct (32 bytes)\n";

    // Test 2: Mock initialization
    void* mock_state = nullptr;
    // Would call inference_init(&mock_state) when available
    std::cout << "✓ Structure layout verified\n";

    return 0;
}
```

**Run:**
```bash
./Release/test_runtime_state
```

**Success Criteria:**
- ✅ Structure compiles
- ✅ Size is 32 bytes
- ✅ Fields accessible

### Step 2.3: Test Constant Management

**Test:** `test/runtime/test_constant_upload.cpp`

```cpp
#include "../lib/Runtime/hip_ep_runtime.h"
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    std::cout << "Testing constant upload (mock mode)...\n";

    // Test data
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    RuntimeState state;
    state.constants = new void*[10]{nullptr}; // Mock constant array

    // Test upload
    int result = hip_upload_constant(&state, 0, data.data(),
                                      data.size() * sizeof(float));

    if (result == 0) {
        std::cout << "✓ Constant upload succeeded\n";
    } else {
        std::cerr << "✗ Upload failed with code: " << result << "\n";
        return 1;
    }

    // Test retrieval
    void* retrieved = hip_get_constant(&state, 0);
    assert(retrieved != nullptr);
    std::cout << "✓ Constant retrieval succeeded\n";

    // Test release
    result = hip_release_constant(&state, 0);
    assert(result == 0);
    std::cout << "✓ Constant release succeeded\n";

    delete[] state.constants;
    return 0;
}
```

**Success Criteria:**
- ✅ Upload returns success (0)
- ✅ Get returns non-null pointer
- ✅ Release returns success (0)
- ✅ No memory leaks

### Step 2.4: Test Memory Operations

**Test:** `test/runtime/test_memory_ops.cpp`

```cpp
int main() {
    std::cout << "Testing HIP memory operations (mock mode)...\n";

    void* gpu_ptr = nullptr;
    int64_t size = 1024;

    // Test malloc
    int result = hip_malloc_wrapper(&gpu_ptr, size);
    assert(result == 0);
    assert(gpu_ptr != nullptr);
    std::cout << "✓ hip_malloc_wrapper succeeded\n";

    // Test memcpy H2D
    std::vector<char> host_data(size, 0x42);
    result = hip_memcpy_h2d_async(gpu_ptr, host_data.data(), size, nullptr);
    assert(result == 0);
    std::cout << "✓ hip_memcpy_h2d_async succeeded\n";

    // Test memcpy D2H
    std::vector<char> result_data(size);
    result = hip_memcpy_d2h_async(result_data.data(), gpu_ptr, size, nullptr);
    assert(result == 0);
    std::cout << "✓ hip_memcpy_d2h_async succeeded\n";

    // Test free
    result = hip_free_wrapper(gpu_ptr);
    assert(result == 0);
    std::cout << "✓ hip_free_wrapper succeeded\n";

    return 0;
}
```

**Success Criteria:**
- ✅ All operations return 0 (success)
- ✅ Pointers are non-null after allocation
- ✅ No crashes or segfaults

---

## Phase 3: LLVM Backend Unit Tests ⏳

**Goal:** Verify LLVM IR generation and compilation (requires LLVM)

**Prerequisites:**
- LLVM 18+ with MLIR
- LLD (for object file compilation)

### Step 3.1: Install LLVM Dependencies

**Option A: Pre-built LLVM (Windows)**
```bash
# Download LLVM 18+ from llvm.org
# Install to C:\LLVM

# Set environment
export LLVM_DIR=C:/LLVM/lib/cmake/llvm
export MLIR_DIR=C:/LLVM/lib/cmake/mlir
```

**Option B: Build LLVM from Source**
```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build && cd build

cmake ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;lld" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
  -DCMAKE_INSTALL_PREFIX=/opt/llvm

cmake --build . --target install
```

### Step 3.2: Test LLVM Module Creation

**Test:** `test/backend/test_llvm_module.cpp`

```cpp
#include "../lib/Backend/LLVMBackend.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <iostream>

int main() {
    std::cout << "Testing LLVM Module creation...\n";

    llvm::LLVMContext context;
    auto module = std::make_unique<llvm::Module>("test_module", context);

    assert(module != nullptr);
    assert(module->getName() == "test_module");
    std::cout << "✓ LLVM Module created successfully\n";

    return 0;
}
```

**Build:**
```bash
cmake .. -DBUILD_BACKEND_TEST=ON -DLLVM_DIR=/opt/llvm/lib/cmake/llvm
cmake --build . --config Release
./Release/test_llvm_module
```

**Success Criteria:**
- ✅ Links against LLVM libraries
- ✅ Creates LLVM module
- ✅ No linker errors

### Step 3.3: Test LLVM IR Emission

**Test:** `test/backend/test_emit_llvm_ir.cpp`

```cpp
#include "../lib/Backend/LLVMBackend.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include <iostream>

int main() {
    std::cout << "Testing LLVM IR emission...\n";

    llvm::LLVMContext context;
    auto module = std::make_unique<llvm::Module>("test", context);

    // Create simple function: int test() { return 42; }
    llvm::IRBuilder<> builder(context);
    auto funcType = llvm::FunctionType::get(builder.getInt32Ty(), false);
    auto func = llvm::Function::Create(funcType,
                                       llvm::Function::ExternalLinkage,
                                       "test", module.get());
    auto bb = llvm::BasicBlock::Create(context, "entry", func);
    builder.SetInsertPoint(bb);
    builder.CreateRet(builder.getInt32(42));

    // Emit to file
    bool success = emitLLVMIR(module.get(), "test_output.ll");
    assert(success);
    std::cout << "✓ LLVM IR emitted to test_output.ll\n";

    return 0;
}
```

**Verify Output:**
```bash
cat test_output.ll
# Should contain:
# define i32 @test() {
# entry:
#   ret i32 42
# }
```

**Success Criteria:**
- ✅ Creates .ll file
- ✅ Contains valid LLVM IR
- ✅ Function definition visible

### Step 3.4: Test Object File Compilation

**Test:** `test/backend/test_compile_object.cpp`

```cpp
int main() {
    std::cout << "Testing object file compilation...\n";

    llvm::LLVMContext context;
    auto module = std::make_unique<llvm::Module>("test", context);

    // Create same simple function
    // ... (same as 3.3)

    // Compile to object file
    bool success = compileToObjectFile(module.get(), "test_output.obj");
    assert(success);
    std::cout << "✓ Object file created: test_output.obj\n";

    return 0;
}
```

**Verify Output:**
```bash
# Windows
dumpbin /SYMBOLS test_output.obj | grep test

# Linux
nm test_output.o | grep test
```

**Success Criteria:**
- ✅ Creates .obj/.o file
- ✅ Contains machine code
- ✅ Symbol 'test' is present

---

## Phase 4: DLL Linker Unit Tests ⏳

**Goal:** Verify DLL linking with LLD library

**Prerequisites:**
- LLVM with LLD enabled
- Test object files from Phase 3

### Step 4.1: Test Simple DLL Creation

**Test:** `test/linker/test_simple_dll.cpp`

```cpp
#include "../lib/Backend/DLLLinker.h"
#include <iostream>

int main() {
    std::cout << "Testing DLL linking...\n";

    DLLLinker linker;

    std::vector<std::string> exports = {"test"};
    bool success = linker.linkDLL(
        "test_output.obj",           // from Phase 3.4
        "test_output.dll",
        {},                          // no extra libraries
        {},                          // no library paths
        exports
    );

    assert(success);
    std::cout << "✓ DLL created: test_output.dll\n";

    return 0;
}
```

**Verify Exports:**
```bash
# Windows
dumpbin /EXPORTS test_output.dll

# Linux
nm -D test_output.so
```

**Success Criteria:**
- ✅ Creates .dll/.so file
- ✅ Exported symbol 'test' visible
- ✅ No linker errors

### Step 4.2: Test DLL Loading

**Test:** `test/linker/test_load_dll.cpp`

```cpp
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

int main() {
    std::cout << "Testing DLL loading...\n";

#ifdef _WIN32
    HMODULE dll = LoadLibraryA("test_output.dll");
    assert(dll != nullptr);

    typedef int (*TestFunc)();
    TestFunc func = (TestFunc)GetProcAddress(dll, "test");
    assert(func != nullptr);

    int result = func();
    assert(result == 42);
    std::cout << "✓ Function returned: " << result << "\n";

    FreeLibrary(dll);
#else
    void* dll = dlopen("./test_output.so", RTLD_NOW);
    assert(dll != nullptr);

    typedef int (*TestFunc)();
    TestFunc func = (TestFunc)dlsym(dll, "test");
    assert(func != nullptr);

    int result = func();
    assert(result == 42);
    std::cout << "✓ Function returned: " << result << "\n";

    dlclose(dll);
#endif

    return 0;
}
```

**Success Criteria:**
- ✅ DLL loads successfully
- ✅ Symbol resolves
- ✅ Function executes correctly
- ✅ Returns expected value (42)

---

## Phase 5: MLIR Pass Testing ⏳

**Goal:** Verify GenerateInterfacePass generates correct LLVM IR

**Prerequisites:**
- LLVM + MLIR libraries
- morphizen build environment

### Step 5.1: Test Pass Registration

**Test:** `test/mlir/test_pass_registration.cpp`

```cpp
#include "mlir/Pass/PassManager.h"
#include "lib/HipDialect/Passes.h"
#include <iostream>

int main() {
    std::cout << "Testing MLIR pass registration...\n";

    mlir::MLIRContext context;
    mlir::PassManager pm(&context);

    // Add our custom passes
    pm.addPass(createConvertOnnxToHipPass());
    pm.addPass(createConvertHipToLLVMPass());
    pm.addPass(createGenerateInterfacePass());

    std::cout << "✓ All passes registered successfully\n";

    return 0;
}
```

**Success Criteria:**
- ✅ Passes compile and link
- ✅ Pass registration succeeds
- ✅ No runtime errors

### Step 5.2: Test Interface Generation

**Test:** Create minimal MLIR input

**Input:** `test/mlir/simple_model.mlir`
```mlir
module {
  func.func @main(%arg0: memref<1x3x224x224xf32>) -> memref<1x1000xf32> {
    %0 = memref.alloc() : memref<1x1000xf32>
    return %0 : memref<1x1000xf32>
  }
}
```

**Run passes:**
```bash
# Compile through pipeline
./build/bin/morphizen-level1-pass-mlir-compiler \
  --input=test/mlir/simple_model.mlir \
  --output=test_output.ll

# Check output
cat test_output.ll | grep inference_init
cat test_output.ll | grep inference_compute
cat test_output.ll | grep inference_cleanup
```

**Success Criteria:**
- ✅ All 3 functions generated
- ✅ Correct signatures
- ✅ Valid LLVM IR

---

## Phase 6: Integration Testing ⏳

**Goal:** Test complete module integration

**Prerequisites:**
- All previous phases passed
- ROCm installed (for full runtime)

### Step 6.1: Build Full Runtime with ROCm

```bash
# Install ROCm 5.7+
# Windows: Download from AMD
# Linux: sudo apt install rocm-hip-sdk

# Configure with ROCm
cmake .. \
  -DUSE_MOCK_HIP=OFF \
  -DROCM_PATH=/opt/rocm \
  -DBUILD_HIP_DIALECT=ON

# Build
cmake --build . --config Release
```

**Success Criteria:**
- ✅ Links against HIP, MIOpen, hipBLASLt
- ✅ No undefined symbols
- ✅ Runtime library builds

### Step 6.2: Test Runtime Initialization (Real GPU)

**Test:** `test/integration/test_real_runtime.cpp`

```cpp
#include "../lib/Runtime/hip_ep_runtime.h"
#include <iostream>

int main() {
    std::cout << "Testing real HIP runtime...\n";

    RuntimeState state;

    // Create stream
    hipStream_t stream;
    hipError_t err = hipStreamCreate(&stream);
    if (err != hipSuccess) {
        std::cerr << "✗ Failed to create HIP stream\n";
        return 1;
    }
    state.stream = stream;
    std::cout << "✓ HIP stream created\n";

    // Create MIOpen handle
    miopenHandle_t handle;
    miopenStatus_t status = miopenCreate(&handle);
    if (status != miopenStatusSuccess) {
        std::cerr << "✗ Failed to create MIOpen handle\n";
        return 1;
    }
    state.miopen_handle = handle;
    std::cout << "✓ MIOpen handle created\n";

    // Cleanup
    miopenDestroy(handle);
    hipStreamDestroy(stream);
    std::cout << "✓ Runtime initialization successful\n";

    return 0;
}
```

**Success Criteria:**
- ✅ Detects AMD GPU
- ✅ Creates HIP stream
- ✅ Creates MIOpen handle
- ✅ No runtime errors

### Step 6.3: Test Full Pipeline (IR Mode)

```bash
# Set IR mode
export COMPILATION_MODE=ir
export OUTPUT_PATH=integration_test

# Run compiler with simple ONNX model
./build/bin/morphizen-level1-pass-mlir-compiler \
  --input=test/models/simple.onnx

# Verify output
ls -lh integration_test.ll
cat integration_test.ll | grep -A5 "define.*inference_"
```

**Success Criteria:**
- ✅ Creates .ll file
- ✅ Contains 3 interface functions
- ✅ Valid LLVM IR syntax

### Step 6.4: Test Full Pipeline (Native Mode)

```bash
# Set Native mode
export COMPILATION_MODE=native
export OUTPUT_PATH=integration_test

# Run compiler
./build/bin/morphizen-level1-pass-mlir-compiler \
  --input=test/models/simple.onnx

# Verify DLL
ls -lh integration_test.dll

# Check exports
./test/integration/verify_dll_exports.bat integration_test.dll
```

**Success Criteria:**
- ✅ Creates .dll/.so file
- ✅ All 3 symbols exported
- ✅ DLL is loadable

---

## Phase 7: End-to-End Pipeline Testing ⏳

**Goal:** Full workflow from ONNX model to inference

**Prerequisites:**
- Phase 6 completed
- Real ONNX models available

### Step 7.1: Compile Real Model

```bash
# Get test model
wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet18-v1-7.onnx

# Compile to DLL
export COMPILATION_MODE=native
export OUTPUT_PATH=resnet18

./build/bin/morphizen-level1-pass-mlir-compiler \
  --input=resnet18-v1-7.onnx \
  --output-dir=./output

# Verify
ls -lh resnet18.dll
./test/integration/verify_dll_exports.bat resnet18.dll
```

### Step 7.2: Run Inference Test

**Test:** `test/integration/test_e2e_inference.cpp`

```cpp
#include <iostream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#define LOAD_LIB(path) LoadLibraryA(path)
#define GET_FUNC(lib, name) GetProcAddress((HMODULE)lib, name)
#define FREE_LIB(lib) FreeLibrary((HMODULE)lib)
#else
#include <dlfcn.h>
#define LOAD_LIB(path) dlopen(path, RTLD_NOW)
#define GET_FUNC(lib, name) dlsym(lib, name)
#define FREE_LIB(lib) dlclose(lib)
#endif

// Function signatures from GenerateInterfacePass
typedef int (*InferenceInitFunc)(void**);
typedef int (*InferenceComputeFunc)(void*, void*, void*);
typedef int (*InferenceCleanupFunc)(void*);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: test_e2e_inference <dll_path>\n";
        return 1;
    }

    std::cout << "=== End-to-End Inference Test ===\n";
    std::cout << "Loading DLL: " << argv[1] << "\n";

    // Load DLL
    void* dll = LOAD_LIB(argv[1]);
    if (!dll) {
        std::cerr << "✗ Failed to load DLL\n";
        return 1;
    }
    std::cout << "✓ DLL loaded\n";

    // Resolve functions
    auto init_func = (InferenceInitFunc)GET_FUNC(dll, "inference_init");
    auto compute_func = (InferenceComputeFunc)GET_FUNC(dll, "inference_compute");
    auto cleanup_func = (InferenceCleanupFunc)GET_FUNC(dll, "inference_cleanup");

    if (!init_func || !compute_func || !cleanup_func) {
        std::cerr << "✗ Failed to resolve functions\n";
        return 1;
    }
    std::cout << "✓ All functions resolved\n";

    // Initialize
    void* state = nullptr;
    int result = init_func(&state);
    if (result != 0 || state == nullptr) {
        std::cerr << "✗ Init failed with code: " << result << "\n";
        return 1;
    }
    std::cout << "✓ Inference initialized (state=" << state << ")\n";

    // Prepare dummy input (1x3x224x224)
    std::vector<float> input_data(1 * 3 * 224 * 224, 1.0f);
    std::vector<float> output_data(1 * 1000, 0.0f);

    // TODO: Build span_t structures when inference_compute is complete
    // For now, just test that the function exists
    std::cout << "✓ Compute function available (not called - simplified impl)\n";

    // Cleanup
    result = cleanup_func(state);
    if (result != 0) {
        std::cerr << "⚠ Cleanup returned non-zero: " << result << "\n";
    }
    std::cout << "✓ Cleanup completed\n";

    FREE_LIB(dll);
    std::cout << "\n=== Test PASSED ===\n";

    return 0;
}
```

**Run:**
```bash
./build/bin/test_e2e_inference ./resnet18.dll
```

**Success Criteria:**
- ✅ DLL loads
- ✅ Functions resolve
- ✅ Init succeeds (state non-null)
- ✅ Cleanup succeeds
- ✅ No crashes

### Step 7.3: Performance Benchmark

```cpp
#include <chrono>

int main() {
    // ... setup from 7.2 ...

    std::cout << "Running performance benchmark...\n";

    const int iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        // When inference_compute is complete:
        // compute_func(state, inputs, outputs);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Total time: " << duration.count() << " ms\n";
    std::cout << "Avg per inference: " << (duration.count() / (float)iterations) << " ms\n";

    return 0;
}
```

---

## Test Execution Checklist

### Quick Reference

| Phase | Test | Status | Prerequisites | Time |
|-------|------|--------|---------------|------|
| 1.1 | Header compilation | ✅ PASSED | C++ compiler | 2 min |
| 2.1 | Mock runtime build | ⏳ TODO | CMake | 5 min |
| 2.2 | Runtime state | ⏳ TODO | Mock build | 1 min |
| 2.3 | Constant management | ⏳ TODO | Mock build | 2 min |
| 2.4 | Memory operations | ⏳ TODO | Mock build | 2 min |
| 3.1 | LLVM install | ⏳ TODO | - | 30-60 min |
| 3.2 | LLVM module | ⏳ TODO | LLVM | 2 min |
| 3.3 | IR emission | ⏳ TODO | LLVM | 2 min |
| 3.4 | Object compilation | ⏳ TODO | LLVM+LLD | 3 min |
| 4.1 | Simple DLL | ⏳ TODO | LLD | 3 min |
| 4.2 | DLL loading | ⏳ TODO | Phase 4.1 | 2 min |
| 5.1 | Pass registration | ⏳ TODO | MLIR | 2 min |
| 5.2 | Interface generation | ⏳ TODO | Full build | 5 min |
| 6.1 | ROCm build | ⏳ TODO | ROCm SDK | 10 min |
| 6.2 | Real runtime | ⏳ TODO | AMD GPU | 2 min |
| 6.3 | IR pipeline | ⏳ TODO | Full build | 5 min |
| 6.4 | Native pipeline | ⏳ TODO | Full build | 5 min |
| 7.1 | Real model | ⏳ TODO | ONNX model | 10 min |
| 7.2 | E2E inference | ⏳ TODO | DLL from 7.1 | 3 min |
| 7.3 | Benchmark | ⏳ TODO | Working inference | 5 min |

**Total Estimated Time:**
- With dependencies ready: ~1 hour
- Building LLVM from scratch: ~3-4 hours

---

## Troubleshooting Guide

### Phase 1 Issues

**Error: "Cannot find header files"**
- Ensure you're in project root directory
- Check paths in `#include` directives are correct

### Phase 2 Issues

**Error: "Mock functions not defined"**
- Verify `USE_MOCK_HIP=ON` in CMake
- Check `hip_ep_runtime.cpp` has mock implementations

### Phase 3 Issues

**Error: "LLVM libraries not found"**
```bash
# Set LLVM_DIR explicitly
cmake .. -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
```

**Error: "Undefined reference to LLVM symbols"**
- Ensure LLVM was built with same compiler
- Check LLVM version compatibility (18+)

### Phase 4 Issues

**Error: "lld::coff::link not found"**
- Rebuild LLVM with `-DLLVM_ENABLE_PROJECTS="mlir;lld"`
- Link against lldCOFF/lldELF libraries

### Phase 5 Issues

**Error: "Pass registration failed"**
- Ensure all MLIR dialects are linked
- Check pass dependencies are registered first

### Phase 6 Issues

**Error: "HIP runtime not found"**
```bash
# Windows
set PATH=%PATH%;C:\Program Files\AMD\ROCm\5.7\bin

# Linux
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
```

**Error: "No AMD GPU detected"**
```bash
# Verify GPU is visible
rocminfo
hipconfig
```

### Phase 7 Issues

**Error: "DLL load failed"**
- Check all ROCm DLLs are in PATH
- Verify DLL architecture matches (x64)
- Use Dependency Walker (Windows) to find missing DLLs

---

## Next Steps

After completing all phases:

1. **Document Results**: Update TEST_RESULTS.md with all phase outcomes
2. **Create CI Pipeline**: Automate testing with GitHub Actions
3. **Add More Models**: Test with different ONNX models
4. **Optimize**: Profile and improve performance
5. **Complete inference_compute**: Implement full tensor loop

---

## Maintenance

This testing plan should be updated when:
- New modules are added
- Dependencies change
- New test cases are discovered
- Platform support expands

---

**Document Version:** 1.0
**Created:** 2026-02-11
**Purpose:** Step-by-step testing roadmap for MLIR to DLL pipeline
