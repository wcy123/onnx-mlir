# Mock Runtime Testing

## Purpose: Compiler Development Testing

**These tests validate the compiler-generated code structure, NOT GPU computation correctness.**

This project is a **compiler** that generates code calling HIP/MIOpen/hipBLASLt APIs. The mock runtime allows us to:

1. **Verify API contract correctness** - The generated code calls runtime functions with correct signatures and parameter ordering
2. **Test orchestration logic** - Handle creation, cleanup ordering (LIFO), null parameter handling
3. **Enable fast iteration** - Develop and test without GPU hardware or ROCm installation
4. **Run in CI** - No GPU runners required for compiler development

**Actual computation correctness is validated via end-to-end tests on real AMD GPU hardware.**

---

## Test Files

| Test File | Purpose |
|-----------|---------|
| `test_runtime_state.cpp` | Tests `RuntimeState` lifecycle: `runtime_state_init()` / `runtime_state_cleanup()` |
| `test_mock_runtime.cpp` | Tests individual API functions: convolution, GEMM, memory operations |

**Why two tests?**

- **`test_runtime_state.cpp`** - Focused unit test for the state management that the compiler generates calls to. Verifies init/cleanup can be called multiple times, handles null parameters correctly, and follows LIFO cleanup order.

- **`test_mock_runtime.cpp`** - Broader API coverage test. Verifies that all the operations the compiler might generate (convolution, GEMM, memory copy) have correct function signatures and parameter passing.

Both are needed because the compiler generates code that uses both:
1. State management (init/cleanup lifecycle)
2. Compute operations (convolution, GEMM within the compute phase)

---

## Phase 0 Implementation Status ✅ COMPLETE

**Comprehensive Mock Runtime** - Successfully implemented and tested

### What Was Implemented

1. **Comprehensive Mock Functions** (lib/Runtime/hip_ep_runtime.cpp)
   - ✅ All HIP handle functions (hipStreamCreate/Destroy, hipStreamSynchronize)
   - ✅ All HIP memory functions (hipMalloc, hipFree, hipMemcpyAsync)
   - ✅ All MIOpen functions (miopenCreate/Destroy/SetStream, tensor descriptors, convolution)
   - ✅ All hipBLASLt functions (hipblasLtCreate/Destroy, matrix layouts, matmul)
   - ✅ Print statements showing all parameters for verification

2. **Mock Header** (lib/Runtime/hip_ep_runtime_mock.h)
   - Declares mock handle creation functions
   - Only active when BUILD_MOCK_RUNTIME is defined

3. **Comprehensive Test** (test/runtime/test_mock_runtime.cpp)
   - Tests handle creation (stream, MIOpen, hipBLASLt)
   - Tests convolution operations with full parameter verification
   - Tests GEMM operations with matrix dimension verification
   - Tests memory copy operations (H2D, D2H)
   - Tests cleanup in correct order (LIFO)

### Test Results

```
=== All Mock Runtime Tests PASSED ===
```

All 7 test phases passed:
1. ✅ Handle Creation - Created stream, MIOpen handle, hipBLASLt handle
2. ✅ RuntimeState Structure - Verified opaque design
3. ✅ Constant Management - Functions available
4. ✅ Convolution Operation - Full parameter verification
5. ✅ GEMM Operation - Matrix dimensions verified
6. ✅ Memory Copy Operations - H2D and D2H tested
7. ✅ Cleanup - All resources freed in LIFO order

### Verified Output

The test successfully prints:
- Handle creation with memory addresses
- Convolution with tensor shapes: [1,3,224,224] input, [64,3,7,7] weights, [1,64,112,112] output
- Convolution parameters: pad=[3,3], stride=[2,2], dilation=[1,1]
- GEMM dimensions: M=1000, N=1, K=2048
- Memory operations with sizes and transfer directions
- Cleanup in reverse order (LIFO pattern verified)

## Building and Running

### Standalone Build (Recommended for Testing)

```bash
cd test/runtime
rm -rf build_standalone
mkdir build_standalone
cd build_standalone

# Configure
cmake -G "Visual Studio 17 2022" -A x64 ..

# Build
cmake --build . --config Release

# Run
./Release/test_mock_runtime.exe
```

### Full Project Build (with LLVM/MLIR)

If you need to build as part of the full project:

```bash
cd <project-root>
cmake -B build_mock_test -DBUILD_MOCK_RUNTIME=1 -DCMAKE_BUILD_TYPE=Release
cmake --build build_mock_test --config Release --target test_mock_runtime
./build_mock_test/test/Release/test_mock_runtime.exe
```

## Why Mock Runtime? (Benefits for Compiler Development)

When developing a compiler, you need to test that the **generated code is structurally correct** before running it on actual hardware. The mock runtime enables this by:

| Benefit | Why It Matters for Compiler Development |
|---------|----------------------------------------|
| **Test without ROCm** | Compiler developers can work on Windows/macOS without AMD GPU or ROCm. The real runtime runs on Linux with AMD GPUs. |
| **Verify API contracts** | The mock prints all parameters, so you can see exactly what the compiler-generated code is passing to each function. |
| **CI/CD friendly** | GitHub Actions doesn't have GPU runners. Mock tests can run on any CI system to catch regressions. |
| **Fast iteration** | Building/testing with real GPU takes longer. Mock tests run in seconds. |
| **Debug orchestration** | Print output shows initialization order, cleanup order (LIFO), and resource management. |

**Example**: If the compiler generates a call to `miopenConvolutionForward()` with wrong padding values, the mock prints:
```
[MOCK] pad=[3,3], stride=[2,2]
```
You immediately see the bug without needing a GPU.

## What Can Be Tested

**Currently testable:**
- Runtime library API correctness
- Parameter passing to GPU functions
- Memory allocation patterns
- Cleanup sequencing (LIFO verification)
- Error handling paths

**Not yet testable (requires full pipeline):**
- Generated MLIR interface functions (inference_init/compute/cleanup)
- End-to-end model execution
- Actual GPU computation
- Performance benchmarking

## Files Created/Modified

### Created:
- `lib/Runtime/hip_ep_runtime_mock.h` - Mock function declarations
- `test/runtime/test_mock_runtime.cpp` - Comprehensive test
- `test/runtime/standalone_CMakeLists.txt` - Standalone build config (copied to CMakeLists.txt)
- `test/runtime/build_and_test.sh` - Build script (optional)
- `test/runtime/README.md` - This file

### Modified:
- `lib/Runtime/hip_ep_runtime.cpp` - Added comprehensive mock implementations with prints
- `lib/Runtime/CMakeLists.txt` - Respect explicit BUILD_MOCK_RUNTIME flag
- `test/CMakeLists.txt` - Added mock runtime test target

## Technical Details

### Mock Implementation Strategy

All mock functions follow the same pattern:
1. Print the operation being performed
2. For operations with complex parameters (convolution, GEMM), print all relevant details
3. For handle creation, allocate fake memory (`malloc(8)`) and print address
4. For memory operations, use real `malloc`/`memcpy` but print GPU-style messages
5. Always return success (0) unless testing error paths

### Example Output Analysis

From a typical test run:

```
[MOCK] miopenConvolutionForward(
[MOCK]   input_shape=[1,3,224,224],
[MOCK]   weights_shape=[64,3,7,7],
[MOCK]   output_shape=[1,64,112,112],
[MOCK]   pad=[3,3], stride=[2,2], dilation=[1,1])
```

This tells us:
- ✅ Convolution called with correct input dimensions (1 batch, 3 channels, 224x224 image)
- ✅ Using 64 filters of size 7x7 over 3 input channels
- ✅ Output is correctly calculated as 1x64x112x112 (stride=2 halves spatial dims)
- ✅ Padding and stride parameters passed correctly

## Troubleshooting

### Build Errors

If you see linker errors about missing mock functions:
- Ensure BUILD_MOCK_RUNTIME=1 is defined
- Check that hip_ep_runtime_mock.h is included
- Verify mock functions are `extern "C"` (not `static`)

### Runtime Errors

The mock runtime should never fail unless testing error paths. If you see unexpected failures:
- Check that all handles are created before use
- Verify cleanup happens in reverse order (LIFO)
- Ensure memory is not freed twice

### Missing Print Output

If mock functions aren't printing:
- Verify BUILD_MOCK_RUNTIME=1 is defined during compilation
- Check that printf statements are present in mock functions
- Ensure stdout is not being buffered (flush if needed)
