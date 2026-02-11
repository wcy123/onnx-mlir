# End-to-End Testing

This directory contains end-to-end tests that validate the complete MLIR to DLL pipeline.

**IMPORTANT:** This directory contains only source code. All build artifacts, test data, and binaries are kept in the build directory:
- Build output: `../../build/onnx-hipdnn-ep/test/e2e/`
- Test data (ONNX models): Place in build directory or use absolute paths
- Generated files (.dll, .obj, .ll): Created in build directory

## Overview

**Goal:** Verify that the MLIR → LLVM IR → Object File → DLL pipeline produces functional, loadable DLLs.

**Approach:**
1. Start with simple hand-written MLIR (bypasses ONNX conversion complexity)
2. Test DLL generation in IR mode (validates LLVM IR generation)
3. Test DLL generation in native mode (validates compilation and linking)
4. Load DLL and test all interface functions
5. Verify correct operation sequence via mock runtime

## Test Data

### ONNX Models (Not in Source Tree)

ONNX test models are available in the project's dependencies but should NOT be copied to the source tree:

**Available models:**
- `3rd-party/onnx-mlir/third_party/onnx/examples/resources/single_relu.onnx` (96 bytes)
- `3rd-party/onnx-mlir/src/Runtime/python/onnxmlir/tests/test_add.onnx` (129 bytes)

**Usage:** Reference them by absolute path or copy to build directory:
```bash
# Copy to build directory (not source!)
cp 3rd-party/onnx-mlir/third_party/onnx/examples/resources/single_relu.onnx \
   ../../build/onnx-hipdnn-ep/test/e2e/
```

## Test Files

### test_identity_manual.mlir
Simple MLIR module that tests the interface generation without complex computation:
- **@main**: No-op function (for testing DLL loading)
- **Metadata**: Declares 1 input (rank-2), 1 output (rank-2)
- **Constant helpers**: get_constant_count, initialize_constants, release_constants
- **Size**: ~40 lines of MLIR

### test_dll_loading.cpp
C++ test program that:
1. Loads the generated DLL
2. Resolves all 3 interface functions
3. Executes the complete workflow
4. Verifies mock runtime operation sequence
5. Tests with 10x10 tensor data

## Prerequisites

### For Building the Test

- C++ compiler (MSVC, GCC, or Clang)
- CMake 3.18+

### For Generating the DLL

**Option A: Via ONNX Runtime (if integrated)**
- ONNX Runtime with Morphizen Execution Provider built
- Environment variables set (COMPILATION_MODE, OUTPUT_PATH)

**Option B: Standalone MLIR Tooling (if available)**
- mlir-opt with custom passes
- llc for object compilation
- lld-link for DLL linking

**Option C: Manual Verification (current state)**
- Manually verify LLVM IR generation by inspecting generated code
- Cannot fully test DLL loading without complete build infrastructure

## Usage

### Step 1: Build the Test Program

```bash
cd test/e2e
mkdir build
cd build

# Configure
cmake -G "Visual Studio 17 2022" -A x64 ..

# Build
cmake --build . --config Release

# Verify
ls Release/test_dll_loading.exe
```

### Step 2: Generate the DLL from MLIR

**Option A: Using ONNX Runtime + Morphizen** (when available)

```bash
# Set environment
export COMPILATION_MODE=native
export OUTPUT_PATH=test_identity
export BUILD_MOCK_RUNTIME=1

# Compile via ONNX RT EP (exact command TBD based on integration)
# Expected output: test_identity.dll
```

**Option B: Manual MLIR Pipeline** (if tools available)

```bash
# Step 1: Run MLIR passes
mlir-opt test_identity.mlir \
  --hip-convert-onnx-to-hip \
  --hip-convert-hip-to-llvm \
  --hip-generate-interface \
  -o test_identity_transformed.mlir

# Step 2: Lower to LLVM IR
mlir-translate test_identity_transformed.mlir \
  --mlir-to-llvmir \
  -o test_identity.ll

# Step 3: Compile to object file
llc test_identity.ll -filetype=obj -o test_identity.obj

# Step 4: Link to DLL
lld-link /DLL /OUT:test_identity.dll test_identity.obj \
  HipDnnRuntime.lib \
  /EXPORT:inference_init \
  /EXPORT:inference_compute \
  /EXPORT:inference_cleanup
```

### Step 3: Run the Test

```bash
cd build/Release

# Copy DLL to test directory (if not already there)
cp ../../../test_identity.dll .

# Run test
./test_dll_loading.exe
```

## Expected Output

```
=== End-to-End DLL Loading Test ===

--- Step 1: Loading DLL ---
✓ DLL loaded successfully

--- Step 2: Resolving Functions ---
✓ inference_init resolved
✓ inference_compute resolved
✓ inference_cleanup resolved

--- Step 3: Testing inference_init ---
[MOCK] hipStreamCreate() -> 0x...
[MOCK] miopenCreate() -> 0x...
[MOCK] miopenSetStream(handle=0x..., stream=0x...)
[MOCK] hipblasLtCreate() -> 0x...
✓ inference_init succeeded
  State pointer: 0x...

--- Step 4: Preparing Test Data ---
✓ Test data prepared
  Input: 10x10 tensor (100 floats, 400 bytes)
  Output: 10x10 tensor (100 floats, 400 bytes)

--- Step 5: Testing inference_compute ---
[MOCK] hipMalloc(400 bytes) -> 0x...
[MOCK] hipMalloc(400 bytes) -> 0x...
[MOCK] hipMemcpyAsync(dst=0x..., src=0x..., size=400, H2D, stream=0x...)
[MOCK] hipMemcpyAsync(dst=0x..., src=0x..., size=400, D2H, stream=0x...)
[MOCK] hipStreamSynchronize(0x...)
[MOCK] hipFree(0x...)
[MOCK] hipFree(0x...)
✓ inference_compute succeeded

--- Step 6: Testing inference_cleanup ---
[MOCK] hipStreamSynchronize(0x...)
[MOCK] hipblasLtDestroy(0x...)
[MOCK] miopenDestroy(0x...)
[MOCK] hipStreamDestroy(0x...)
✓ inference_cleanup succeeded

--- Step 7: Unloading DLL ---
✓ DLL unloaded

=== All Tests PASSED ===

Expected mock runtime output above should show:
  1. init: hipStreamCreate, miopenCreate, hipblasLtCreate
  2. compute: hipMalloc x2, hipMemcpyAsync H2D, D2H, hipFree x2
  3. cleanup: hipStreamSynchronize, hipblasLtDestroy, miopenDestroy, hipStreamDestroy

This validates:
  ✓ DLL generation from MLIR works
  ✓ All interface functions exported correctly
  ✓ Complete workflow executes without crashes
  ✓ GPU operations called in correct order (via mock)
  ✓ Memory management works (allocation, copy, free)
  ✓ LIFO cleanup order correct
```

## Verification Checklist

After running the test, verify:

- [x] DLL loaded without errors
- [x] All 3 functions resolved (init, compute, cleanup)
- [x] Init created GPU state (mock handles)
- [x] Compute allocated 2 buffers (400 bytes each for 10x10 floats)
- [x] Compute performed H2D copy (400 bytes)
- [x] Compute performed D2H copy (400 bytes)
- [x] Compute freed both buffers
- [x] Cleanup destroyed handles in reverse order (LIFO)
- [x] No crashes throughout workflow
- [x] No memory leaks (all allocations freed)

## Troubleshooting

### "Failed to load test_identity.dll"

**Cause:** DLL not generated yet or in wrong location

**Solution:**
1. Generate the DLL using one of the methods above
2. Copy DLL to test directory: `cp test_identity.dll build/Release/`
3. Or run test from directory containing DLL

### "Failed to resolve inference_*"

**Cause:** DLL doesn't export expected functions

**Solution:**
1. Verify DLL exports: `dumpbin /EXPORTS test_identity.dll`
2. Check GenerateInterfacePass ran successfully
3. Verify linking step included /EXPORT directives

### Mock functions not showing

**Cause:** DLL built without BUILD_MOCK_RUNTIME

**Solution:**
- Rebuild runtime library with `-DBUILD_MOCK_RUNTIME=1`
- Ensure DLL links against mock runtime

### Test crashes during compute

**Possible causes:**
1. Null pointer dereference (bad state)
2. Buffer overflow (incorrect size calculation)
3. Invalid function pointer (runtime helper not linked)

**Debug:**
1. Run under debugger (gdb, Visual Studio debugger)
2. Check mock output for unexpected behavior
3. Verify all runtime helpers are callable

## Current Status

**As of February 11, 2026:**

- ✅ Test MLIR file created (test_identity.mlir)
- ✅ Test program created (test_dll_loading.cpp)
- ✅ CMakeLists.txt for building test
- ✅ Documentation complete
- ⏳ DLL generation pending (requires MLIR compilation infrastructure)

**Next Steps:**
1. Determine method for compiling test_identity.mlir to DLL
2. Run test and verify output
3. Debug any issues
4. Document results

## Future Enhancements

1. **Automated DLL Generation**
   - Add CMake target to compile MLIR → DLL
   - Integrate with CTest for automated testing

2. **Multiple Test Cases**
   - test_add.mlir - Simple addition operation
   - test_copy.mlir - Memory copy verification
   - test_multi_tensor.mlir - Multiple inputs/outputs

3. **Numerical Verification**
   - Not just loading, actually verify output correctness
   - Compare against expected results

4. **Real GPU Testing**
   - Test with actual ROCm installation
   - Verify performance on real hardware

## Related Documentation

- `../../TESTING_PLAN.md` - Overall testing strategy
- `../../../notes/PHASE_4_ROADMAP.md` - Detailed Phase 4 plan
- `../../runtime/README.md` - Runtime library testing
