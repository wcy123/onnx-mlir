<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Stage 4 DLL Compilation - Progress Summary

## Completed ✅

### 1. LLD CommandLine Conflict Resolution
**Problem**: mlir-hip-compiler and LLD both tried to use LLVM's global CommandLine system, causing:
```
lld-link (LLVM option parsing): Not enough positional command line arguments specified!
```

**Solution**: Replaced LLVM CommandLine (`cl::opt`) with manual argument parsing in mlir-hip-compiler

**Files Modified**:
- `tools/mlir-hip-compiler/main.cpp` - Manual Options struct parsing
- `lib/Backend/DLLLinker.cpp` - LLD library integration, fixed .lib path handling
- `lib/Backend/CMakeLists.txt` - LLD library detection
- `notes/lld-commandline-conflict.md` - Comprehensive documentation

**Result**: LLD successfully invokes without CommandLine errors!

### 2. HipDnnRuntime Library Built
**Status**: Successfully built `HipDnnRuntime.lib` (409 KB)
- Location: `../../build/onnx-hipdnn-ep.2/lib/Runtime/Debug/HipDnnRuntime.lib`
- Mock mode: Yes (ROCm not available)
- IR merging: Disabled (Clang not found during build)

**Functions available in runtime**:
- `hipdnn_ep_state_init` - Create runtime state
- `hipdnn_ep_state_cleanup` - Destroy runtime state
- `hipdnn_ep_get_stream` - Get HIP stream
- `hipdnn_ep_upload_constant` - Upload constant to GPU
- `hipdnn_ep_get_constant` - Get GPU pointer for constant
- `hipdnn_ep_release_constant` - Release constant memory
- `wrap_miopenConvolutionForward` - MIOpen convolution wrapper
- `wrap_hipblasLtGemm` - hipBLASLt GEMM wrapper
- `wrap_hipMalloc`, `wrap_hipFree` - Memory management wrappers
- `wrap_hipMemcpyH2D`, `wrap_hipMemcpyD2H` - Memory copy wrappers

### 3. DEMO.md Updated with Real Output
**Changes**:
- Updated Stage 4 status: Steps 1-7 complete (MLIR → Object → LLD invocation)
- Added real command output from test_identity.dll compilation
- Showed LLD-LINK command successfully invoked
- Documented current blocker (symbol mismatch)
- Listed generated intermediate files

**Commits**:
- `b54112c` - fix: resolve LLD CommandLine conflict in mlir-hip-compiler
- `a638082` - docs: update DEMO.md Stage 4 with successful LLD compilation

### 4. mlir-hip-compiler Improvements
**Search paths updated**:
- Now searches Debug and Release build directories
- Finds HipDnnRuntime.lib successfully

**Library path handling fixed**:
- Checks if `.lib` extension already present before appending
- Prevents double `.lib.lib` extension error

## Current Blocker ⚠️

### Symbol Name Mismatch

**Problem**: Test MLIR (`test/mlir/identity_llvm.mlir`) declares runtime functions with different names than the actual runtime library exports.

**Test MLIR expects**:
```mlir
llvm.func @runtime_state_init(!llvm.ptr) -> i32
llvm.func @runtime_state_cleanup(!llvm.ptr) -> i32
llvm.func @runtime_prepare_inference(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @runtime_cleanup_inference(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
```

**Runtime library exports** (from `hipdnn_ep_runtime.h`):
```c
int hipdnn_ep_state_init(RuntimeState **out_state);
int hipdnn_ep_state_cleanup(RuntimeState *state);
// Note: runtime_prepare_inference and runtime_cleanup_inference don't exist!
```

**Linkage error**:
```
DLL: error: undefined symbol: runtime_state_init
DLL: error: undefined symbol: runtime_state_cleanup
DLL: error: undefined symbol: runtime_prepare_inference
DLL: error: undefined symbol: runtime_cleanup_inference
```

### Root Cause

The test MLIR file was created for an older/different version of the runtime API. The current runtime has evolved to use:
1. Different function names (`hipdnn_ep_*` prefix instead of `runtime_*`)
2. Different inference workflow (no `prepare_inference` / `cleanup_inference` functions)

### Also: mainCRTStartup Error

```
DLL: error: <root>: undefined symbol: mainCRTStartup
```

**Cause**: `/DLL` flag expects no entry point, but linker is looking for `mainCRTStartup`. Need to add `/NOENTRY` flag.

## Solutions (Pick One)

### Option A: Update Test MLIR (Recommended)
Update `test/mlir/identity_llvm.mlir` to use current runtime API:
1. Replace `runtime_state_init` with `hipdnn_ep_state_init`
2. Replace `runtime_state_cleanup` with `hipdnn_ep_state_cleanup`
3. Remove `runtime_prepare_inference` / `runtime_cleanup_inference` calls
4. Use low-level wrappers directly (`wrap_hipMalloc`, `wrap_hipMemcpyH2D`, etc.)

### Option B: Add Runtime Wrapper Functions
Add wrapper functions to `hipdnn_ep_runtime.cpp` with old names:
```cpp
extern "C" int runtime_state_init(void **out_state) {
  return hipdnn_ep_state_init((RuntimeState **)out_state);
}

extern "C" int runtime_state_cleanup(void *state) {
  return hipdnn_ep_state_cleanup((RuntimeState *)state);
}

// Need to implement runtime_prepare_inference and runtime_cleanup_inference
```

### Option C: Use GenerateInterface Pass Output
Instead of using the hand-written `identity_llvm.mlir`, use the output from the GenerateInterface pass which should match the current runtime API.

## Recommendation

**Use Option C**: The GenerateInterface pass automatically generates interface code that matches the current runtime. Testing should use real pipeline output, not hand-written MLIR.

**Steps**:
1. Run full ONNX → HIP → LLVM → Interface pipeline
2. Save generated MLIR with interface
3. Test mlir-hip-compiler with that output
4. Verify DLL links successfully

## What Works Now

**Full pipeline (Steps 1-7)**:
```
Input MLIR
  ↓ Step 1: Parse
  ↓ Step 2: MLIR Passes
  ↓ Step 3: LLVM IR Translation
  ↓ Step 3.5: Runtime Module Linking
  ↓ Step 4: Optimization (O2)
  ↓ Step 5: LLVM IR Emission (.ll file)
  ↓ Step 6: Object Compilation (.obj file)
  ↓ Step 7: LLD Invocation ✅
  ↓ (blocked: symbol resolution)
  ✗ DLL Creation
```

**Generated artifacts** (from test run):
- `test_identity.ll` - LLVM IR (2.1 KB)
- `test_identity.obj` - Object file (1.1 KB, PE/COFF format)
- `test_identity.obj.def` - Export definitions (75 bytes)

**Key achievement**: LLD library successfully integrated and invoked!

## Next Steps

1. **Fix symbol mismatch** (Option C recommended)
2. **Add /NOENTRY to LLD args** (fix mainCRTStartup error)
3. **Test full pipeline** with real GenerateInterface output
4. **Verify DLL exports** with dumpbin
5. **Create output examples** in `../output/` directory
6. **Update DEMO.md** with complete successful DLL compilation

## Additional Fixes Needed

### DLLLinker.cpp
Add `/NOENTRY` flag for DLL without entry point:
```cpp
argStrings.push_back("/DLL");
argStrings.push_back("/NOENTRY");  // Add this
argStrings.push_back("/OUT:" + outputDLL);
```

### Test with Real Pipeline
```bash
# Generate interface MLIR from ONNX
../../build/onnx-hipdnn-ep.2/bin/Debug/hip-opt.exe \
  tools/hip-opt/demo_two_layer_conv.mlir \
  --convert-onnx-to-hip \
  --convert-hip-to-llvm \
  --generate-interface \
  > generated_with_interface.mlir

# Compile to DLL
../../build/onnx-hipdnn-ep.2/bin/Debug/mlir-hip-compiler.exe \
  generated_with_interface.mlir \
  -o inference.dll \
  --mode dll \
  -v \
  --keep
```

## Summary

We've successfully:
- ✅ Resolved LLD CommandLine conflict
- ✅ Built HipDnnRuntime library
- ✅ Integrated LLD as library (no subprocess calls)
- ✅ Fixed library path handling
- ✅ Updated DEMO.md with real evidence
- ✅ Steps 1-7 of compilation pipeline working

Remaining work:
- ⚠️ Fix symbol name mismatch (use GenerateInterface output)
- ⚠️ Add /NOENTRY flag to DLL linker
- ⚠️ Test end-to-end with real pipeline
- ⚠️ Verify DLL exports and create final documentation

**Progress**: ~90% complete for Stage 4 DLL compilation!
