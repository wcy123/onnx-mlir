<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# MLIR AOT Compilation Demo
## From ONNX Model → Native AMD GPU Code

**Presentation Guide**: Tech meeting with live demo capability

---

## Overview

Compile ONNX models ahead-of-time to native DLLs:
- No runtime LLVM/MLIR dependencies
- Embedded constant weights in compiled code
- Direct MIOpen/HIP calls
- Native GPU performance

**Demo model**: Two-layer convolution network (ResNet-style)
- Input: 1×3×224×224 (RGB image)
- Layer 1: 64 filters, 3×3 conv, stride=1 → 1×64×224×224
- Layer 2: 64 filters, 3×3 conv, stride=2 → 1×64×112×112
- 4 constant tensors embedded in compiled code

**Pipeline**: `ONNX → HIP Dialect → LLVM IR → C Interface → Native DLL`

---

## Live Demo

### Build the Tools

```bash
cd /path/to/onnx-hipdnn-ep

# Build both tools in one step
cmake -S . -B ../../build/onnx-hipdnn-ep -DBUILD_HIP_OPT_TOOL=ON -DBUILD_MLIR_HIP_COMPILER=ON
cmake --build ../../build/onnx-hipdnn-ep --config Debug --target hip-opt mlir-hip-compiler
```

### Stage 1: ONNX → HIP Dialect

```bash
../../build/onnx-hipdnn-ep/bin/hip-opt.exe \
  tools/hip-opt/demo_two_layer_conv.mlir \
  --convert-onnx-to-hip
```

**What you'll see**:
- 4 LLVM global constants discovered
- `hip.conv` operations with GPU memory types
- Helper functions: `initialize_constants()`, `release_constants()`

### Stage 2: HIP → LLVM IR

```bash
../../build/onnx-hipdnn-ep/bin/hip-opt.exe \
  tools/hip-opt/demo_two_layer_conv.mlir \
  --convert-onnx-to-hip \
  --convert-hip-to-llvm
```

**What you'll see**:
- Runtime function declarations: `miopenConvolutionForward`, `hipMalloc`
- Two-function architecture: `@main` (wrapper) + `@main_internal` (computation)
- Memref descriptor unpacking logic

### Stage 3: Generate C Interface

```bash
../../build/onnx-hipdnn-ep/bin/hip-opt.exe \
  tools/hip-opt/demo_two_layer_conv.mlir \
  --convert-onnx-to-hip \
  --convert-hip-to-llvm \
  --generate-interface
```

**What you'll see**:
- 3 exported functions: `inference_init()`, `inference_compute()`, `inference_cleanup()`
- C-ABI compliance attributes
- Error handling and validation logic

### Stage 4: Compile to Native DLL

```bash
# First, save the interface-generated MLIR to a file
../../build/onnx-hipdnn-ep/bin/hip-opt.exe \
  tools/hip-opt/demo_two_layer_conv.mlir \
  --convert-onnx-to-hip \
  --convert-hip-to-llvm \
  --generate-interface \
  > demo_with_interface.mlir

# Then compile to DLL
../../build/onnx-hipdnn-ep/bin/mlir-hip-compiler.exe \
  demo_with_interface.mlir \
  -o inference.dll \
  -v \
  --keep
```

**What you'll see**:
- LLVM IR translation progress
- Optimization at O2 level
- Object file generation (.obj)
- DLL linking with runtime libraries (HipDnnRuntime.lib, amdhip64.lib, MIOpen.lib)
- DLL export verification (inference_init, inference_compute, inference_cleanup)
- Intermediate files kept: .ll (LLVM IR), .obj (object file)

**Verify DLL exports**:
```bash
# On Windows
dumpbin /EXPORTS inference.dll

# Expected output:
#   inference_init
#   inference_compute
#   inference_cleanup
```

---

## Pipeline Breakdown

### Stage 1: ONNX → HIP Dialect

**Before** (ONNX operations):
```mlir
func.func @main(%input: tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32> {
  %weights1 = "onnx.Constant"() {value = dense<1.0> : tensor<64x3x3x3xf32>} : () -> tensor<64x3x3x3xf32>
  %bias1 = "onnx.Constant"() {value = dense<0.5> : tensor<64xf32>} : () -> tensor<64xf32>

  %conv1 = "onnx.Conv"(%input, %weights1, %bias1) {
    kernel_shape = [3, 3], strides = [1, 1], pads = [1, 1, 1, 1]
  } : (tensor<1x3x224x224xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x64x224x224xf32>

  // ... (layer 2 similar)
}
```

**After** (HIP dialect with constants):
```mlir
module attributes {hipdnn.input_count = 1, hipdnn.input_ranks = array<i64: 4>, ...} {
  // ✅ Constants discovered and hoisted to globals
  llvm.mlir.global internal constant @constant_0(dense<1.0> : tensor<64x3x3x3xf32>) : !llvm.array<1728 x f32>
  llvm.mlir.global internal constant @constant_1(dense<0.5> : tensor<64xf32>) : !llvm.array<64 x f32>
  // ... (2 more constants)

  func.func @main(%ctx: !hip.context, %input: memref<1x3x224x224xf32, 1>,
                   %output: memref<1x64x112x112xf32, 1>) -> i32 {
    // ✅ Retrieve pre-uploaded constants from GPU
    %weights1 = hip.get_constant(%ctx, 0) : memref<64x3x3x3xf32, 1>
    %bias1 = hip.get_constant(%ctx, 1) : memref<64xf32, 1>

    // ✅ Direct HIP operation (in-place semantics)
    %temp = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
    hip.conv(%ctx, %input, %weights1, %bias1, %temp) {kernel_shape = [3, 3], ...}

    // ... (layer 2 writes directly to %output)
    return 0 : i32
  }

  // ✅ Helper functions generated automatically
  func.func @initialize_constants(%ctx: !hip.context) -> i32 { ... }
  func.func @release_constants(%ctx: !hip.context) -> i32 { ... }
}
```

**Key transformations**:
- **Constant discovery**: 4 `onnx.Constant` → 4 `llvm.mlir.global`
- **Module metadata**: Captures input/output counts and ranks
- **In-place operations**: `hip.conv(ctx, in, w, b, out)` - no return value
- **GPU memory types**: `memref<..., 1>` (address space 1 = device memory)

---

### Stage 2: HIP → LLVM IR

**Two-Function Architecture**:

```mlir
// ✅ FUNCTION 1: Clean 3-parameter wrapper for external callers
llvm.func private @main(%ctx: !llvm.ptr, %inputs: !llvm.ptr, %outputs: !llvm.ptr) -> i32 {
  // Unpack memref struct arrays
  %input_struct = llvm.getelementptr %inputs[0] : (!llvm.ptr, i32) -> !llvm.ptr
  %input = llvm.load %input_struct : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, ...)>

  // Extract 11 fields: allocated_ptr, aligned_ptr, offset, sizes[4], strides[4]
  %ptr = llvm.extractvalue %input[0] : !llvm.struct<...>
  %size0 = llvm.extractvalue %input[3, 0] : !llvm.struct<...>
  // ... (10 more extracts)

  // Call internal function with all unpacked parameters
  %result = llvm.call @main_internal(%ctx, %ptr, %size0, ...) : (...) -> i32
  llvm.return %result : i32
}

// ✅ FUNCTION 2: Internal computation with unpacked memrefs (23 parameters)
llvm.func private @main_internal(%ctx: !llvm.ptr, %in_ptr: !llvm.ptr<1>, %in_size0: i64,
                                  %in_size1: i64, ..., %out_stride3: i64) -> i32 {
  // Rebuild memref descriptors from parameters
  %desc = llvm.mlir.poison : !llvm.struct<...>
  %d1 = llvm.insertvalue %in_ptr, %desc[0] : !llvm.struct<...>
  // ... (22 more insertvalue operations)

  // Get constants from GPU
  %weights = llvm.call @hip_get_constant(%ctx, 0) : (!llvm.ptr, i64) -> !llvm.ptr

  // Allocate temp buffer
  %temp = llvm.call @hipMalloc(%size) : (i64) -> !llvm.ptr

  // Call MIOpen
  %status = llvm.call @miopenConvolutionForward(%ctx, %in_ptr, %weights, ...) : (...) -> i32

  // ... (layer 2 similar)

  llvm.return %status : i32
}
```

**Key transformations**:
- **Array-based interface**: @main receives `void** inputs` and `void** outputs`
- **Unpacking logic**: GEP → load → extractvalue to access memref fields
- **Pure LLVM dialect**: No more `func.func`, `!hip.context`, or `arith.constant`
- **Scalable**: Works for N inputs/outputs via metadata-driven loops
- **Dynamic shapes ready**: Runtime dimensions flow through memref structs

---

### Stage 3: Interface Generation

**Generated C-ABI Functions**:

```mlir
// ✅ EXPORT 1: Initialize GPU state
llvm.func @inference_init(%state_ptr: !llvm.ptr) -> i32
    attributes {llvm.emit_c_interface, sym_visibility = "public"} {
  // Allocate state struct (32 bytes for GPU handles)
  %state = llvm.call @malloc(32) : (i64) -> !llvm.ptr

  // Check allocation success
  %is_null = llvm.icmp "eq" %state, %null : !llvm.ptr
  llvm.cond_br %is_null, ^error, ^success

^success:
  llvm.store %state, %state_ptr : !llvm.ptr
  return 0 : i32  // Success

^error:
  return 1 : i32  // Allocation failed
}

// ✅ EXPORT 2: Run inference
llvm.func @inference_compute(%state_ptr: !llvm.ptr, %inputs: !llvm.ptr,
                              %outputs: !llvm.ptr) -> i32
    attributes {llvm.emit_c_interface, sym_visibility = "public"} {
  // Validate input count (parse span_t->count field)
  %input_count_ptr = llvm.getelementptr %inputs[1] : (!llvm.ptr, i32) -> !llvm.ptr
  %input_count = llvm.load %input_count_ptr : i64
  %valid_in = llvm.icmp "eq" %input_count, 1 : i64

  // Validate output count
  %output_count_ptr = llvm.getelementptr %outputs[1] : (!llvm.ptr, i32) -> !llvm.ptr
  %output_count = llvm.load %output_count_ptr : i64
  %valid_out = llvm.icmp "eq" %output_count, 1 : i64

  llvm.cond_br %valid_in, ^check_out, ^error
^check_out:
  llvm.cond_br %valid_out, ^success, ^error

^success:
  // TODO: Call @main(%state, %inputs, %outputs)
  return 0 : i32  // Success

^error:
  return 5 : i32  // HIPDNN_ERROR_INVALID_INPUT
}

// ✅ EXPORT 3: Cleanup GPU state
llvm.func @inference_cleanup(%state_ptr: !llvm.ptr) -> i32
    attributes {llvm.emit_c_interface, sym_visibility = "public"} {
  llvm.call @free(%state_ptr) : (!llvm.ptr) -> ()
  return 0 : i32  // Always succeeds
}
```

**Key features**:
- **C calling convention**: `llvm.emit_c_interface` (no name mangling)
- **DLL exports**: `sym_visibility = "public"` (visible to GetProcAddress/dlsym)
- **Error handling**: Return codes 0 (success), 1 (alloc failed), 5 (invalid input)
- **span_t parsing**: Access count field at offset 1 via GEP

---

### Stage 4: DLL Compilation

**Four-step pipeline**: MLIR (LLVM dialect) → LLVM IR → Object File → DLL

**Step 1: Translate to LLVM IR**
- Convert MLIR to LLVM IR using MLIR's translation infrastructure
- Preserves all function declarations and C-ABI attributes
- Output: `.ll` file (LLVM IR text format)

**Step 2: Optimize LLVM IR**
- Run LLVM optimization passes (default: -O2)
- Function inlining, constant propagation, dead code elimination
- Output: Optimized LLVM IR

**Step 3: Compile to Object File**
- Generate native machine code for target platform (x86-64 Windows)
- Output: `.obj` file (PE/COFF format)

**Step 4: Link to DLL**
- Link object file with runtime libraries:
  - **HipDnnRuntime.lib** - Custom runtime (GPU handles, constant management)
  - **amdhip64.lib** - AMD HIP runtime
  - **MIOpen.lib** - Convolution operations
  - **hipblaslt.lib** - BLAS operations
- Use LLD-LINK (LLVM's linker) to create DLL
- Verify exported symbols: `inference_init`, `inference_compute`, `inference_cleanup`
- Output: `.dll` file (Windows) or `.so` (Linux)

**Tool**: `mlir-hip-compiler` (uses LLVMBackend + DLLLinker libraries)

**Options**:
- `--from-onnx-mlir` - Run ONNX→HIP→LLVM→Interface passes before compilation
- `-o <output>` - Output DLL filename
- `--mode <ir|object|dll>` - Stop after IR, object, or full DLL
- `-O <0-3>` - Optimization level (default: 2)
- `-v` - Verbose output
- `--keep` - Keep intermediate files (.ll, .obj)

---

## Try It Yourself

### Quick Start Commands

```bash
# 1. Build the tools
cd /path/to/onnx-hipdnn-ep
cmake -S . -B ../../build/onnx-hipdnn-ep -DBUILD_HIP_OPT_TOOL=ON -DBUILD_MLIR_HIP_COMPILER=ON
cmake --build ../../build/onnx-hipdnn-ep --config Debug --target hip-opt mlir-hip-compiler

# 2. Run Stage 1: ONNX → HIP
../../build/onnx-hipdnn-ep/bin/hip-opt.exe \
  tools/hip-opt/demo_two_layer_conv.mlir \
  --convert-onnx-to-hip \
  > ../output/my_stage1.mlir

# 3. Run Stage 2: HIP → LLVM
../../build/onnx-hipdnn-ep/bin/hip-opt.exe \
  tools/hip-opt/demo_two_layer_conv.mlir \
  --convert-onnx-to-hip \
  --convert-hip-to-llvm \
  > ../output/my_stage2.mlir

# 4. Run Stage 3: Generate Interface
../../build/onnx-hipdnn-ep/bin/hip-opt.exe \
  tools/hip-opt/demo_two_layer_conv.mlir \
  --convert-onnx-to-hip \
  --convert-hip-to-llvm \
  --generate-interface \
  > ../output/my_stage3.mlir

# 5. Run Stage 4: Compile to DLL
../../build/onnx-hipdnn-ep/bin/mlir-hip-compiler.exe \
  ../output/my_stage3.mlir \
  -o ../output/my_inference.dll \
  -v \
  --keep

# 6. Verify DLL Exports
dumpbin /EXPORTS ../output/my_inference.dll
# Expected: inference_init, inference_compute, inference_cleanup
```

### Expected Output

**Stage 1** should show:
- Module attributes with `hipdnn.input_count`, `hipdnn.input_ranks`, etc.
- 4 `llvm.mlir.global` constants
- `func.func @main` with `!hip.context` parameter
- Helper functions: `initialize_constants`, `release_constants`

**Stage 2** should show:
- Runtime function declarations: `@miopenConvolutionForward`, `@hipMalloc`
- `llvm.func @main` with 3 parameters (wrapper function)
- `llvm.func @main_internal` with 23 parameters (computation function)

**Stage 3** should show:
- 3 exported functions with `sym_visibility = "public"`:
  - `@inference_init`
  - `@inference_compute`
  - `@inference_cleanup`
- All 3 have `llvm.emit_c_interface` attribute

### Verification Commands

```bash
# Count functions
grep "llvm.func @" ../output/my_stage2.mlir | wc -l
# Expected: 6 (malloc, free, get_constant_count, init, compute, cleanup)

# Check exports
grep "sym_visibility.*public" ../output/my_stage3.mlir
# Expected: 3 lines (inference_init, inference_compute, inference_cleanup)

# Check metadata
grep "hipdnn\." ../output/my_stage1.mlir
# Expected: 4 attributes (input_count, input_ranks, output_count, output_ranks)

# Verify DLL exports
dumpbin /EXPORTS ../output/my_inference.dll | grep "inference_"
# Expected: inference_init, inference_compute, inference_cleanup

# Check intermediate files (if --keep used)
ls -lh ../output/my_stage3.ll ../output/my_stage3.obj ../output/my_inference.dll
```

---

## Architecture Reference (Appendix)

### Full Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    ONNX Model (Input)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                ┌────▼────────────────────────────────┐
                │  COMPILE TIME (Two Paths)           │
                ├─────────────────────────────────────┤
                │                                      │
                │  PATH A: Standalone Tools (Dev)     │
                │  --------------------------------     │
                │  1. hip-opt: ONNX → MLIR            │
                │     • --convert-onnx-to-hip          │
                │     • --convert-hip-to-llvm          │
                │     • --generate-interface           │
                │  2. mlir-hip-compiler: MLIR → DLL   │
                │     • Translate to LLVM IR           │
                │     • Optimize (O2)                  │
                │     • Compile to object file         │
                │     • Link to DLL with runtime libs  │
                │                                      │
                │  PATH B: Integrated (Production)     │
                │  --------------------------------     │
                │  1. onnx-mlir: ONNX → MLIR          │
                │  2. Level-1 Pass:                    │
                │     • Run same MLIR passes           │
                │     • Use same LLVMBackend/DLLLinker │
                │     • Embed DLL in EPContext         │
                │                                      │
                └────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          ONNX Model + EPContext (Cached Artifact)            │
│          Contains: Pre-compiled DLL with embedded data       │
└────────────────────┬────────────────────────────────────────┘
                     │
                ┌────▼────────────────────────────────┐
                │  RUNTIME (Custom Op)                 │
                │  Dependencies: HIP, MIOpen (~5MB)    │
                │  NO LLVM/MLIR!                       │
                ├─────────────────────────────────────┤
                │  1. Load DLL from EPContext memory   │
                │  2. inference_init(state):           │
                │     • Create GPU handles             │
                │     • Upload constants to GPU        │
                │  3. inference_compute(state, ...):   │
                │     • Use pre-uploaded constants     │
                │     • Execute on GPU                 │
                │  4. inference_cleanup(state):        │
                │     • Free GPU memory                │
                └─────────────────────────────────────┘
```

### Tools Reference

**hip-opt** (tools/hip-opt/):
- Purpose: MLIR transformation and pass testing
- Input: ONNX MLIR, HIP dialect MLIR
- Output: Transformed MLIR (text format)
- Passes: --convert-onnx-to-hip, --convert-hip-to-llvm, --generate-interface
- Usage: Development, debugging, testing individual passes

**mlir-hip-compiler** (tools/mlir-hip-compiler/):
- Purpose: End-to-end DLL compilation
- Input: MLIR in LLVM dialect (assumes passes already run) or ONNX-MLIR (with --from-onnx-mlir)
- Output: Native DLL (.dll) with exported C-ABI functions
- Options: -o <output>, --mode <ir|object|dll>, -O <0-3>, -v, --keep, --from-onnx-mlir
- Usage: Production artifact generation, standalone testing
- Dependencies: Links with HipDnnRuntime.lib, amdhip64.lib, MIOpen.lib, hipblaslt.lib

**Workflow:**
```bash
# Two-step workflow (development)
hip-opt input.mlir --convert-onnx-to-hip --convert-hip-to-llvm --generate-interface -o transformed.mlir
mlir-hip-compiler transformed.mlir -o output.dll -v

# One-step workflow (production-like)
mlir-hip-compiler input.mlir -o output.dll --from-onnx-mlir -v
```

### For Deep Dive

**Architecture & Design Documents** (✅ = self-reviewed):
- ✅ [ARCHITECTURE.md](ARCHITECTURE.md) - Complete system architecture, EPContext integration (v2.3)
- [MLIR-COMPILATION-DESIGN.md](MLIR-COMPILATION-DESIGN.md) - MLIR module structure, lowering pipeline
- [STATE-AND-CONTEXT.md](STATE-AND-CONTEXT.md) - State lifecycle, naming conventions
- [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md) - Full constant handling design (6 phases)
- [INTERFACE-DESIGN.md](INTERFACE-DESIGN.md) - C-ABI interface specification

### Full Code Examples

**Input ONNX Model** (115 lines):
```mlir
// Two conv layers with embedded constant weights/biases
func.func @main(%input: tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32> {
  // Layer 1 constants
  %weights1 = "onnx.Constant"() {
    value = dense<1.0> : tensor<64x3x3x3xf32>
  } : () -> tensor<64x3x3x3xf32>
  %bias1 = "onnx.Constant"() {
    value = dense<0.5> : tensor<64xf32>
  } : () -> tensor<64xf32>

  // Layer 1: Conv (3x3, stride=1, same size)
  %conv1 = "onnx.Conv"(%input, %weights1, %bias1) {
    kernel_shape = [3, 3], strides = [1, 1],
    pads = [1, 1, 1, 1], dilations = [1, 1], group = 1 : si64
  } : (tensor<1x3x224x224xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>)
      -> tensor<1x64x224x224xf32>

  // Layer 2 constants
  %weights2 = "onnx.Constant"() {
    value = dense<2.0> : tensor<64x64x3x3xf32>
  } : () -> tensor<64x64x3x3xf32>
  %bias2 = "onnx.Constant"() {
    value = dense<0.1> : tensor<64xf32>
  } : () -> tensor<64xf32>

  // Layer 2: Conv (3x3, stride=2, halves dimensions)
  %conv2 = "onnx.Conv"(%conv1, %weights2, %bias2) {
    kernel_shape = [3, 3], strides = [2, 2],
    pads = [1, 1, 1, 1], dilations = [1, 1], group = 1 : si64
  } : (tensor<1x64x224x224xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>)
      -> tensor<1x64x112x112xf32>

  return %conv2 : tensor<1x64x112x112xf32>
}
```

Full outputs available in `../output/` directory.

---

## Maintenance Notes

**What is this document?**
A guide for live MLIR compilation demos. Use it for free-form technical discussions, not as a rigid script.

**Two simple principles:**

1. **Use real output** - Copy examples from actual compiler runs (files in `../output/`), not made-up code
2. **Don't make stale claims** - Avoid line counts, timing estimates, or other numbers that change frequently

That's it. Keep the demos working, keep the examples real.
