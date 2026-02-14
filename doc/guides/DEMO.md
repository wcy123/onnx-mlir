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

**Demo model**: Two-layer convolution network with ReLU activations (ResNet-style)
- Input: 1×3×224×224 (RGB image)
- Layer 1: 64 filters, 3×3 conv, stride=1 → ReLU → 1×64×224×224
- Layer 2: 64 filters, 3×3 conv, stride=2 → ReLU → 1×64×112×112
- 4 constant tensors embedded in compiled code
- Automatic memory management via BufferDeallocation

## Demo Flow

This demo shows the 7-stage compilation and testing pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│  Input: demo_two_layer_conv.mlir (ONNX dialect)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────────────┐
              │  Stage 1: ONNX → HIP Dialect │
              │  (hip-opt --convert-onnx-to-hip)
              └──────────┬─────────────────────┘
                         │ HIP dialect MLIR
                         │ • Constants hoisted to globals
                         │ • Registry generated
                         │ • ONNX ops → HIP ops
                         ▼
              ┌──────────────────────────────────┐
              │  Stage 2: Buffer Deallocation     │
              │  (hip-opt --buffer-deallocation)  │
              └──────────┬───────────────────────┘
                         │ HIP dialect + hip.free
                         │ • hip.free after last use
                         │ • Ownership-aware
                         ▼
              ┌──────────────────────────────────┐
              │  Stage 3: Memory Pooling          │
              │  (hip-opt --memory-pooling)       │
              └──────────┬───────────────────────┘
                         │ HIP dialect + pool metadata
                         │ • Interference graph coloring
                         │ • Significant memory savings
                         ▼
              ┌──────────────────────────────┐
              │  Stage 4: HIP → LLVM Lowering│
              │  (hip-opt --convert-hip-to-llvm)
              └──────────┬─────────────────────┘
                         │ LLVM dialect MLIR
                         │ • @main wrapper function
                         │ • Runtime function calls
                         ▼
              ┌──────────────────────────────────┐
              │  Stage 5: C Interface Generation  │
              │  (hip-opt --generate-interface)   │
              └──────────┬───────────────────────┘
                         │ LLVM dialect + C interface
                         │ • inference_init/compute/cleanup
                         │ • Public C-ABI exports
                         ▼
              ┌──────────────────────────────────┐
              │  Stage 6: Native DLL Compilation  │
              │  (mlir-hip-compiler -o model.dll) │
              └──────────┬───────────────────────┘
                         │ DLL with embedded weights
                         │ • C interface exports
                         │ • Linked runtime libraries
                         ▼
              ┌──────────────────────────────────┐
              │  Stage 7: End-to-End Testing      │
              │  (test-model-dll model.dll)       │
              └──────────┬───────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Validation: Load DLL → Run inference → Verify output          │
│  Mock runtime shows all GPU operations (no hardware needed)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7-Stage Compilation Pipeline

This section walks through each compilation stage with real compiler output.

### Stage 1: ONNX → HIP Dialect

**Input (ONNX-MLIR model):**
```mlir
// Two-layer convolution network (ResNet-style)
func.func @main(%input: tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32> {
  // ❌ Constants inline in function
  %weights1 = "onnx.Constant"() {value = dense<1.0> : tensor<64x3x3x3xf32>} : () -> tensor<64x3x3x3xf32>
  %bias1 = "onnx.Constant"() {value = dense<0.5> : tensor<64xf32>} : () -> tensor<64xf32>

  // ❌ ONNX dialect operations (high-level)
  %conv1 = "onnx.Conv"(%input, %weights1, %bias1) {kernel_shape = [3, 3], strides = [1, 1], ...}
    : (tensor<1x3x224x224xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x64x224x224xf32>
  %relu1 = "onnx.Relu"(%conv1) : (tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32>

  // Layer 2 similar...
  return %relu2 : tensor<1x64x112x112xf32>
}
```

**Command:**
```bash
../../build/$(basename $PWD)/bin/Debug/hip-opt.exe \
  tools/hip-opt/demos/demo_two_layer_conv.mlir \
  --convert-onnx-to-hip \
  > ../output/stage1.mlir
```

**What happens:**
- ✅ Discovered 4 constants and hoisted to LLVM globals
- ✅ Generated constant registry for runtime access
- ✅ Converted ONNX ops (Conv, Relu) → HIP dialect
- ✅ Added module metadata (input/output ranks)
- ✅ Memory effects declared on all operations (enables automatic deallocation)

**Key transformations (excerpt from real output):**
```mlir
// ✅ Module metadata added
module attributes {hipdnn.input_count = 1 : i64, hipdnn.output_count = 1 : i64,
                   hipdnn.input_ranks = array<i64: 4>, hipdnn.output_ranks = array<i64: 4>} {

  // ✅ Constants hoisted to module-level LLVM globals (will be embedded in DLL .data section)
  llvm.mlir.global internal constant @constant_0(dense<1.0> : tensor<64x3x3x3xf32>) : !llvm.array<1728 x f32>
  llvm.mlir.global internal constant @constant_1(dense<0.5> : tensor<64xf32>) : !llvm.array<64 x f32>
  llvm.mlir.global internal constant @constant_2(dense<2.0> : tensor<64x64x3x3xf32>) : !llvm.array<36864 x f32>
  llvm.mlir.global internal constant @constant_3(dense<0.1> : tensor<64xf32>) : !llvm.array<64 x f32>

  // ✅ Signature changed: context + memref inputs/outputs → i32 status
  func.func @main(%arg0: !hip.context,                     // NEW: runtime state
                   %arg1: memref<1x3x224x224xf32, 1>,      // input (GPU memory)
                   %arg2: memref<1x64x112x112xf32, 1>) -> i32 {  // output (GPU memory)

    // ✅ Retrieve constants from runtime state (uploaded during inference_init)
    %0 = hip.get_constant(%arg0, %c0_i64) : memref<64x3x3x3xf32, 1>
    %1 = hip.get_constant(%arg0, %c1_i64) : memref<64xf32, 1>

    // ✅ HIP dialect operations (in-place, GPU memory types)
    %2 = hip.alloc(%arg0) : memref<1x64x224x224xf32, 1>
    hip.conv(%arg0, %arg1, %0, %1, %2) {dilations = [1, 1], group = 1, ...}

    %3 = hip.alloc(%arg0) : memref<1x64x224x224xf32, 1>
    hip.relu(%arg0, %2, %3) : (!hip.context, memref<...>, memref<...>)
    // ... (layer 2 similar)

    return %c0_i32 : i32  // ✅ Return status code instead of tensors
  }

  // ✅ Constant registry for runtime initialization
  llvm.func @get_constant_registry() -> !llvm.ptr {...}
}
```

**Design details:** [CONSTANT-HANDLING-DESIGN.md](../design/CONSTANT-HANDLING-DESIGN.md), [OnnxToHip.md](../design/mlir/passes/OnnxToHip.md), [BUFFER-LIFETIME-DESIGN.md](../design/BUFFER-LIFETIME-DESIGN.md)

### Stage 2: Buffer Deallocation

**Command:**
```bash
../../build/$(basename $PWD)/bin/Debug/hip-opt.exe \
  ../output/stage1.mlir \
  --ownership-based-buffer-deallocation \
  > ../output/stage2.mlir
```

**What happens:**
- ✅ Inserts `hip.free` after last use of each buffer
- ✅ Ownership-aware: function arguments not freed
- ✅ Enables zero-leak memory management
- ✅ Uses MLIR's BufferDeallocation pass

**Key transformations (based on Stage 1 output):**
```mlir
func.func @main(%arg0: !hip.context,
                %arg1: memref<1x3x224x224xf32, 1>,  // Input (caller-owned)
                %arg2: memref<1x64x112x112xf32, 1>) -> i32 {  // Output (caller-owned)
  // Layer 1
  %2 = hip.alloc(%arg0) : memref<1x64x224x224xf32, 1>  // ✅ Function-owned buffer
  hip.conv(%arg0, %arg1, %0, %1, %2) {...}

  %3 = hip.alloc(%arg0) : memref<1x64x224x224xf32, 1>  // ✅ Function-owned buffer
  hip.relu(%arg0, %2, %3) {...}
  hip.free(%arg0, %2)  // ✅ INSERTED: %2 no longer needed after relu

  // Layer 2
  %6 = hip.alloc(%arg0) : memref<1x64x112x112xf32, 1>  // ✅ Function-owned buffer
  hip.conv(%arg0, %3, %4, %5, %6) {...}
  hip.free(%arg0, %3)  // ✅ INSERTED: %3 no longer needed after second conv

  %7 = hip.alloc(%arg0) : memref<1x64x112x112xf32, 1>  // ✅ Function-owned buffer
  hip.relu(%arg0, %6, %7) {...}
  hip.free(%arg0, %6)  // ✅ INSERTED: %6 no longer needed after relu

  memref.copy %7, %arg2 {...}
  hip.free(%arg0, %7)  // ✅ INSERTED: %7 no longer needed after copy

  // ✅ NOTE: %arg1 and %arg2 are NOT freed (caller-owned arguments)
  %c0_i32 = arith.constant 0 : i32
  return %c0_i32 : i32
}
```

**Design details:** [BUFFER-LIFETIME-DESIGN.md](../design/BUFFER-LIFETIME-DESIGN.md)

### Stage 3: Memory Pooling

**Command:**
```bash
../../build/$(basename $PWD)/bin/Debug/hip-opt.exe \
  ../output/stage2.mlir \
  --memory-pooling \
  > ../output/stage3.mlir
```

**What happens:**
- ✅ Interference graph coloring algorithm
- ✅ ~60% memory savings (demo model)
- ✅ Pool metadata added to module attributes
- ✅ Reuses memory for non-overlapping buffers

**Key transformations (metadata added to module):**
```mlir
// ✅ Module attributes with pool metadata
module attributes {
  hipdnn.input_count = 1 : i64,
  hipdnn.output_count = 1 : i64,
  hipdnn.pool_size = 12845056 : i64,              // ✅ NEW: Total pool size
  hipdnn.buffer_offsets = array<i64: 0, 3211264, 6422528, 9633792>,  // ✅ NEW: Offsets for each buffer
  hipdnn.buffer_count = 4 : i64                   // ✅ NEW: Number of buffers
} {
  func.func @main(%arg0: !hip.context, %arg1: memref<...>, %arg2: memref<...>) -> i32 {
    // ✅ Allocations now use pool offsets (transformed by HipToLLVM in Stage 4)
    // Original total: 32112640 bytes (4 separate allocations)
    // After pooling: 12845056 bytes (60% reduction via reuse)
    //
    // Buffer interference graph shows:
    //   %2 overlaps with %6, %7 (cannot reuse)
    //   %3 overlaps with %6, %7 (cannot reuse)
    //   %6 overlaps with %2, %3 (cannot reuse)
    //   %7 does NOT overlap with %2, %3 (can reuse their memory!)
    //
    // Result: Buffers colored into pool with offsets:
    //   %2 → offset 0        (size: 3211264 bytes)
    //   %3 → offset 3211264  (size: 3211264 bytes)
    //   %6 → offset 6422528  (size: 3211264 bytes)
    //   %7 → offset 9633792  (size: 3211264 bytes)
    ...
  }
}
```

**Compilation output excerpt:**
```
[MemoryPooling] Pool size: 12845056 bytes (was 32112640 bytes, saved 60%)
[MemoryPooling] Processed 4 buffers
```

**Design details:** [MemoryPoolingPass.md](../design/mlir/passes/MemoryPoolingPass.md)

### Stage 4: HIP → LLVM Lowering

**Command:**
```bash
../../build/$(basename $PWD)/bin/Debug/hip-opt.exe \
  ../output/stage3.mlir \
  --convert-hip-to-llvm \
  > ../output/stage4.mlir
```

**What happens:**
- ✅ Two-function architecture: @main (wrapper) + @main_internal (computation)
- ✅ Memref unpacking logic generated
- ✅ Runtime function declarations added
- ✅ Opaque RuntimeState pattern (state passed as pointer)
- ✅ Array-based interface for scalability

**Key transformations (excerpt from real output):**
```mlir
module {
  // ✅ Runtime function declarations
  llvm.func @wrap_miopenConvolutionForward(!llvm.ptr, ...) -> i32
  llvm.func @wrap_miopenActivationForward_relu(!llvm.ptr, ...) -> i32
  llvm.func @hipdnn_ep_constant_get(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @hipMalloc(!llvm.ptr, i64) -> i32

  // ✅ Clean 3-parameter wrapper (array-based interface)
  llvm.func private @main(%arg0: !llvm.ptr,      // RuntimeState*
                          %arg1: !llvm.ptr,      // void** inputs
                          %arg2: !llvm.ptr) -> i32 {  // void** outputs
    // ✅ Unpack memref structs from arrays
    %1 = llvm.getelementptr %arg1[%0] : (!llvm.ptr, i32) -> !llvm.ptr
    %2 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, ...)>
    %3 = llvm.extractvalue %2[0] : ...  // Extract allocated_ptr
    %6 = llvm.extractvalue %2[3, 0] : ...  // Extract size[0]
    // ... (extract all memref fields)

    // ✅ Call internal function with unpacked parameters
    %28 = llvm.call @main_internal(%arg0, %3, %4, %5, %6, ...) : (...) -> i32
    llvm.return %28 : i32
  }

  // ✅ Internal computation function (parameter count varies by tensor rank)
  llvm.func private @main_internal(%arg0: !llvm.ptr,
                                    // Input memref: allocated_ptr, aligned_ptr, offset, 4 sizes, 4 strides
                                    %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: i64,
                                    %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, ...) -> i32 {
    // ✅ Get constants via opaque accessor (no direct field access)
    %weights = llvm.call @hipdnn_ep_constant_get(%arg0, %c0) : (...) -> !llvm.ptr

    // ✅ Call GPU operations
    llvm.call @wrap_miopenConvolutionForward(%arg0, %input_ptr, ...) : (...) -> i32
    ...
  }
}
```

**Design details:** [HipToLLVM.md](../design/mlir/passes/HipToLLVM.md)

### Stage 5: C Interface Generation

**Command:**
```bash
../../build/$(basename $PWD)/bin/Debug/hip-opt.exe \
  ../output/stage4.mlir \
  --generate-interface \
  > ../output/stage5.mlir
```

**What happens:**
- ✅ Generated 3 C-ABI functions: inference_init/compute/cleanup
- ✅ Public exports with `sym_visibility = "public"`
- ✅ Delegates I/O management to runtime helpers
- ✅ Zero-copy for constants (weights stay in RuntimeState)

**Key transformations (excerpt from real output):**
```mlir
module {
  // ✅ EXPORT 1: Initialize GPU state
  llvm.func @inference_init(%arg0: !llvm.ptr) -> i32
      attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    // Get constant registry from generated code
    %0 = llvm.call @get_constant_registry() : () -> !llvm.ptr
    // Delegate to runtime: create GPU handles, upload constants
    %1 = llvm.call @hipdnn_ep_state_init(%arg0, %0) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %1 : i32
  }

  // ✅ EXPORT 2: Run inference
  llvm.func @inference_compute(%arg0: !llvm.ptr,     // RuntimeState*
                                %arg1: !llvm.ptr,     // span_t* inputs
                                %arg2: !llvm.ptr) -> i32  // span_t* outputs
      attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    // Allocate memref struct holders
    %input_memref = llvm.alloca ...
    %output_memref = llvm.alloca ...

    // ✅ Prepare input: parse span_t, validate, alloc GPU, H2D transfer
    %status_in = llvm.call @hipdnn_ep_tensor_prepare_input(
        %arg0, %arg1, %c0_i64, %c4_i64, %input_memref) : (...) -> i32

    // ✅ Prepare output: parse span_t, alloc GPU (no H2D)
    %status_out = llvm.call @hipdnn_ep_tensor_prepare_output(
        %arg0, %arg2, %c0_i64, %c4_i64, %output_memref) : (...) -> i32

    // ✅ Call generated @main function
    %result = llvm.call @main(%arg0, %input_memref, %output_memref) : (...) -> i32

    // ✅ Finalize: D2H transfer, sync stream
    llvm.call @hipdnn_ep_tensor_finalize_output(...) : (...) -> i32

    // ✅ Free temporary GPU buffers (constants stay in RuntimeState)
    llvm.call @hipdnn_ep_tensor_free_input(...) : (...) -> ()
    llvm.return %result : i32
  }

  // ✅ EXPORT 3: Cleanup GPU state
  llvm.func @inference_cleanup(%arg0: !llvm.ptr) -> i32
      attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %result = llvm.call @hipdnn_ep_state_cleanup(%arg0) : (!llvm.ptr) -> i32
    llvm.return %result : i32
  }
}
```

**Design details:** [INTERFACE-DESIGN.md](../design/mlir/INTERFACE-DESIGN.md), [GenerateInterfacePass.md](../design/mlir/passes/GenerateInterfacePass.md)

### Stage 6: Native DLL Compilation

**Command:**
```bash
../../build/$(basename $PWD)/bin/Debug/mlir-hip-compiler.exe \
  tools/hip-opt/demos/demo_two_layer_conv.mlir \
  --from-onnx-mlir \
  -o ../output/demo_two_layer.dll \
  --mode dll \
  -v
```

**Output:**
```
=== MLIR to HIP DLL Compiler ===
Input: tools/hip-opt/demos/demo_two_layer_conv.mlir
Output: ../output/demo_two_layer.dll

--- Step 1: Parsing MLIR ---
✓ MLIR parsed successfully

--- Step 2: Running MLIR Passes ---
✓ MLIR passes completed

--- Step 3: Translating to LLVM IR ---
✓ LLVM IR generated

--- Step 3.5: Linking Runtime Module ---
✓ Runtime module linked (enables cross-module inlining)

--- Step 4: Optimizing LLVM IR (O2) ---
✓ Optimization completed (Runtime calls inlined)

--- Step 6: Compiling to Object File ---
✓ Object file created: ../output/demo_two_layer.obj

--- Step 7: Linking to DLL ---
✓ DLL created: ../output/demo_two_layer.dll

--- Step 8: Verifying DLL Exports ---
✓ All expected exports present

=== Compilation Successful ===
Output: ../output/demo_two_layer.dll
```

**What happens:**
- ✅ Complete pipeline: MLIR → LLVM IR → Object → DLL
- ✅ IR-level runtime merging (enables cross-module inlining)
- ✅ Optimization at -O2 level
- ✅ Verified DLL exports (init/compute/cleanup)
- ✅ Linked with amdhip64.lib, MIOpen.lib, hipblaslt.lib

**Design details:** [RUNTIME-ARCHITECTURE.md](../design/RUNTIME-ARCHITECTURE.md)

### Stage 7: End-to-End Testing

**Command:**
```bash
../../build/$(basename $PWD)/bin/Debug/test-model-dll.exe \
  ../output/demo_two_layer.dll
```

**Output:**
```
[MOCK] hipStreamCreate() -> 000002F070F36E20
[MOCK] miopenCreate() -> 000002F070F36D30
[MOCK] miopenSetStream(handle=000002F070F36D30, stream=000002F070F36E20)
[MOCK] hipblasLtCreate() -> 000002F070F366A0
[MOCK] hipMalloc(147456 bytes) -> 000002F070F40FE0
[MOCK] hipMemcpy(dst=000002F070F40FE0, src=00007FFBE6750080, size=147456, H2D)
[MOCK] hipMalloc(256 bytes) -> 000002F070F65020
[MOCK] hipMemcpy(dst=000002F070F65020, src=00007FFBE6774080, size=256, H2D)
[MOCK] hipMalloc(256 bytes) -> 000002F070F65160
[MOCK] hipMemcpy(dst=000002F070F65160, src=00007FFBE6774180, size=256, H2D)
[MOCK] hipMalloc(6912 bytes) -> 000002F070F3D810
[MOCK] hipMemcpy(dst=000002F070F3D810, src=00007FFBE6774280, size=6912, H2D)
[MOCK] hipMalloc(12845056 bytes) -> 000002F0711E0070
[MOCK] hipMalloc(602112 bytes) -> 000002F0721500B0
[MOCK] hipMemcpyAsync(dst=000002F0721500B0, src=000002F070F652C0, size=602112, H2D, stream=000002F070F36E20)
[MOCK] wrap_miopenActivationForward_relu(input=000002F0711E0070, output=000002F0711E0070)
[MOCK] wrap_miopenConvolutionForward(
[MOCK]   input=[1,64,224,224],
[MOCK]   weights=[64,64,3,3],
[MOCK]   output=[1,64,112,112],
[MOCK]   stride=[2,2], pad=[1,1,1,1], dilation=[1,1], group=1)
[MOCK] wrap_miopenActivationForward_relu(input=000002F0711E0070, output=000002F0711E0070)
```

**What happens:**
- ✅ DLL loads successfully
- ✅ All exports resolved (inference_init/compute/cleanup)
- ✅ Test execution completes
- ✅ Mock runtime shows GPU operations (no hardware needed)

---

## Try It Yourself

### Build the Tools

```bash
cd /path/to/onnx-hipdnn-ep

# Build tools (hip-opt, mlir-hip-compiler, test-model-dll)
cmake -S . -B ../../build/$(basename $PWD) -DBUILD_HIP_OPT_TOOL=ON -DBUILD_MLIR_HIP_COMPILER=ON
cmake --build ../../build/$(basename $PWD) --config Debug --target hip-opt mlir-hip-compiler test-model-dll
```

### Run Key Stages

```bash
# Stage 1: ONNX → HIP Dialect (redirect stderr to filter debug output)
../../build/$(basename $PWD)/bin/Debug/hip-opt.exe \
  tools/hip-opt/demos/demo_two_layer_conv.mlir \
  --convert-onnx-to-hip \
  2>/dev/null > ../output/stage1.mlir

# Stage 2-5: Complete pipeline (easiest approach - avoids debug output issues)
../../build/$(basename $PWD)/bin/Debug/hip-opt.exe \
  tools/hip-opt/demos/demo_two_layer_conv.mlir \
  --convert-onnx-to-hip \
  --ownership-based-buffer-deallocation \
  --memory-pooling \
  --convert-hip-to-llvm \
  --generate-interface \
  2>/dev/null > ../output/stage5.mlir

# Stage 6: Native DLL Compilation (runs full pipeline automatically)
../../build/$(basename $PWD)/bin/Debug/mlir-hip-compiler.exe \
  tools/hip-opt/demos/demo_two_layer_conv.mlir \
  --from-onnx-mlir \
  -o ../output/demo_two_layer.dll \
  --mode dll \
  -v \
  --keep

# Stage 7: End-to-End Testing
../../build/$(basename $PWD)/bin/Debug/test-model-dll.exe \
  ../output/demo_two_layer.dll
```

**Note**:
- Stage 6 with `--from-onnx-mlir` runs Stages 1-5 automatically
- Use `2>/dev/null` to filter debug output when saving intermediate MLIR files
- For viewing transformation results, chain multiple passes in one command

---

## Tools Reference

### hip-opt
MLIR transformation tool for testing individual passes.
- **Input**: ONNX/HIP MLIR
- **Output**: Transformed MLIR
- **Passes**: `--convert-onnx-to-hip`, `--ownership-based-buffer-deallocation`, `--memory-pooling`, `--convert-hip-to-llvm`, `--generate-interface`

### mlir-hip-compiler
End-to-end DLL compiler (runs full pipeline automatically).
- **Input**: MLIR (or ONNX-MLIR with `--from-onnx-mlir`)
- **Output**: Native DLL with C-ABI exports
- **Options**: `-o <output>`, `--mode <ir|object|dll>`, `-O <0-3>`, `-v`, `--keep`

### test-model-dll
DLL testing and validation tool.
- **Input**: Compiled model DLL
- **Output**: Test results (PASSED/FAILED)
- **Validates**: DLL loading, export resolution, inference execution
- **Note**: Uses mock runtime (no GPU required)

---

## Full Code Examples

**Input ONNX Model** (`tools/hip-opt/demos/demo_two_layer_conv.mlir`):
```mlir
// Two conv layers with ReLU activations and embedded constant weights/biases
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

  // ReLU activation after first conv
  %relu1 = "onnx.Relu"(%conv1) : (tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32>

  // Layer 2 constants
  %weights2 = "onnx.Constant"() {
    value = dense<2.0> : tensor<64x64x3x3xf32>
  } : () -> tensor<64x64x3x3xf32>
  %bias2 = "onnx.Constant"() {
    value = dense<0.1> : tensor<64xf32>
  } : () -> tensor<64xf32>

  // Layer 2: Conv (3x3, stride=2, halves dimensions)
  %conv2 = "onnx.Conv"(%relu1, %weights2, %bias2) {
    kernel_shape = [3, 3], strides = [2, 2],
    pads = [1, 1, 1, 1], dilations = [1, 1], group = 1 : si64
  } : (tensor<1x64x224x224xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>)
      -> tensor<1x64x112x112xf32>

  // ReLU activation after second conv
  %relu2 = "onnx.Relu"(%conv2) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>

  return %relu2 : tensor<1x64x112x112xf32>
}
```

Full intermediate outputs available in `../output/` directory.

---

## Maintenance Guidelines

⚠️ **CRITICAL**: This document is used for stakeholder presentations and live demos. Inaccurate examples damage credibility.

### Update Workflow

**When to update this document**: After any compiler changes, refactoring, or new features:

```
1. Make code changes
2. Test and commit code
3. Update DEMO.md ← LAST STEP (this document)
```

### How to Update (6 Rules)

**Rule 1: Regenerate Real Output**
```bash
cd /path/to/onnx-hipdnn-ep
# Run all stages, save to ../output/
../../build/$(basename $PWD)/bin/Debug/hip-opt.exe tools/hip-opt/demos/demo_two_layer_conv.mlir --convert-onnx-to-hip > ../output/stage1_onnx_to_hip.mlir
# ... (all stages)
```
Copy actual output into this document. Never fabricate examples.

**Rule 2: Avoid Numbers That Go Stale**
- ❌ "23 parameters", "150528 elements", "runs in 2.5 seconds"
- ✅ "unpacked memref parameters (count varies)", "tensor shape [1,3,224,224]"
- Exception: One concrete example per concept with "(demo model)" note

**Rule 3: Use Relative Paths**
- ❌ `C:/Develop/m/build/onnx-hipdnn-ep/bin/Debug/hip-opt.exe`
- ✅ `../../build/$(basename $PWD)/bin/Debug/hip-opt.exe`

**Rule 4: Make Commands Reproducible**
- Use `$(basename $PWD)` - works on any checkout
- Use `tools/hip-opt/demos/` - correct paths from project root
- Anyone should copy-paste and succeed

**Rule 5: Validate Everything** ⚠️ **MANDATORY**
```bash
# Before committing DEMO.md, run EVERY command in sequence
# Fix any that fail
# Update output files to match current compiler
```
**Broken demos in meetings are unacceptable.**

**Rule 6: Explain Key Points**
- Excerpt important parts of output (not full dumps)
- Add 1-2 sentence explanations of what each stage does
- Make it pedagogical for stakeholders

### Checklist Before Committing

- [ ] All commands run successfully
- [ ] Output files in `../output/` are current
- [ ] Paths use `$(basename $PWD)` pattern
- [ ] No hardcoded user-specific paths
- [ ] Examples match actual compiler output
- [ ] Key points are explained, not just dumped

**Bottom line**: Keep this document in sync with code reality. Test before presenting.
