# MLIR-Based AOT Compilation for AMD ROCm
## Technical Demo: ONNX → Native GPU Code

**What**: MLIR-based ahead-of-time (AOT) compilation pipeline that transforms ONNX models to native GPU code for AMD ROCm.

**Pipeline**: `ONNX → ONNX-MLIR → HIP Dialect → LLVM IR → Native DLL → EPContext`

---

## Demo: Two-Layer Convolution with Constant Weights

### Input: ONNX Model

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

### After `--convert-onnx-to-hip`

**Command**: `hip-opt test_constants.mlir --convert-onnx-to-hip`

```mlir
module {
  // ✅ Constants embedded as LLVM globals (will be in DLL .data section)
  llvm.mlir.global internal constant @constant_0(dense<1.0> : tensor<64x3x3x3xf32>)
    : !llvm.array<1728 x f32>
  llvm.mlir.global internal constant @constant_1(dense<0.5> : tensor<64xf32>)
    : !llvm.array<64 x f32>
  llvm.mlir.global internal constant @constant_2(dense<2.0> : tensor<64x64x3x3xf32>)
    : !llvm.array<36864 x f32>
  llvm.mlir.global internal constant @constant_3(dense<0.1> : tensor<64xf32>)
    : !llvm.array<64 x f32>

  // ✅ Main inference function - constants retrieved from GPU state
  func.func @main(%ctx: !hip.context,                      // GPU execution state
                  %input: memref<1x3x224x224xf32, 1>,      // GPU memory
                  %output: memref<1x64x112x112xf32, 1>)    // GPU memory
                  -> i32 {                                  // Status code

    // Get pre-uploaded constants from state (no allocation!)
    %c0 = arith.constant 0 : i64
    %weights1 = hip.get_constant(%ctx, %c0) : memref<64x3x3x3xf32, 1>
    %c1 = arith.constant 1 : i64
    %bias1 = hip.get_constant(%ctx, %c1) : memref<64xf32, 1>

    // Conv 1: In-place execution
    %temp1 = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
    hip.conv(%ctx, %input, %weights1, %bias1, %temp1)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]}

    // Get layer 2 constants
    %c2 = arith.constant 2 : i64
    %weights2 = hip.get_constant(%ctx, %c2) : memref<64x64x3x3xf32, 1>
    %c3 = arith.constant 3 : i64
    %bias2 = hip.get_constant(%ctx, %c3) : memref<64xf32, 1>

    // Conv 2: In-place execution
    %temp2 = hip.alloc(%ctx) : memref<1x64x112x112xf32, 1>
    hip.conv(%ctx, %temp1, %weights2, %bias2, %temp2)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [2, 2]}

    memref.copy %temp2, %output
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }

  // ✅ Initialization metadata function (returns constant count)
  llvm.func @get_constant_count() -> i64 {
    %0 = llvm.mlir.constant(4 : i64) : i64
    llvm.return %0 : i64
  }

  // ✅ Upload all constants to GPU during initialization
  func.func @initialize_constants(%ctx: !hip.context) -> i32 {
    %0 = llvm.mlir.addressof @constant_0 : !llvm.ptr
    %c0 = arith.constant 0 : i64
    %c6912 = arith.constant 6912 : i64  // 64×3×3×3 × 4 bytes
    hip.upload_constant(%ctx, %c0, %0, %c6912) : (!llvm.ptr)

    %1 = llvm.mlir.addressof @constant_1 : !llvm.ptr
    %c1 = arith.constant 1 : i64
    %c256 = arith.constant 256 : i64    // 64 × 4 bytes
    hip.upload_constant(%ctx, %c1, %1, %c256) : (!llvm.ptr)

    %2 = llvm.mlir.addressof @constant_2 : !llvm.ptr
    %c2 = arith.constant 2 : i64
    %c147456 = arith.constant 147456 : i64  // 64×64×3×3 × 4 bytes
    hip.upload_constant(%ctx, %c2, %2, %c147456) : (!llvm.ptr)

    %3 = llvm.mlir.addressof @constant_3 : !llvm.ptr
    %c3 = arith.constant 3 : i64
    hip.upload_constant(%ctx, %c3, %3, %c256) : (!llvm.ptr)

    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }

  // ✅ Free all GPU constant memory during cleanup
  func.func @release_constants(%ctx: !hip.context) -> i32 {
    %c0 = arith.constant 0 : i64
    hip.release_constant(%ctx, %c0)
    %c1 = arith.constant 1 : i64
    hip.release_constant(%ctx, %c1)
    %c2 = arith.constant 2 : i64
    hip.release_constant(%ctx, %c2)
    %c3 = arith.constant 3 : i64
    hip.release_constant(%ctx, %c3)
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}
```

### After `--convert-hip-to-llvm`

**Command**: `hip-opt test_constants.mlir --convert-onnx-to-hip --convert-hip-to-llvm`

Pure LLVM dialect output (key sections):

```mlir
module {
  // Runtime function declarations
  llvm.func @miopenConvolutionForward(!llvm.ptr, !llvm.ptr, ...) -> i32
  llvm.func @hipMalloc(i64) -> !llvm.ptr

  // Main function signature: memrefs unpacked to LLVM struct fields
  llvm.func @main(
    %arg0: !llvm.ptr,                    // context
    %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: i64,  // input memref fields
    %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64,        // input.sizes[4]
    %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64,      // input.strides[4]
    // ... weights, bias, output memref fields ...
  ) -> i32 {

    // Reconstruct memref descriptors from parameters
    %0 = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)>
    %1 = llvm.insertvalue %arg1, %0[0] : ...
    // ... (build all memref descriptors)

    // hip.alloc → hipMalloc + descriptor construction
    %52 = llvm.call @hipMalloc(%size) : (i64) -> !llvm.ptr
    %54 = llvm.addrspacecast %53 : !llvm.ptr to !llvm.ptr<1>
    // ... (build memref descriptor for allocated buffer)

    // hip.conv → miopenConvolutionForward with extracted pointers
    %68 = llvm.extractvalue %descriptor[1] : ...  // Extract aligned_ptr from input
    %69 = llvm.addrspacecast %68 : !llvm.ptr<1> to !llvm.ptr
    // ... (extract weights, bias, output pointers)
    %87 = llvm.call @miopenConvolutionForward(
      %arg0, %input_ptr, %weights_ptr, %bias_ptr, %output_ptr,
      %kernel_h, %kernel_w, %stride_h, %stride_w,
      %pad_top, %pad_left, %pad_bottom, %pad_right,
      %dilation_h, %dilation_w, %group
    ) : (!llvm.ptr, !llvm.ptr, ...) -> i32

    // memref.copy → llvm.intr.memcpy
    "llvm.intr.memcpy"(%dest_ptr, %src_ptr, %size)
      <{isVolatile = false}> : (!llvm.ptr<1>, !llvm.ptr<1>, i64) -> ()

    // Return success
    %107 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %107 : i32
  }
}
```

**Ready for compilation**:
- Pure LLVM dialect (no HIP, memref, or arith operations)
- Calls to MIOpen runtime functions
- Can be translated to LLVM IR via `mlir-translate --mlir-to-llvmir`
- LLVM IR compiles to native DLL

---

## Key Innovations

### 1. **Constant Handling**
- **4 onnx.Constant ops** → **4 LLVM globals** → embedded in DLL
- Uploaded to GPU **once** in `initialize_constants()`
- Retrieved via `hip.get_constant(ctx, index)` - zero overhead
- No constants in function signatures - scales to 1000+ layer models

### 2. **State-Based Architecture**
- Opaque `void* state` in C interface (backend-agnostic)
- Concrete `!hip.context` in MLIR (HIP-specific internals)
- Contains: GPU handles, pre-uploaded constant pointers, streams
- Clean separation of initialization vs. execution

### 3. **In-Place Semantics**
- Operations: `hip.conv(ctx, input, weights, bias, output)` - no return value
- Functions: Outputs as arguments, return i32 status code
- Matches GPU library APIs (MIOpen, hipBLAS) directly

### 4. **AOT Compilation**
- EPContext stores pre-compiled native DLL
- No LLVM/MLIR dependencies at runtime
- Compiled code loaded from memory (MemoryModule)

### 5. **Type Safety**
- ONNX-MLIR provides typed operations (`ONNXConvOp`, not string matching)
- Pattern matching at compile time (catches errors early)
- Semantic operand access (`convOp.getX()`, not `getOperand(0)`)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ONNX Model (Input)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                ┌────▼────────────────────────────────┐
                │  COMPILE TIME (Level-1 Pass)        │
                │  Dependencies: LLVM, MLIR, HIP      │
                ├─────────────────────────────────────┤
                │  1. ONNX → MLIR (onnx-mlir)        │
                │  2. Pattern lowering: ONNX → HIP    │
                │     • Discover constants            │
                │     • Generate LLVM globals         │
                │     • Create init/cleanup functions │
                │  3. HIP → LLVM lowering             │
                │  4. LLVM IR → Native DLL            │
                │  5. Embed DLL in EPContext          │
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

---

## For More Details

**Architecture & Design**:
- [ARCHITECTURE.md](ARCHITECTURE.md) - Complete system architecture, EPContext integration, interface design
- [MLIR-COMPILATION-DESIGN.md](MLIR-COMPILATION-DESIGN.md) - MLIR module structure, lowering pipeline, function designs
- [STATE-AND-CONTEXT.md](STATE-AND-CONTEXT.md) - State structure, lifecycle, naming conventions
- [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md) - Full constant handling design, all 6 phases

---

## Try It Yourself

```bash
# Build the compiler
cd /path/to/onnx-hipdnn-ep
cmake -S . -B ../../build/onnx-hipdnn-ep -DBUILD_HIP_OPT_TOOL=ON
cmake --build ../../build/onnx-hipdnn-ep --config Debug --target hip-opt

# Run ONNX → HIP transformation
../../build/onnx-hipdnn-ep/bin/hip-opt.exe \
  tools/hip-opt/test_constants.mlir \
  --convert-onnx-to-hip

# Run full pipeline: ONNX → HIP → LLVM
../../build/onnx-hipdnn-ep/bin/hip-opt.exe \
  tools/hip-opt/test_constants.mlir \
  --convert-onnx-to-hip \
  --convert-hip-to-llvm
```

---

## Current Status (2026-02-10)

✅ **Implemented**:
- ONNX → HIP conversion with pattern-based lowering
- HIP → LLVM lowering with runtime calls
- Constant handling (all 6 phases)
- Two-layer convolution demo

📋 **Next**:
- Runtime wrapper (miopenConvolutionForward)
- End-to-end integration
- ResNet50 support
