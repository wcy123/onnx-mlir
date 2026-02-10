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

**Command**: `hip-opt demo_two_layer_conv.mlir --convert-onnx-to-hip`

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

**Command**: `hip-opt demo_two_layer_conv.mlir --convert-onnx-to-hip --convert-hip-to-llvm`

Pure LLVM dialect output (key sections):

```mlir
module {
  // Runtime function declarations
  llvm.func @miopenConvolutionForward(!llvm.ptr, !llvm.ptr, ...) -> i32
  llvm.func @hipMalloc(i64) -> !llvm.ptr
  llvm.func @hip_get_constant(!llvm.ptr, i64) -> !llvm.ptr

  // Constants embedded in DLL .data section
  llvm.mlir.global internal constant @constant_0(dense<1.0> : tensor<64x3x3x3xf32>)
    : !llvm.array<1728 x f32>
  llvm.mlir.global internal constant @constant_1(dense<0.5> : tensor<64xf32>)
    : !llvm.array<64 x f32>
  // ... (constant_2, constant_3)

  // Main function: memrefs unpacked to (ptr, ptr, offset, sizes[4], strides[4])
  llvm.func @main(%arg0: !llvm.ptr,  /* context */
                  /* input: 11 params */ ...,
                  /* output: 11 params */ ...) -> i32 {

    // 1. Reconstruct input/output memref descriptors from parameters
    %input_desc = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, ...)>
    %input_desc = llvm.insertvalue %arg1, %input_desc[0] : ...
    // ... (build complete descriptor)

    // 2. Get pre-uploaded constants from GPU state
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %weights1_ptr = llvm.call @hip_get_constant(%arg0, %c0)
    %weights1_gpu = llvm.addrspacecast %weights1_ptr : !llvm.ptr to !llvm.ptr<1>
    // ... (build memref descriptor for weights1)

    %c1 = llvm.mlir.constant(1 : i64) : i64
    %bias1_ptr = llvm.call @hip_get_constant(%arg0, %c1)
    // ... (build descriptor)

    // 3. hip.alloc → hipMalloc
    %size = llvm.mlir.constant(3211264 : i64) : i64  // temp buffer
    %ptr = llvm.call @hipMalloc(%size) : (i64) -> !llvm.ptr
    %gpu_ptr = llvm.addrspacecast %ptr : !llvm.ptr to !llvm.ptr<1>
    // ... (build descriptor)

    // 4. hip.conv → miopenConvolutionForward
    %input_ptr = llvm.extractvalue %input_desc[1] : ...
    %weights_ptr = llvm.extractvalue %weights_desc[1] : ...
    %bias_ptr = llvm.extractvalue %bias_desc[1] : ...
    %output_ptr = llvm.extractvalue %output_desc[1] : ...

    llvm.call @miopenConvolutionForward(
      %arg0, %input_ptr, %weights_ptr, %bias_ptr, %output_ptr,
      3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ) : (!llvm.ptr, !llvm.ptr, ...) -> i32

    // 5. Get layer 2 constants and repeat
    // ... (similar pattern for conv2)

    // 6. memref.copy → llvm.intr.memcpy
    "llvm.intr.memcpy"(%dest, %src, %size) : ...

    llvm.return %c0 : i32
  }

  // Initialization functions (from constant handling)
  llvm.func @get_constant_count() -> i64 {
    %0 = llvm.mlir.constant(4 : i64) : i64
    llvm.return %0 : i64
  }

  llvm.func @initialize_constants(%arg0: !llvm.ptr) -> i32 {
    %0 = llvm.mlir.addressof @constant_0 : !llvm.ptr
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c6912 = llvm.mlir.constant(6912 : i64) : i64
    llvm.call @hip_upload_constant(%arg0, %c0, %0, %c6912)
    // ... (upload constant_1, constant_2, constant_3)
    llvm.return %c0_i32 : i32
  }

  llvm.func @release_constants(%arg0: !llvm.ptr) -> i32 {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @hip_release_constant(%arg0, %c0)
    // ... (release all constants)
    llvm.return %c0_i32 : i32
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
  tools/hip-opt/demo_two_layer_conv.mlir \
  --convert-onnx-to-hip

# Run full pipeline: ONNX → HIP → LLVM
../../build/onnx-hipdnn-ep/bin/hip-opt.exe \
  tools/hip-opt/demo_two_layer_conv.mlir \
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
