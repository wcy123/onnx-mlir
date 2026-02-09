# MLIR Compilation Pipeline Demo

**Goal**: Compile ONNX Conv operation to AMD GPU native code.

**Pipeline**: `ONNX-MLIR → HIP Dialect → LLVM IR → Native DLL`

---

## Input: ONNX-MLIR Model

Real output from `onnx-mlir-opt model.onnx`:

```mlir
module {
  func.func @main_graph(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32> {
    %0 = "onnx.Constant"() {value = dense<1.0> : tensor<64x3x7x7xf32>} : () -> tensor<64x3x7x7xf32>
    %1 = "onnx.Constant"() {value = dense<0.5> : tensor<64xf32>} : () -> tensor<64xf32>

    %2 = "onnx.Conv"(%arg0, %0, %1) {
      kernel_shape = [7, 7],
      strides = [2, 2],
      pads = [3, 3, 3, 3],
      dilations = [1, 1],
      group = 1 : si64
    } : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>

    return %2 : tensor<1x64x112x112xf32>
  }
}
```

**Key points**:
- Standard onnx-mlir output - uses tensor types
- No GPU runtime context yet
- Input: 1×3×224×224 (batch, channels, height, width)
- Conv: 7×7 kernel, stride 2, padding 3
- Output: 1×64×112×112

---

## After `--convert-onnx-to-hip`

**⚠️ Status**: Pattern implemented, but needs fixes:
1. Add `%ctx: !hip.context` parameter during conversion
2. Handle tensor→memref conversion (bufferization)

**Expected output** (after fixes):

```mlir
module {
  // Context parameter added by conversion
  func.func @main_graph(%ctx: !hip.context, %arg0: tensor<1x3x224x224xf32>)
      -> tensor<1x64x112x112xf32> {

    %0 = "onnx.Constant"() {value = dense<1.0> : tensor<64x3x7x7xf32>} : () -> tensor<64x3x7x7xf32>
    %1 = "onnx.Constant"() {value = dense<0.5> : tensor<64xf32>} : () -> tensor<64xf32>

    // ONNX Conv → HIP Conv
    // Context passed as first argument
    %2 = hip.conv(%ctx, %arg0, %0, %1) {
      kernel_shape = [7, 7],
      strides = [2, 2],
      pads = [3, 3, 3, 3],
      dilations = [1, 1],
      group = 1 : i64
    } : (!hip.context, tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>)
        -> tensor<1x64x112x112xf32>

    return %2 : tensor<1x64x112x112xf32>
  }
}
```

**Changes**:
- Function signature: Added `%ctx: !hip.context` as first parameter
- Operation: `"onnx.Conv"(...)` → `hip.conv(%ctx, ...)`
- Attributes: Preserved (kernel_shape, strides, pads, etc.)
- Types: Still tensors (hip.conv currently expects memrefs - needs fix)

**Design Notes**:
- `!hip.context` is opaque pointer to runtime state (stream, miopenHandle, hipblasHandle, weights)
- Context extraction (getting miopenHandle) happens in HIP→LLVM lowering, not here
- All HIP operations take context as first argument

**TODO** (before this works):
1. Modify conversion pattern to add context parameter to function
2. Either:
   - Option A: Add bufferization pass to convert tensor→memref before HIP lowering
   - Option B: Modify HipOps.td to accept both tensor and memref types

---

## After Bufferization (Intermediate Step)

**Note**: This step is needed because HIP operations currently require memref types.

```mlir
module {
  func.func @main_graph(%ctx: !hip.context, %arg0: memref<1x3x224x224xf32>)
      -> memref<1x64x112x112xf32> {

    // Constants become memrefs
    %0 = memref.get_global @conv_weight : memref<64x3x7x7xf32>
    %1 = memref.get_global @conv_bias : memref<64xf32>

    // HIP operations with memref types
    %2 = hip.conv(%ctx, %arg0, %0, %1) {
      kernel_shape = [7, 7],
      strides = [2, 2],
      pads = [3, 3, 3, 3],
      dilations = [1, 1],
      group = 1 : i64
    } : (!hip.context, memref<1x3x224x224xf32>, memref<64x3x7x7xf32>, memref<64xf32>)
        -> memref<1x64x112x112xf32>

    return %2 : memref<1x64x112x112xf32>
  }
}
```

---

## After `--convert-hip-to-llvm`

```mlir
module {
  // MIOpen API declarations
  llvm.func @miopenCreateTensorDescriptor() -> !llvm.ptr
  llvm.func @miopenSetTensorDescriptor(!llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @miopenCreateConvolutionDescriptor() -> !llvm.ptr
  llvm.func @miopenSetConvolutionDescriptor(!llvm.ptr, i32, i32, i32, i32, i32, i32) -> i32
  llvm.func @miopenConvolutionForward(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.func @hipMalloc(i64) -> !llvm.ptr
  llvm.func @hipFree(!llvm.ptr)

  func.func @main_graph(%ctx: !llvm.ptr, %input: !llvm.ptr) -> !llvm.ptr {
    // Extract miopenHandle from context (offset 8 bytes = field 1)
    %c8 = llvm.mlir.constant(8 : i64) : i64
    %miopen_field_ptr = llvm.getelementptr %ctx[%c8] : (!llvm.ptr, i64) -> !llvm.ptr
    %miopen_handle = llvm.load %miopen_field_ptr : !llvm.ptr -> !llvm.ptr

    // Extract stream from context (offset 0 = field 0)
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %stream_field_ptr = llvm.getelementptr %ctx[%c0] : (!llvm.ptr, i64) -> !llvm.ptr
    %stream = llvm.load %stream_field_ptr : !llvm.ptr -> !llvm.ptr

    // Allocate GPU memory for output (1×64×112×112×4 bytes = 3,211,264 bytes)
    %output_size = llvm.mlir.constant(3211264 : i64) : i64
    %output_ptr = llvm.call @hipMalloc(%output_size) : (i64) -> !llvm.ptr

    // Create tensor descriptors
    %input_desc = llvm.call @miopenCreateTensorDescriptor() : () -> !llvm.ptr
    %weight_desc = llvm.call @miopenCreateTensorDescriptor() : () -> !llvm.ptr
    %output_desc = llvm.call @miopenCreateTensorDescriptor() : () -> !llvm.ptr
    %conv_desc = llvm.call @miopenCreateConvolutionDescriptor() : () -> !llvm.ptr

    // Set input descriptor: NCHW format, [1, 3, 224, 224]
    %dims_input = llvm.mlir.addressof @dims_1_3_224_224 : !llvm.ptr
    llvm.call @miopenSetTensorDescriptor(%input_desc, %float32, %dims_input, %strides_input) : (...)

    // Set weight descriptor: [64, 3, 7, 7]
    %dims_weight = llvm.mlir.addressof @dims_64_3_7_7 : !llvm.ptr
    llvm.call @miopenSetTensorDescriptor(%weight_desc, %float32, %dims_weight, %strides_weight) : (...)

    // Set output descriptor: [1, 64, 112, 112]
    %dims_output = llvm.mlir.addressof @dims_1_64_112_112 : !llvm.ptr
    llvm.call @miopenSetTensorDescriptor(%output_desc, %float32, %dims_output, %strides_output) : (...)

    // Set convolution descriptor: pad=3, stride=2, dilation=1
    %pad = llvm.mlir.constant(3 : i32) : i32
    %stride = llvm.mlir.constant(2 : i32) : i32
    %dilation = llvm.mlir.constant(1 : i32) : i32
    llvm.call @miopenSetConvolutionDescriptor(%conv_desc, %pad, %pad, %stride, %stride, %dilation, %dilation) : (...)

    // Execute convolution: output = Conv(input, weight, bias)
    %alpha = llvm.mlir.constant(1.0 : f32) : f32
    %beta = llvm.mlir.constant(0.0 : f32) : f32
    llvm.call @miopenConvolutionForward(
      %miopen_handle,
      %alpha, %input_desc, %input,
      %weight_desc, %weight_ptr,
      %conv_desc, %algo,
      %beta, %output_desc, %output_ptr,
      %workspace, %workspace_size
    ) : (...) -> i32

    // Cleanup descriptors
    llvm.call @miopenDestroyTensorDescriptor(%input_desc) : (!llvm.ptr) -> ()
    llvm.call @miopenDestroyTensorDescriptor(%weight_desc) : (!llvm.ptr) -> ()
    llvm.call @miopenDestroyTensorDescriptor(%output_desc) : (!llvm.ptr) -> ()
    llvm.call @miopenDestroyConvolutionDescriptor(%conv_desc) : (!llvm.ptr) -> ()

    return %output_ptr : !llvm.ptr
  }
}
```

**Key transformations**:
1. `!hip.context` → `!llvm.ptr` (opaque pointer)
2. Context field extraction:
   - Stream: `GEP %ctx[0]` (offset 0)
   - MIOpen handle: `GEP %ctx[8]` (offset 8, field 1)
3. `hip.conv()` → Direct MIOpen API calls:
   - Create descriptors
   - Configure convolution parameters
   - Execute `miopenConvolutionForward`
   - Cleanup resources

**Note**: Context field offsets match State struct layout:
```c
struct State {
  hipStream_t stream;         // offset 0
  miopenHandle_t miopen;      // offset 8  ← extracted here
  hipblasLtHandle_t hipblas;  // offset 16
  void** gpu_weights;         // offset 24
};
```

---

## Final: Native DLL

LLVM IR compiles to x86-64 machine code calling AMD GPU libraries:

```asm
inference_compute:
    push rbp
    mov rbp, rsp

    ; Extract miopenHandle from context (offset 8)
    mov rax, [rdi+8]        ; ctx->miopen

    ; Call miopenConvolutionForward
    mov rdi, rax            ; handle
    lea rsi, [rbp-32]       ; alpha
    mov rdx, [rbp-40]       ; input_desc
    mov rcx, [rbp-48]       ; input_ptr
    ; ... more arguments
    call miopenConvolutionForward@PLT

    ; Return status
    xor eax, eax            ; return 0 (success)
    pop rbp
    ret
```

**Result**:
- Native AMD GPU code
- No LLVM/MLIR runtime needed
- ~1-10ms load time from EPContext
- Ready for `dlsym("inference_compute")`

---

## Current Implementation Status

### ✅ Working: CLI Option Fix

**Fixed**: hip-opt now builds without CLI option conflicts
- Solution: Conditional CLI registration with `ONNX_MLIR_ENABLE_CLI_REGISTRATION`
- Status: Committed to fork (5ebbaa60)

### ✅ Working: HIP → LLVM

**Test**: `tools/hip-opt/test.mlir`
```bash
hip-opt test.mlir --convert-hip-to-llvm
```

**Output**: Verified LLVM IR with `hipMalloc`/`hipFree` calls

### ⚠️ Partially Working: ONNX → HIP

**Status**: Pattern compiles, but has runtime issues:

**Issue 1**: Context parameter not added
- Pattern expects `%ctx: !hip.context` as first function parameter
- Regular onnx-mlir models don't have this
- **Fix needed**: Modify pattern to add context parameter during conversion

**Issue 2**: Type mismatch (tensor vs memref)
- HIP operations expect `memref<>` types
- ONNX operations output `tensor<>` types
- Pattern currently creates `hip.conv` with tensors → type error
- **Fix needed**: Add bufferization pass or modify HipOps to accept tensors

**Test file**: `tools/hip-opt/demo_input.mlir`
```bash
# Current: Fails with "expected !hip.context parameter"
hip-opt demo_input.mlir --convert-onnx-to-hip
```

### 📋 Next Steps

**Phase 1: Fix ONNX → HIP Conversion**
1. Update pattern to add `%ctx: !hip.context` parameter
2. Add bufferization pass or modify type constraints
3. Test with demo_input.mlir

**Phase 2: Type System Updates**
1. Rename `!hip.handle` → `!hip.context` in HipDialect.td
2. Update all operation definitions
3. Update HipToLLVM.cpp with correct offset calculations

**Phase 3: End-to-End Test**
```
ONNX model → ONNX-MLIR → HIP → LLVM IR → DLL → EPContext
```

**Phase 4: Update Documentation**
- DEMO.md with real test outputs
- MLIR-COMPILATION-DESIGN.md with implementation details

---

## References

- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md) - Full system design
- **MLIR Lowering**: [MLIR-COMPILATION-DESIGN.md](MLIR-COMPILATION-DESIGN.md) - Detailed module structure
- **ONNX Integration**: [ONNX-MLIR-INTEGRATION.md](ONNX-MLIR-INTEGRATION.md) - Build setup
- **Fork Summary**: [ONNX_MLIR_FORK_SUMMARY.md](ONNX_MLIR_FORK_SUMMARY.md) - onnx-mlir fork changes
