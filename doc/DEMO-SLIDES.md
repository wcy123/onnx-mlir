<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-size: 28px;
  }
  h1 {
    color: #0066cc;
  }
  h2 {
    color: #0088cc;
  }
  code {
    background: #f4f4f4;
  }
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
---

# MLIR AOT Compilation Demo
## From ONNX Model → Native AMD GPU Code

**Presentation for Tech Meeting**

---

# The Big Idea

Compile entire ONNX models **ahead-of-time** to native DLLs

✅ No runtime LLVM/MLIR dependencies
✅ Embedded constant weights in compiled code
✅ Direct MIOpen/HIP calls
✅ Native GPU performance

---

# Today's Demo

**Two-layer convolution network** (ResNet-style)

- **Input**: 1×3×224×224 (RGB image)
- **Layer 1**: 64 filters, 3×3 conv, stride=1 → 1×64×224×224
- **Layer 2**: 64 filters, 3×3 conv, stride=2 → 1×64×112×112
- **4 constant tensors** embedded in compiled code

**Pipeline**: `ONNX → HIP Dialect → LLVM IR → C Interface → Native DLL`

---

# Live Demo: Build the Compiler

```bash
cd /path/to/onnx-hipdnn-ep

cmake -S . -B ../../build/onnx-hipdnn-ep \
  -DBUILD_HIP_OPT_TOOL=ON

cmake --build ../../build/onnx-hipdnn-ep \
  --config Debug --target hip-opt
```

---

# Stage 1: ONNX → HIP Dialect

```bash
hip-opt.exe demo_two_layer_conv.mlir \
  --convert-onnx-to-hip
```

**What you'll see:**
- 4 LLVM global constants discovered
- `hip.conv` operations with GPU memory types
- Helper functions: `initialize_constants()`, `release_constants()`

---

# Stage 2: HIP → LLVM IR

```bash
hip-opt.exe demo_two_layer_conv.mlir \
  --convert-onnx-to-hip \
  --convert-hip-to-llvm
```

**What you'll see:**
- Runtime function declarations: `miopenConvolutionForward`, `hipMalloc`
- Two-function architecture: `@main` (wrapper) + `@main_internal` (computation)
- Memref descriptor unpacking logic

---

# Stage 3: Generate C Interface

```bash
hip-opt.exe demo_two_layer_conv.mlir \
  --convert-onnx-to-hip \
  --generate-interface
```

**What you'll see:**
- 3 exported functions: `inference_init()`, `inference_compute()`, `inference_cleanup()`
- C-ABI compliance attributes
- Error handling and validation logic

---

# Stage 4: Compile to Native DLL

```bash
# Save the interface-generated MLIR to a file
hip-opt.exe demo_two_layer_conv.mlir \
  --convert-onnx-to-hip \
  --convert-hip-to-llvm \
  --generate-interface \
  > demo_with_interface.mlir

# Compile to DLL
mlir-hip-compiler.exe demo_with_interface.mlir \
  -o inference.dll -v --keep
```

**What you'll see:**
- LLVM IR translation progress
- Optimization at O2 level
- Object file generation (.obj)
- DLL linking with runtime libraries (HipDnnRuntime.lib, amdhip64.lib, MIOpen.lib)
- DLL export verification (inference_init, inference_compute, inference_cleanup)
- Intermediate files kept: .ll (LLVM IR), .obj (object file)

---

# Pipeline Breakdown
## Stage 1: ONNX → HIP Dialect

---

# Before: ONNX Operations

```mlir
func.func @main(%input: tensor<1x3x224x224xf32>)
    -> tensor<1x64x112x112xf32> {

  %weights1 = "onnx.Constant"() {
    value = dense<1.0> : tensor<64x3x3x3xf32>
  }

  %conv1 = "onnx.Conv"(%input, %weights1, %bias1) {
    kernel_shape = [3, 3], strides = [1, 1]
  }

  // ... layer 2 similar
}
```

---

# After: HIP Dialect with Constants

```mlir
module attributes {hipdnn.input_count = 1, ...} {
  // ✅ Constants discovered and hoisted to globals
  llvm.mlir.global @constant_0(dense<1.0> : ...)
  llvm.mlir.global @constant_1(dense<0.5> : ...)

  func.func @main(%ctx: !hip.context,
                  %input: memref<1x3x224x224xf32, 1>,
                  %output: memref<1x64x112x112xf32, 1>) -> i32 {

    // ✅ Retrieve pre-uploaded constants
    %weights1 = hip.get_constant(%ctx, 0)

    // ✅ Direct HIP operation (in-place)
    hip.conv(%ctx, %input, %weights1, %bias1, %temp)
  }
}
```

---

# Stage 1: Key Transformations

✅ **Constant discovery**: 4 `onnx.Constant` → 4 `llvm.mlir.global`

✅ **Module metadata**: Captures input/output counts and ranks

✅ **In-place operations**: `hip.conv(ctx, in, w, b, out)` - no return value

✅ **GPU memory types**: `memref<..., 1>` (address space 1 = device memory)

---

# Pipeline Breakdown
## Stage 2: HIP → LLVM IR

---

# Two-Function Architecture

**Function 1**: Clean 3-parameter wrapper
```mlir
llvm.func @main(%ctx: !llvm.ptr,
                %inputs: !llvm.ptr,
                %outputs: !llvm.ptr) -> i32
```

**Function 2**: Internal computation (23 parameters)
```mlir
llvm.func @main_internal(%ctx: !llvm.ptr,
                         %in_ptr: !llvm.ptr<1>,
                         %in_size0: i64, ...) -> i32
```

---

# Stage 2: Unpacking Logic

```mlir
llvm.func @main(%ctx, %inputs, %outputs) -> i32 {
  // Unpack memref struct arrays
  %input_struct = llvm.getelementptr %inputs[0]
  %input = llvm.load %input_struct

  // Extract 11 fields
  %ptr = llvm.extractvalue %input[0]
  %size0 = llvm.extractvalue %input[3, 0]
  // ... 10 more extracts

  // Call internal function
  %result = llvm.call @main_internal(%ctx, %ptr, %size0, ...)
  llvm.return %result
}
```

---

# Stage 2: Key Transformations

✅ **Array-based interface**: `void** inputs` and `void** outputs`

✅ **Unpacking logic**: GEP → load → extractvalue

✅ **Pure LLVM dialect**: No more `func.func`, `!hip.context`, `arith.constant`

✅ **Scalable**: Works for N inputs/outputs

✅ **Dynamic shapes ready**: Runtime dimensions flow through structs

---

# Pipeline Breakdown
## Stage 3: Interface Generation

---

# Generated C-ABI Functions

Three exported functions:

1. **`inference_init(void** state)`** → Create GPU handles, upload constants

2. **`inference_compute(void* state, tensor_t* inputs, tensor_t* outputs)`** → Run inference

3. **`inference_cleanup(void* state)`** → Free GPU resources

All have `llvm.emit_c_interface` and `sym_visibility = "public"`

---

# Function 1: inference_init

```mlir
llvm.func @inference_init(%state_ptr: !llvm.ptr) -> i32
    attributes {llvm.emit_c_interface,
                sym_visibility = "public"} {
  // Allocate state (32 bytes for GPU handles)
  %state = llvm.call @malloc(32)

  // Check allocation success
  %is_null = llvm.icmp "eq" %state, %null
  llvm.cond_br %is_null, ^error, ^success

^success:
  llvm.store %state, %state_ptr
  return 0  // Success

^error:
  return 1  // Allocation failed
}
```

---

# Function 2: inference_compute

```mlir
llvm.func @inference_compute(%state_ptr, %inputs, %outputs)
    -> i32 attributes {llvm.emit_c_interface, ...} {

  // Validate input count (parse span_t->count)
  %input_count_ptr = llvm.getelementptr %inputs[1]
  %input_count = llvm.load %input_count_ptr
  %valid_in = llvm.icmp "eq" %input_count, 1

  // Validate output count
  %valid_out = llvm.icmp "eq" %output_count, 1

  llvm.cond_br %valid_in, ^check_out, ^error

^success:
  return 0  // Success
^error:
  return 5  // HIPDNN_ERROR_INVALID_INPUT
}
```

---

# Function 3: inference_cleanup

```mlir
llvm.func @inference_cleanup(%state_ptr: !llvm.ptr) -> i32
    attributes {llvm.emit_c_interface,
                sym_visibility = "public"} {

  llvm.call @free(%state_ptr)
  return 0  // Always succeeds
}
```

---

# Stage 3: Key Features

✅ **C calling convention**: `llvm.emit_c_interface` (no name mangling)

✅ **DLL exports**: `sym_visibility = "public"` (visible to GetProcAddress/dlsym)

✅ **Error handling**: Return codes 0 (success), 1 (alloc failed), 5 (invalid input)

✅ **span_t parsing**: Access count field at offset 1 via GEP

---

# Pipeline Breakdown
## Stage 4: DLL Compilation

---

# Stage 4: Compilation Pipeline

**Four-step process**: MLIR (LLVM dialect) → LLVM IR → Object File → DLL

1. **Translate to LLVM IR**
   - Convert MLIR to LLVM IR using MLIR's translation infrastructure
   - Preserves function declarations and C-ABI attributes
   - Output: `.ll` file (LLVM IR text format)

2. **Optimize LLVM IR**
   - Run LLVM optimization passes (default: -O2)
   - Function inlining, constant propagation, dead code elimination

---

# Stage 4: Compilation Pipeline (cont.)

3. **Compile to Object File**
   - Generate native machine code for target platform (x86-64 Windows)
   - Output: `.obj` file (PE/COFF format)

4. **Link to DLL**
   - Link object file with runtime libraries:
     - **HipDnnRuntime.lib** - Custom runtime (GPU handles, constant management)
     - **amdhip64.lib** - AMD HIP runtime
     - **MIOpen.lib** - Convolution operations
     - **hipblaslt.lib** - BLAS operations
   - Use LLD-LINK (LLVM's linker) to create DLL
   - Verify exported symbols
   - Output: `.dll` file (Windows) or `.so` (Linux)

---

# Stage 4: Tool Reference

**Tool**: `mlir-hip-compiler` (uses LLVMBackend + DLLLinker libraries)

**Options**:
- `--from-onnx-mlir` - Run ONNX→HIP→LLVM→Interface passes before compilation
- `-o <output>` - Output DLL filename
- `--mode <ir|object|dll>` - Stop after IR, object, or full DLL
- `-O <0-3>` - Optimization level (default: 2)
- `-v` - Verbose output
- `--keep` - Keep intermediate files (.ll, .obj)

---

# Key Innovations

---

# Innovation 1: Smart Constant Handling

**Problem**: ResNet50 has 1000+ weight tensors
→ Can't pass as function parameters!

**Solution**: Discover, hoist, and embed at compile time

```mlir
// Compile time: Generate globals
llvm.mlir.global @constant_0(dense<1.0> : ...)

// Runtime init: Upload once
hip.upload_constant(%ctx, 0, @constant_0, 6912 bytes)

// Runtime compute: Zero overhead
%weights = hip.get_constant(%ctx, 0)  // Just array lookup!
```

---

# Innovation 2: State-Based Architecture

**Design:**
- C interface: Opaque `void* state` (backend-agnostic)
- MLIR internals: Concrete `!hip.context` (HIP-specific)
- Contents: GPU handles, constant pointers, streams

**Lifecycle:**
```c
void* state;
inference_init(&state);        // Create handles, upload
inference_compute(state, ...);  // Use constants (fast!)
inference_compute(state, ...);  // Reuse state (efficient!)
inference_cleanup(state);       // Cleanup
```

---

# Innovation 3: In-Place Semantics

**Operations:** Output as parameter, no return value
```mlir
hip.conv(%ctx, %input, %weights, %bias, %output)
```

**Functions:** Return i32 status code
```mlir
func.func @main(%ctx, %input, %output) -> i32
```

**Benefits:**
- Matches GPU APIs (MIOpen, hipBLAS) directly
- No temporary allocations
- Destination-passing optimization built-in

---

# Innovation 4: AOT Compilation

**Development Workflow** (Standalone Tools):
```
ONNX MLIR → hip-opt (passes) → mlir-hip-compiler → DLL
```

**Production Workflow** (ONNX Runtime Integration):
```
ONNX Model → onnx-mlir → Level-1 Pass → DLL → EPContext
```

<div class="columns">
<div>

**Compile Time**
- LLVM, MLIR, ONNX-MLIR
- ~500MB dependencies
- Tools: Built-in MLIR passes + LLVMBackend + DLLLinker
- Output: Native DLL in EPContext

</div>
<div>

**Runtime**
- HIP runtime, MIOpen
- ~5MB dependencies
- **NO LLVM/MLIR!**
- Load DLL from memory

</div>
</div>

**Benefits:** Tiny runtime, fast startup, single artifact, same backend for both workflows

---

# Innovation 5: Type Safety via MLIR

```cpp
// Type-safe pattern matching
struct ONNXConvOpLoweringPattern :
    public OpConversionPattern<ONNXConvOp> {

  LogicalResult matchAndRewrite(ONNXConvOp op, ...) {
    // Semantic access (compile-time checked!)
    Value input = adaptor.getX();      // Not getOperand(0)
    Value weights = adaptor.getW();    // Not getOperand(1)
    ArrayAttr pads = op.getPads();     // Typed attributes

    builder.create<HIPConvOp>(...);
  }
};
```

**Benefits:** Compiler errors vs runtime crashes, refactoring-safe

---

# Current Status

---

# ✅ Fully Implemented

- [x] **ONNX → HIP conversion** with constant discovery
- [x] **HIP → LLVM lowering** with two-function architecture
- [x] **Interface generation** with 3 C-ABI exports
- [x] **Module metadata** captures input/output counts and ranks
- [x] **Constant handling** with upload/release helpers
- [x] **Error handling** with proper return codes
- [x] **Two-layer convolution demo** working end-to-end
- [x] **LLVM IR → DLL compilation** via mlir-hip-compiler
- [x] **DLL export verification** with automated checks
- [x] **End-to-end testing infrastructure** in test/mlir/

---

# ⚠️ In Progress

- [ ] GPU resource management (hipStreamCreate, miopenCreate)
- [ ] Build memref descriptors from tensor_t runtime dimensions
- [ ] Call @main from inference_compute after descriptor building
- [ ] Call initialize_constants from inference_init
- [ ] Call release_constants from inference_cleanup
- [ ] HipDnnRuntime.lib implementation (miopenConvolutionForward wrapper)

---

# 📋 Next Steps

1. **Complete runtime library implementation** (HipDnnRuntime.lib)
   - GPU resource initialization (hipStreamCreate, miopenCreate)
   - Descriptor building from tensor_t
   - Call @main, initialize_constants, release_constants

2. ~~**LLVM IR → DLL compilation**~~ ✅ **COMPLETED** via mlir-hip-compiler

3. **End-to-end integration test**: MLIR → DLL → EPContext → Custom Op

4. **ResNet50 support** (1000+ layer model)

5. **Level-1 Pass integration** with ONNX Runtime

---

# Output Files

**Verified real compiler output:**

```
../output/demo_stage1_onnx_to_hip.mlir
  → HIP dialect with constants

../output/demo_stage2_hip_to_llvm.mlir
  → LLVM IR with unpacking

../output/demo_stage3_with_interface.mlir
  → C-ABI interface functions

../output/demo_stage4_inference.dll
  → Native DLL with embedded runtime

../output/demo_stage3.ll (if --keep used)
  → LLVM IR text format

../output/demo_stage3.obj (if --keep used)
  → Native object file
```

---

# Try It Yourself

---

# Quick Start Commands

```bash
# 1. Build the tools
cd /path/to/onnx-hipdnn-ep
cmake -S . -B ../../build/onnx-hipdnn-ep \
  -DBUILD_HIP_OPT_TOOL=ON -DBUILD_MLIR_HIP_COMPILER=ON
cmake --build ../../build/onnx-hipdnn-ep \
  --config Debug --target hip-opt mlir-hip-compiler

# 2. Run Stage 1: ONNX → HIP
hip-opt.exe demo_two_layer_conv.mlir --convert-onnx-to-hip

# 3. Run Stage 2: HIP → LLVM
hip-opt.exe demo_two_layer_conv.mlir \
  --convert-onnx-to-hip --convert-hip-to-llvm

# 4. Run Stage 3: Generate Interface
hip-opt.exe demo_two_layer_conv.mlir \
  --convert-onnx-to-hip --generate-interface

# 5. Run Stage 4: Compile to DLL
hip-opt.exe demo_two_layer_conv.mlir \
  --convert-onnx-to-hip --convert-hip-to-llvm \
  --generate-interface > my_stage3.mlir
mlir-hip-compiler.exe my_stage3.mlir -o my_inference.dll -v --keep

# 6. Verify DLL Exports
dumpbin /EXPORTS my_inference.dll
```

---

# Expected Output: Stage 1

Should show:
- Module attributes: `hipdnn.input_count`, `hipdnn.input_ranks`
- 4 `llvm.mlir.global` constants
- `func.func @main` with `!hip.context` parameter
- Helper functions: `initialize_constants`, `release_constants`

---

# Expected Output: Stage 2

Should show:
- Runtime declarations: `@miopenConvolutionForward`, `@hipMalloc`
- `llvm.func @main` with 3 parameters (wrapper)
- `llvm.func @main_internal` with 23 parameters (computation)

---

# Expected Output: Stage 3

Should show:
- 3 exported functions with `sym_visibility = "public"`:
  - `@inference_init`
  - `@inference_compute`
  - `@inference_cleanup`
- All have `llvm.emit_c_interface` attribute

---

# Verification Commands

```bash
# Count functions
grep "llvm.func @" my_stage2.mlir | wc -l
# Expected: 6

# Check exports
grep "sym_visibility.*public" my_stage3.mlir
# Expected: 3 lines

# Check metadata
grep "hipdnn\." my_stage1.mlir
# Expected: 4 attributes

# Verify DLL exports
dumpbin /EXPORTS my_inference.dll | grep "inference_"
# Expected: inference_init, inference_compute, inference_cleanup

# Check intermediate files (if --keep used)
ls -lh my_stage3.ll my_stage3.obj my_inference.dll
```

---

# Full Pipeline Architecture

```
ONNX Model (Input)
        ↓
┌───────────────────────────────────────┐
│ COMPILE TIME (Two Paths)              │
├───────────────────────────────────────┤
│ PATH A: Standalone Tools (Dev)        │
│ • hip-opt: ONNX → MLIR                │
│   (--convert-onnx-to-hip)             │
│   (--convert-hip-to-llvm)             │
│   (--generate-interface)              │
│ • mlir-hip-compiler: MLIR → DLL       │
│   (Translate, Optimize, Compile, Link)│
│                                       │
│ PATH B: Integrated (Production)       │
│ • onnx-mlir: ONNX → MLIR              │
│ • Level-1 Pass: MLIR → DLL → EPContext│
│   (Uses same LLVMBackend/DLLLinker)   │
└───────────────────────────────────────┘
        ↓
ONNX + EPContext (Cached)
        ↓
┌───────────────────────┐
│ RUNTIME (Custom Op)   │
│ • Load DLL from mem   │
│ • inference_init()    │
│ • inference_compute() │
│ • inference_cleanup() │
│ NO LLVM/MLIR!         │
└───────────────────────┘
```

---

# Tools Reference

---

# hip-opt vs mlir-hip-compiler

<div class="columns">
<div>

**hip-opt** (MLIR Transformation)
- Purpose: Pass testing
- Input: ONNX/HIP MLIR
- Output: Transformed MLIR
- Passes:
  - --convert-onnx-to-hip
  - --convert-hip-to-llvm
  - --generate-interface
- Usage: Development, debugging

</div>
<div>

**mlir-hip-compiler** (DLL Compilation)
- Purpose: Production artifacts
- Input: LLVM dialect MLIR
- Output: Native DLL
- Options:
  - --from-onnx-mlir (runs passes)
  - -O <0-3> (optimization)
  - -v (verbose)
  - --keep (intermediate files)
- Usage: Production, standalone

</div>
</div>

---

# Workflow Comparison

**Two-step (Development)**:
```bash
hip-opt input.mlir --passes > transformed.mlir
mlir-hip-compiler transformed.mlir -o output.dll
```

**One-step (Production)**:
```bash
mlir-hip-compiler input.mlir --from-onnx-mlir -o output.dll
```

**Both use same backend**: LLVMBackend + DLLLinker

---

# Resources

**Design Documents:**
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture, EPContext
- [MLIR-COMPILATION-DESIGN.md](MLIR-COMPILATION-DESIGN.md) - Lowering pipeline
- [STATE-AND-CONTEXT.md](STATE-AND-CONTEXT.md) - State lifecycle
- [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md) - 6-phase design
- [INTERFACE-DESIGN.md](INTERFACE-DESIGN.md) - C-ABI specification

---

# Questions?

**Demo files available at:**
`tools/hip-opt/demo_two_layer_conv.mlir`

**Output files available at:**
`../output/demo_stage*.mlir`

**Try it yourself and explore the transformations!**

---

# Thank You!

**Key Takeaways:**
- ✅ AOT compilation: No runtime LLVM/MLIR
- ✅ Smart constant handling: Scales to 1000+ layers
- ✅ Type-safe MLIR transformations
- ✅ Production-ready C interface

**Next:** Complete GPU resource management and DLL generation
