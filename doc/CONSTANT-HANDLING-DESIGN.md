# Constant Handling Design
## ONNX Initializers in MLIR-based Compilation Pipeline

**Status**: Design in Progress
**Date**: 2026-02-10
**Related**: ARCHITECTURE.md, MLIR-COMPILATION-DESIGN.md

---

## Problem Statement

In a typical deep learning model (e.g., ResNet50), there are 100+ convolutional layers, each with weights and biases. In ONNX-MLIR, these appear as `onnx.Constant` operations within the function body:

```mlir
func.func @main_graph(%input: tensor<1x3x224x224xf32>) -> tensor<...> {
  %w0 = "onnx.Constant"() {value = dense<...> : tensor<64x3x3x3xf32>}
  %b0 = "onnx.Constant"() {value = dense<...> : tensor<64xf32>}
  %conv0 = "onnx.Conv"(%input, %w0, %b0) {...}

  %w1 = "onnx.Constant"() {value = dense<...> : tensor<128x64x3x3xf32>}
  %b1 = "onnx.Constant"() {value = dense<...> : tensor<128xf32>}
  %conv1 = "onnx.Conv"(%conv0, %w1, %b1) {...}

  // ... 98 more Conv operations with their constants
}
```

**Naive approach problem**: If we treat constants as function arguments during ONNX→HIP conversion, we end up with:

```mlir
func.func @main(%ctx: !hip.context,
                %input: memref<...>,
                %w0: memref<...>, %b0: memref<...>,
                %w1: memref<...>, %b1: memref<...>,
                // ... 196 more constant arguments
                %output: memref<...>) -> i32
```

This creates **200+ function arguments**, which is:
- ❌ Unmaintainable
- ❌ Inefficient (calling convention overhead)
- ❌ Doesn't match the execution model (constants are loaded once, reused many times)

---

## Proposed Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│  ONNX Model (model.onnx)                                     │
│  - graph.initializer[] contains constant data                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  ONNX-MLIR Import                                            │
│  - Initializers → onnx.Constant operations in func.func      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  ConvertOnnxToHipPass (MODULE-level)                         │
│                                                               │
│  Phase 1: Discovery                                          │
│  - Walk all func.func in module                              │
│  - Identify ONNX functions (have tensor types + ONNX ops)    │
│  - Discover all onnx.Constant operations                     │
│  - Assign global indices: 0, 1, 2, ..., N-1                 │
│                                                               │
│  Phase 2: Generate LLVM Globals                              │
│  - llvm.mlir.global @weight_0, @bias_0, @weight_1, ...      │
│  - Embed dense<...> constant data in DLL                     │
│                                                               │
│  Phase 3: Generate Initialization Functions                  │
│  - llvm.func @get_constant_count() -> i64                    │
│  - llvm.func @initialize_constants(state_ptr) -> i32         │
│  - llvm.func @release_constants(state_ptr) -> i32            │
│                                                               │
│  Phase 4: Transform ONNX Functions                           │
│  - Convert onnx.Constant to load from state                  │
│  - Remove constants from function arguments                  │
│  - Add %ctx parameter                                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Generated LLVM IR (in compiled DLL)                         │
│                                                               │
│  - Global constants (CPU memory)                             │
│  - @get_constant_count() -> 200                              │
│  - @initialize_constants(state*) -> uploads to GPU           │
│  - @release_constants(state*) -> frees GPU memory            │
│  - @main(state*, input*, output*) -> uses gpu_weights[]     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Runtime: inference_init()                                   │
│                                                               │
│  1. count = get_constant_count()          // 200             │
│  2. state->gpu_weights = new void*[count]                    │
│  3. initialize_constants(state)           // Upload to GPU   │
│  4. return state                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Design Details

### 1. Module-Level Pass

The `ConvertOnnxToHipPass` must operate at module level to handle constants properly:

```cpp
class ConvertOnnxToHipPass
    : public PassWrapper<ConvertOnnxToHipPass, OperationPass<ModuleOp>> {

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Capabilities:
    // - Walk all functions
    // - Create module-level globals
    // - Generate module-level functions
    // - Maintain global constant registry
  }
};
```

**Why module-level?**
- Need to discover constants across ALL functions (including subgraphs)
- Need to create module-level `llvm.mlir.global` operations
- Need to generate module-level initialization functions
- Need shared constant registry with global indexing

### 2. ONNX Function Identification

Not all `func.func` in the module are ONNX-MLIR functions. We need to identify and process only ONNX functions:

```cpp
bool isOnnxFunction(func::FuncOp funcOp) {
  auto funcType = funcOp.getFunctionType();

  // Quick filter: ONNX functions use tensor types
  bool hasTensorTypes = llvm::any_of(funcType.getInputs(), [](Type t) {
    return isa<TensorType>(t);
  }) || llvm::any_of(funcType.getResults(), [](Type t) {
    return isa<TensorType>(t);
  });

  if (!hasTensorTypes)
    return false;

  // Confirm: must have ONNX dialect operations
  bool hasOnnxOps = false;
  funcOp.walk([&](Operation *op) {
    if (auto *dialect = op->getDialect()) {
      if (isa<ONNXDialect>(dialect)) {
        hasOnnxOps = true;
        return WalkResult::interrupt();
      }
    }
  });

  return hasOnnxOps;
}
```

**Benefits:**
- ✅ Pass can coexist with other MLIR passes
- ✅ Order-independent in pass manager
- ✅ Won't break non-ONNX functions

### 3. Constant Discovery and Index Assignment

**Strategy**: Sequential order across all ONNX functions in the module.

```cpp
struct ConstantInfo {
  size_t globalIndex;      // Index in state->gpu_weights[]
  ElementsAttr value;      // The dense<...> constant data
  Type type;               // tensor<64x3x3x3xf32>
  size_t sizeInBytes;      // For allocation/transfer
  StringRef name;          // Generated name (e.g., "weight_0")
};

DenseMap<Value, ConstantInfo> constantRegistry;
size_t nextGlobalIndex = 0;

// Walk all ONNX functions
for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
  if (!isOnnxFunction(funcOp))
    continue;

  // Discover constants in this function
  funcOp.walk([&](ONNXConstantOp constOp) {
    ConstantInfo info;
    info.globalIndex = nextGlobalIndex++;
    info.value = constOp.getValueAttr().cast<ElementsAttr>();
    info.type = constOp.getType();
    info.sizeInBytes = calculateSizeInBytes(info.type);
    info.name = "constant_" + std::to_string(info.globalIndex);

    constantRegistry[constOp.getResult()] = info;
  });
}
```

**Future optimization (TODO)**: Deduplication - if the same constant appears in multiple places, share GPU memory.

### 4. Generated LLVM Globals

For each discovered constant, generate an `llvm.mlir.global`:

```mlir
// Example: weight tensor with 64*3*3*3 = 1728 elements
llvm.mlir.global internal constant @constant_0(dense<[1.0, 2.0, ...]> : tensor<64x3x3x3xf32>)
  : !llvm.array<1728 x f32>

llvm.mlir.global internal constant @constant_1(dense<[0.5, 0.5, ...]> : tensor<64xf32>)
  : !llvm.array<64 x f32>

// ... 198 more globals
```

**Implementation:**
```cpp
OpBuilder builder(moduleOp.getBodyRegion());

for (auto& [value, info] : constantRegistry) {
  // Create global constant with embedded data
  auto globalOp = builder.create<LLVM::GlobalOp>(
    moduleOp.getLoc(),
    convertTypeToLLVM(info.type),
    /*isConstant=*/true,
    LLVM::Linkage::Internal,
    info.name,
    info.value  // Embed dense<...> data
  );
}
```

### 5. HIP Dialect Operations for Constants

Three operations are defined in HIP dialect for constant management:

```tablegen
// Get reference to pre-uploaded constant (used in @main)
def Hip_GetConstantOp : Hip_Op<"get_constant"> {
  let arguments = (ins Hip_ContextType:$ctx, I64:$index);
  let results = (outs AnyMemRef:$result);
  let summary = "Get reference to pre-uploaded constant from state";
}

// Upload constant data to GPU (used in @initialize_constants)
def Hip_UploadConstantOp : Hip_Op<"upload_constant"> {
  let arguments = (ins Hip_ContextType:$ctx, I64:$index,
                       LLVM_AnyPointer:$cpu_data, I64:$size);
  let summary = "Upload constant data to GPU and store in state->gpu_weights[index]";
}

// Release constant from GPU (used in @release_constants)
def Hip_ReleaseConstantOp : Hip_Op<"release_constant"> {
  let arguments = (ins Hip_ContextType:$ctx, I64:$index);
  let summary = "Free GPU memory for constant at state->gpu_weights[index]";
}
```

**Semantics at HIP dialect level:**
- `hip.upload_constant`: Allocate GPU memory, copy data from CPU, store pointer in state
- `hip.release_constant`: Free GPU memory for this constant
- `hip.get_constant`: Return memref descriptor referencing GPU memory

**Lowering to LLVM** (deferred to HIP→LLVM pass):
- Naive: individual `hipMalloc` + `hipMemcpy` per constant
- Optimized: batch allocation, memory pooling, etc.

### 6. Generated Initialization Functions

Three functions are generated in HIP dialect:

```mlir
// 1. Query constant count (pure LLVM, no HIP ops needed)
llvm.func @get_constant_count() -> i64 {
  %count = llvm.mlir.constant(200 : i64) : i64
  llvm.return %count : i64
}

// 2. Upload all constants to GPU
func.func @initialize_constants(%ctx: !hip.context) -> i32 {
  // Constant 0: weights
  %data_0 = llvm.mlir.addressof @constant_0 : !llvm.ptr
  %size_0 = llvm.mlir.constant(6912 : i64) : i64  // 64*3*3*3*sizeof(float)
  %index_0 = llvm.mlir.constant(0 : i64) : i64
  hip.upload_constant(%ctx, %index_0, %data_0, %size_0)

  // Constant 1: bias
  %data_1 = llvm.mlir.addressof @constant_1 : !llvm.ptr
  %size_1 = llvm.mlir.constant(256 : i64) : i64
  %index_1 = llvm.mlir.constant(1 : i64) : i64
  hip.upload_constant(%ctx, %index_1, %data_1, %size_1)

  // ... repeat for all 200 constants

  %success = llvm.mlir.constant(0 : i32) : i32
  return %success : i32
}

// 3. Release all constants
func.func @release_constants(%ctx: !hip.context) -> i32 {
  %index_0 = llvm.mlir.constant(0 : i64) : i64
  hip.release_constant(%ctx, %index_0)

  %index_1 = llvm.mlir.constant(1 : i64) : i64
  hip.release_constant(%ctx, %index_1)

  // ... repeat for all 200 constants

  %success = llvm.mlir.constant(0 : i32) : i32
  return %success : i32
}
```

### 7. How @main Accesses Constants

**After ONNX→HIP conversion**, `@main` uses `hip.get_constant`:

```mlir
func.func @main(%ctx: !hip.context,
                %input: memref<1x3x224x224xf32, 1>,
                %output: memref<1x64x224x224xf32, 1>) -> i32 {

  // Get pre-uploaded constants (already on GPU)
  %weights = hip.get_constant(%ctx, 0) : (!hip.context, i64) -> memref<64x3x3x3xf32, 1>
  %bias = hip.get_constant(%ctx, 1) : (!hip.context, i64) -> memref<64xf32, 1>

  // Allocate output buffer
  %temp = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>

  // Convolution using pre-uploaded weights
  hip.conv(%ctx, %input, %weights, %bias, %temp) {...}

  // Copy to output
  memref.copy %temp, %output

  %success = llvm.mlir.constant(0 : i32) : i32
  return %success : i32
}
```

**Key insight**: Constants are accessed via `hip.get_constant`, NOT passed as function arguments. This keeps the function signature clean regardless of model size.

### 8. Runtime C Interface

**State Structure**:
```c
struct State {
    hipStream_t stream;
    miopenHandle_t miopenHandle;
    hipblasLtHandle_t hipblasHandle;
    void** gpu_weights;  // Array of GPU pointers (pre-uploaded constants)
};
```

**Runtime calls generated functions**:
```c
extern "C" int64_t get_constant_count();
extern "C" int initialize_constants(void* state_ptr);
extern "C" int release_constants(void* state_ptr);

int inference_init(void** state_ptr) {
    State* state = new State();
    state->gpu_weights = new void*[get_constant_count()];
    initialize_constants(state);  // Upload to GPU
    *state_ptr = state;
    return 0;
}

int inference_release(void* state_ptr) {
    release_constants(state_ptr);  // Free GPU memory
    delete state;
    return 0;
}
```

---

## Resolved Design Questions

### Question 2: How do ONNX functions access constants? ✅ RESOLVED

**Decision**: Use `hip.get_constant` operation in HIP dialect.

**Rationale**:
- Maintains clean abstraction: ONNX→HIP stays in HIP dialect
- Explicit semantics: "get reference to pre-uploaded constant"
- Flexible lowering: HIP→LLVM can optimize implementation
- Not ambiguous: clearly retrieves from state, doesn't upload

See Section 7 for usage example.

### Question 3: How do function calls handle constants?

**Context**: ONNX subgraphs (from If, Loop, Scan) become separate `func.func` that may reference constants.

**Example:**
```mlir
func.func @main_graph(%input: tensor<...>) -> tensor<...> {
  %w0 = "onnx.Constant"() {value = dense<...>}
  %result = func.call @subgraph_if_then(%input, %w0)
  ...
}

func.func @subgraph_if_then(%arg0: tensor<...>, %arg1: tensor<...>) -> tensor<...> {
  %w1 = "onnx.Constant"() {value = dense<...>}
  %conv = "onnx.Conv"(%arg0, %arg1, %w1)
  ...
}
```

**Option A: Pass %ctx through call chain**
```mlir
// After conversion
func.func @main(%ctx: !hip.context, %input: memref<...>, %output: memref<...>) -> i32 {
  %temp = ...
  func.call @subgraph_if_then(%ctx, %input, %temp)
  // Subgraph loads its constants from %ctx
}

func.func @subgraph_if_then(%ctx: !hip.context, %arg0: memref<...>, %arg1: memref<...>) -> ... {
  %w1 = hip.load_weight(%ctx, 5)  // Load from state
  ...
}
```

Constants passed as arguments are REMOVED - subgraphs load from state directly.

**Option B: Pass both %ctx and pre-loaded constants**
```mlir
func.func @main(%ctx: !hip.context, %input: memref<...>, %output: memref<...>) -> i32 {
  %w0 = hip.load_weight(%ctx, 0)
  %temp = ...
  func.call @subgraph_if_then(%ctx, %input, %w0, %temp)
}
```

But this reintroduces constant arguments...

**TODO**: Decide which approach to use.

---

## Implementation Phases

### Phase 1: Module-Level Pass Infrastructure
- [ ] Convert `ConvertOnnxToHipPass` from function-level to module-level
- [ ] Implement `isOnnxFunction()` helper
- [ ] Test: Pass can process multiple functions in module
- [ ] Test: Pass skips non-ONNX functions

### Phase 2: Constant Discovery
- [ ] Implement constant registry (DenseMap)
- [ ] Walk all ONNX functions and discover `onnx.Constant` operations
- [ ] Assign sequential global indices
- [ ] Calculate sizes in bytes
- [ ] Test: Registry correctly tracks all constants

### Phase 3: LLVM Global Generation
- [ ] Generate `llvm.mlir.global` for each constant
- [ ] Embed `dense<...>` constant data
- [ ] Test: Globals appear in LLVM IR output

### Phase 4: Initialization Functions
- [ ] Generate `@get_constant_count()` function
- [ ] Generate `@initialize_constants()` function
  - Extract state->gpu_weights
  - For each constant: hipMalloc, hipMemcpy, store pointer
- [ ] Generate `@release_constants()` function
  - For each constant: load pointer, hipFree
- [ ] Test: Functions compile and link

### Phase 5: Constant Access (TBD - depends on Question 2)
- [ ] Decide: `hip.load_weight` operation vs direct LLVM
- [ ] Implement constant loading in `@main`
- [ ] Test: Convolution uses loaded weights

### Phase 6: Function Call Handling (TBD - depends on Question 3)
- [ ] Decide: ctx-only vs ctx+constants
- [ ] Implement subgraph conversion
- [ ] Test: Multi-function models work

### Phase 7: Integration
- [ ] Update `inference_init()` to call generated functions
- [ ] Update `inference_release()` to call cleanup
- [ ] End-to-end test with ResNet50

---

## Future Optimizations (TODO)

1. **Constant Deduplication**: If the same constant appears multiple times, share GPU memory
2. **Lazy Loading**: Only upload constants that are actually used
3. **Compression**: Compress constant data in DLL, decompress during upload
4. **Quantization**: Support INT8/INT4 quantized constants
5. **Cached Upload**: Cache upload once across multiple model instances

---

## References

- MLIR Module-Level Passes: https://mlir.llvm.org/docs/PassManagement/
- LLVM GlobalOp: https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlir-global
- HIP Runtime API: https://rocm.docs.amd.com/projects/HIP/

---

**Document Status**: In Progress - Question 3 remains open
