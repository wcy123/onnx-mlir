<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Architecture Design

**MLIR-based AOT Compilation with EPContext for AMD ROCm**

**Version:** 2.0
**Date:** 2026-02-11
**Branch:** `mlir-integration`

---

## Table of Contents

- [System Context](#system-context)
- [System Architecture](#system-architecture)
- [Key Design Decisions](#key-design-decisions)
  - [1. Native DLL vs LLVM IR Storage](#1-native-dll-vs-llvm-ir-storage)
  - [2. Memory DLL Loading vs Disk Files](#2-memory-dll-loading-vs-disk-files)
  - [3. Pattern-Based Lowering vs Manual Transformation](#3-pattern-based-lowering-vs-manual-transformation)
  - [4. Stateful Interface (init/compute/cleanup)](#4-stateful-interface-initcomputecleanup)
  - [5. Embedded Constants vs External Files](#5-embedded-constants-vs-external-files)
  - [6. Synchronous Execution vs Async](#6-synchronous-execution-vs-async)
  - [7. Full Model Fusion vs Per-Op Execution](#7-full-model-fusion-vs-per-op-execution)
  - [8. Standalone Resources vs Shared Context](#8-standalone-resources-vs-shared-context)
- [Quality Attributes](#quality-attributes)
- [Design Principles](#design-principles)
- [ONNX-MLIR Integration](#onnx-mlir-integration)
- [References](#references)

---

## System Context

### Problem Statement

[ONNX Runtime](https://onnxruntime.ai/) execution with JIT compilation incurs significant overhead:
- First inference startup: 100-500ms compilation delay
- Runtime dependencies: 50-200 MB LLVM/MLIR libraries
- Repeated compilation: Same model recompiled on every process start

### Solution Architecture

Ahead-of-time (AOT) compilation to native GPU code stored in ONNX Runtime's [EPContext](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html):
- **Compile once** at model load/conversion time → native DLL
- **Store in EPContext** embedded in ONNX model (industry standard: TensorRT EP, QNN EP, VitisAI EP)
- **Load from memory** at runtime using [MemoryModule](https://github.com/fancycode/MemoryModule) (~50KB dependency)

### Success Metrics

- Load time: <10ms (vs 100-500ms JIT compilation)
- Runtime size: ~5MB (vs 50-200MB with LLVM/MLIR)
- Compilation elimination: Zero JIT overhead on inference startup

### Stakeholders

- **Primary:** [ONNX Runtime](https://onnxruntime.ai/) users deploying AMD ROCm models
- **Secondary:** ROCm ecosystem (demonstrates MLIR adoption for GPU inference)

---

## System Architecture

### Two-Stage Flow

```
┌─────────────────────────────────────────────────────────────┐
│               COMPILE-TIME (Level-1 Pass)                    │
├─────────────────────────────────────────────────────────────┤
│  Dependencies: LLVM, MLIR, onnx-mlir, HIP headers, MIOpen   │
│                                                              │
│  ONNX Model                                                 │
│      ↓                                                       │
│  ONNX → MLIR (onnx-mlir)                                    │
│      ↓                                                       │
│  ONNX dialect → HIP dialect (OnnxToHip Pass)                │
│      ↓                                                       │
│  HIP dialect → LLVM dialect (HipToLLVM Pass)                │
│      ↓                                                       │
│  Generate C interface (GenerateInterfacePass)               │
│      ↓                                                       │
│  LLVM IR → Native DLL                                       │
│      ↓                                                       │
│  Embed DLL in EPContext → ONNX model with EPContext         │
└─────────────────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                  RUNTIME (Custom Op)                         │
├─────────────────────────────────────────────────────────────┤
│  Dependencies: MemoryModule (~50KB), HIP runtime, MIOpen    │
│  NO LLVM/MLIR at runtime                                    │
│                                                              │
│  Load EPContext → Extract DLL bytes → Load DLL from memory  │
│      ↓                                                       │
│  Resolve entry points (inference_init/compute/cleanup)      │
│      ↓                                                       │
│  Execute GPU inference → Return results                     │
│                                                              │
│  Performance: ~1-10ms load time, zero compilation overhead  │
└─────────────────────────────────────────────────────────────┘
```

### Component Dependencies

- **Level-1 Pass** produces native DLL, depends on [MLIR](https://mlir.llvm.org/) infrastructure
- **Custom Op** consumes DLL from [EPContext](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html), zero MLIR dependencies
- **Interface contract:** C ABI with 3 functions (see [Design Decision #4](#4-stateful-interface-initcomputecleanup))

---

## Key Design Decisions

### 1. Native DLL vs LLVM IR Storage

**Decision:** Compile to native machine code (DLL) stored in [EPContext](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html), not LLVM IR or MLIR bytecode.

**Rationale:**
- Zero JIT overhead: Compilation happens once, not on every inference startup
- Lightweight runtime: No LLVM libraries (50-200 MB savings)
- Fast loading: ~1-10ms to load DLL vs 100-500ms for JIT compilation
- EPContext philosophy: Purpose is to eliminate recompilation

**Trade-offs:**

| Aspect | Native DLL | LLVM IR |
|--------|-----------|---------|
| **Startup time** | ~1-10ms (load from memory) | ~100-500ms (JIT compilation) |
| **Runtime deps** | [50KB MemoryModule](https://github.com/fancycode/MemoryModule) | 50-200MB LLVM/MLIR |
| **Portability** | GPU arch-specific (gfx1150, gfx1030) | Cross-architecture |
| **EPContext size** | ~100MB (ResNet50 with weights) | ~10MB (IR only) |
| **Validation** | Requires arch detection at runtime | Compile-time flexible |

---

### 2. Memory DLL Loading vs Disk Files

**Decision:** Load DLL directly from [EPContext](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html) memory buffer using [MemoryModule](https://github.com/fancycode/MemoryModule), no disk I/O.

**Rationale:**
- Cleaner deployment: No temporary files, no disk permissions needed
- Fast loading: ~1-10ms to parse PE and map to memory
- WebNN compatibility: Required for no-disk-access constraint
- Simple integration: MemoryModule ~1000 lines, [MPL 2.0 license](https://github.com/fancycode/MemoryModule/blob/master/LICENSE.txt)

**Trade-offs:**

| Aspect | Memory Loading | Disk Files |
|--------|---------------|-----------|
| **Deployment** | Single ONNX file | ONNX + separate DLL |
| **Startup** | ~1-10ms (memory map) | ~5-20ms (disk I/O + load) |
| **Dependencies** | [MemoryModule](https://github.com/fancycode/MemoryModule) (~50KB) | OS loader (zero deps) |
| **Security** | DLL in ONNX (user must trust) | Separate DLL (easier scanning) |

---

### 3. Pattern-Based Lowering vs Manual Transformation

**Decision:** Use [MLIR's pattern rewriting framework](https://mlir.llvm.org/docs/DialectConversion/) with typed operations from [onnx-mlir](https://github.com/onnx/onnx-mlir).

**Rationale:**
- MLIR best practice: Standard way to implement dialect conversions
- Type safety: Compile-time checked operation types vs runtime string matching
- Extensible: Adding operations = adding pattern classes
- Productivity: Semantic operand access (getX(), getW()) vs manual operand indexing
- Maintainability: ONNX spec updates handled via onnx-mlir submodule updates

**Trade-offs:**

| Aspect | Pattern-Based | Manual Transform |
|--------|--------------|------------------|
| **Type safety** | Compile-time checked | Runtime crashes possible |
| **Maintainability** | ONNX spec updates via submodule | Manual updates per spec change |
| **Learning curve** | Must learn [MLIR patterns](https://mlir.llvm.org/docs/PatternRewriter/) | Straightforward imperative code |
| **Attribute access** | Semantic getters | Manual string lookup + cast |

---

### 4. Stateful Interface (init/compute/cleanup)

**Decision:** Three-function lifecycle vs single stateless function.

**Rationale:**
- GPU memory allocation is expensive: ~35ms per GB ([HIP Issue #3809](https://github.com/ROCm/hip/issues/3809))
- Typical ResNet50 inference: ~5-10ms GPU compute
- Allocating per inference would add **3-4x overhead**
- Industry best practice: "Allocate once, reuse" ([CUDA](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)/[HIP optimization guides](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html))

**Trade-offs:**

| Aspect | Stateful (init/compute/cleanup) | Stateless (single function) |
|--------|--------------------------------|----------------------------|
| **Performance** | Allocate once (~35ms), reuse forever | Allocate per inference (~35ms each) |
| **Complexity** | 3 functions, state management | 1 function, simple |
| **Multi-model** | Each model manages own state | Simple resource sharing |
| **Error handling** | Separate init errors from compute errors | Single error path |

**Interface signature:**
- `int inference_init(void** out_state)` - Create GPU resources once
- `int inference_compute(void* state, span_t* inputs, span_t* outputs)` - Execute inference
- `int inference_cleanup(void* state)` - Free GPU resources

See [INTERFACE-DESIGN.md](mlir/INTERFACE-DESIGN.md) for complete specification.

---

### 5. Embedded Constants vs External Files

**Decision:** Embed weights/biases directly in DLL, not separate files.

**Rationale:**
- Single artifact: [EPContext](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html) is final compiled model (recompile to change weights)
- Deployment simplicity: No external dependencies or file paths
- Compiled code controls layout: Can optimize constant organization
- Semantic consistency: Constants are part of compiled code, not configuration

**Trade-offs:**

| Aspect | Embedded Constants | External Files |
|--------|-------------------|---------------|
| **DLL size** | ~100MB (ResNet50) | ~5MB (code only) |
| **Deployment** | Single ONNX file | ONNX + weight files |
| **Weight updates** | Requires recompilation | Can swap files |
| **Loading** | Constants ready at init | Extra I/O on startup |

See [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md) for implementation details.

---

### 6. Synchronous Execution vs Async

**Decision:** `inference_compute()` blocks until GPU work completes.

**Rationale:**
- Matches [ONNX Runtime CustomOp](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html) semantics (synchronous `Compute()`)
- Simpler interface (no separate sync function needed)
- Sufficient for single fused node = entire model execution
- Can add async variant later without breaking existing interface

**Trade-offs:**

| Aspect | Synchronous | Asynchronous |
|--------|------------|--------------|
| **Interface** | 3 functions | 4+ functions (compute + wait/poll) |
| **Complexity** | Simple blocking | Requires state management |
| **ORT integration** | Matches CustomOp semantics | Requires ORT async support |
| **Future-proof** | Can extend later | Hard to simplify later |

---

### 7. Full Model Fusion vs Per-Op Execution

**Decision:** Entire ONNX model graph fused into single [CustomOp](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html) node.

**Rationale:**
- Keep intermediates on GPU: No CPU round-trips between operations
- Simplifies interface: Only model inputs/outputs cross CPU-GPU boundary
- Enables global optimizations: Can fuse across operation boundaries
- Matches [EPContext](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html) design pattern: One compilation unit per subgraph

**Trade-offs:**

| Aspect | Full Model Fusion | Per-Op CustomOps |
|--------|------------------|------------------|
| **GPU efficiency** | Intermediates never leave GPU | CPU-GPU copy per operation |
| **Compilation** | All-or-nothing (one op fails = all fails) | Gradual fallback to CPU |
| **Debugging** | Harder to isolate failures | Easier per-op debugging |
| **Flexibility** | Cannot mix CustomOp + native ORT ops | Can fallback individual ops |

---

### 8. Standalone Resources vs Shared Context

**Decision:** Each compiled model creates its own GPU handles (hipStream, miopenHandle, etc.).

**Rationale:**
- **Zero CustomOp dependencies:** CustomOp has no HIP headers, backend-agnostic interface
- **Extensibility:** Adding libraries (rocFFT, rocRAND) doesn't change interface
- **Portability:** Same CustomOp code works with different GPU backends
- **Acceptable overhead:** 200MB extra memory on 64GB GPU (0.3%)
- **Architectural cleanliness:** Clean separation between runtime and compiled code

**Trade-offs:**

| Aspect | Standalone (Chosen) | Shared Context |
|--------|----------------------|----------------|
| **Memory overhead** | 3x for 3 models (~300MB) | 1x (~100MB kernel cache) |
| **CustomOp deps** | Zero GPU headers | Requires HIP headers |
| **Portability** | Backend-agnostic | HIP-specific interface |
| **Extensibility** | Interface stable (add libraries = no change) | Interface grows with libraries |
| **GPU utilization** | More memory used | Better memory efficiency |

**Memory Context:**
- [MI250X](https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html): 64GB memory → 200MB overhead = 0.3%
- [MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html): 192GB memory → 200MB overhead = 0.1%

See [MEMORY-MANAGEMENT.md](MEMORY-MANAGEMENT.md) for detailed memory allocation strategy.

---

## Quality Attributes

### Performance

**Target:** <10ms DLL load time, zero JIT overhead

**Approach:**
- [MemoryModule](https://github.com/fancycode/MemoryModule) for fast in-memory DLL loading
- [Stateful interface](#4-stateful-interface-initcomputecleanup): Allocate GPU resources once, reuse across inferences
- [Native compilation](#1-native-dll-vs-llvm-ir-storage): Eliminate JIT overhead entirely

**Key Constraint:** GPU memory allocation is expensive (~35ms/GB), driving the [stateful interface decision](#4-stateful-interface-initcomputecleanup)

### Scalability

**Dynamic shapes:**
- Compile with static ranks, runtime dimensions
- Example: `memref<1x?x?x?xf32>` (batch=1 static, height/width dynamic)
- Dimensions loaded at runtime from tensor metadata

**Variable I/O:**
- Interface uses `span_t` to handle N inputs/M outputs
- No recompilation needed for different input/output counts

See [DYNAMIC-SHAPE-DESIGN.md](DYNAMIC-SHAPE-DESIGN.md) for complete design.

### Portability

**Current State:**
- Interface is backend-agnostic (C ABI, no GPU-specific types)
- Implementation is HIP-specific (all passes target HIP/MIOpen)

**Design Intention:**
- CustomOp has zero GPU backend dependencies ([Decision #8](#8-standalone-resources-vs-shared-context))
- Theoretically can swap backends via different compiled DLL
- However: Current passes hardcode HIP operations throughout

**Realistic Assessment:**
- Interface provides abstraction layer for future portability
- Full backend portability would require new dialect + lowering passes (~1000+ lines)

### Reliability

**Error handling strategy:**
- Status codes returned from all interface functions (0=success, non-zero=error)
- Validation at interface boundaries (input/output count, state pointer validity)
- Type safety enforced at [MLIR](https://mlir.llvm.org/) compilation time

**Known Limitations:**
- Runtime validation is minimal (no DLL signature checking)
- GPU architecture mismatch detection planned but not required for MVP

### Security

**Type safety:**
- [MLIR](https://mlir.llvm.org/) provides compile-time type checking
- Operation type mismatches caught during compilation

**Runtime validation:**
- Limited: DLL loaded from [EPContext](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html) without signature verification
- Trust model: EPContext is part of ONNX model (user must trust the model)

**Known Gaps:**
- No DLL signature validation
- No GPU architecture validation at runtime
- No bounds checking on tensor dimensions

---

## Design Principles

These principles guided the architectural decisions:

### 1. Zero GPU Backend Dependencies in CustomOp

**Principle:** CustomOp must be backend-agnostic (no HIP/CUDA/SYCL headers).

**Rationale:** Enables future portability, clean separation of concerns

**Application:** All GPU operations abstracted behind `inference_*()` function pointers loaded from DLL

Related decision: [#8 Standalone Resources vs Shared Context](#8-standalone-resources-vs-shared-context)

### 2. Allocate GPU Resources Once

**Principle:** Create GPU handles/memory at initialization, reuse across inferences.

**Rationale:** GPU memory allocation is expensive (~35ms/GB), would dominate inference time

**Application:** Three-function lifecycle (init allocates, compute reuses, cleanup frees)

Related decision: [#4 Stateful Interface](#4-stateful-interface-initcomputecleanup)

### 3. Type-Safe MLIR Patterns

**Principle:** Use typed [MLIR](https://mlir.llvm.org/) operations, not string matching.

**Rationale:** Catch errors at compile time, improve maintainability

**Application:** [Pattern-based lowering](#3-pattern-based-lowering-vs-manual-transformation) with [onnx-mlir](https://github.com/onnx/onnx-mlir) typed operations

### 4. Compile-Time Known, Runtime Flexible

**Principle:** Ranks static (compile-time), dimensions dynamic (runtime).

**Rationale:** Balance between optimization (static info) and flexibility (dynamic shapes)

**Application:** [MLIR types](https://mlir.llvm.org/docs/Dialects/Builtin/#ranked-tensor-types) encode static ranks with dynamic dimensions (`memref<1x?x?x?xf32>`)

### 5. Single Source of Truth for Constants

**Principle:** Embed constants in DLL, not external files.

**Rationale:** Simplifies deployment, ensures consistency, enables optimizations

**Application:** ONNX Constant nodes lowered to LLVM globals in compiled DLL

Related decision: [#5 Embedded Constants vs External Files](#5-embedded-constants-vs-external-files)

---

## ONNX-MLIR Integration

### Rationale

Use [onnx-mlir](https://github.com/wcy123/onnx-mlir) fork for type-safe ONNX dialect instead of manual string parsing.

**Key Benefits:**

| Benefit | Without onnx-mlir | With onnx-mlir |
|---------|------------------|----------------|
| **Type safety** | String matching: `if (opName == "Conv")` | Typed patterns: `OpConversionPattern<ONNXConvOp>` |
| **Attribute access** | Manual cast + null check | Semantic getters: `getKernelShape()` |
| **Maintainability** | Manual updates per ONNX spec change | Auto-updated via submodule |
| **Error messages** | "Invalid operand 0" | "Conv input X must be 4D tensor" |

### Integration Points

1. **Level-1 Pass:** Parse ONNX → ONNXOps (typed) → HIP dialect
2. **Transform Passes:** [Pattern-based lowering](#3-pattern-based-lowering-vs-manual-transformation) using typed ONNX operations
3. **Shape Inference:** Reuse onnx-mlir's shape inference for dynamic shapes

See [MLIR-COMPILATION-DESIGN.md](MLIR-COMPILATION-DESIGN.md) for complete pipeline details.

### Non-Integration

**Runtime:** Custom Op has zero MLIR dependencies (only consumes pre-compiled DLL)

---

## References

### Specifications
- [INTERFACE-DESIGN.md](mlir/INTERFACE-DESIGN.md) - Complete C interface specification
- [HIP-DIALECT-DESIGN.md](mlir/HIP-DIALECT-DESIGN.md) - HIP dialect operations and semantics
- [MLIR-COMPILATION-DESIGN.md](MLIR-COMPILATION-DESIGN.md) - MLIR module structure and lowering

### Implementation Details
- [MEMORY-MANAGEMENT.md](MEMORY-MANAGEMENT.md) - Memory allocation strategy
- [DYNAMIC-SHAPE-DESIGN.md](DYNAMIC-SHAPE-DESIGN.md) - Dynamic shape handling
- [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md) - Constant embedding in DLL

### External References
- [ONNX Runtime](https://onnxruntime.ai/) - AI inference runtime
- [ONNX Runtime EP Context Design](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html) - EPContext mechanism
- [ONNX Runtime Custom Operators](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html) - CustomOp API
- [MLIR](https://mlir.llvm.org/) - Multi-Level Intermediate Representation
- [MLIR Dialect Conversion Guide](https://mlir.llvm.org/docs/DialectConversion/) - Pattern rewriting framework
- [MLIR Pattern Rewriter](https://mlir.llvm.org/docs/PatternRewriter/) - Pattern matching API
- [onnx-mlir](https://github.com/onnx/onnx-mlir) - ONNX MLIR dialect (upstream)
- [onnx-mlir fork](https://github.com/wcy123/onnx-mlir) - Windows build fixes
- [MemoryModule](https://github.com/fancycode/MemoryModule) - In-memory DLL loading library
- [AMD ROCm](https://rocm.docs.amd.com/) - AMD GPU computing platform
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/) - HIP API documentation
- [HIP Performance Guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html) - Optimization best practices
- [AMD Instinct MI250X](https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html) - Data center GPU
- [AMD Instinct MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) - Next-gen data center GPU

---

**Document History:**
- v2.0 (2026-02-11): Restructured to focus on architectural decisions, removed implementation details
- v1.0 (2026-02-09): Initial architecture document
