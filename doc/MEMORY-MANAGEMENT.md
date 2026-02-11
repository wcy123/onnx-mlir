<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Memory Management Strategy

**Date:** 2026-02-09
**Status:** Design Document

---

## Overview

GPU memory allocation is expensive (~35ms per GB). Optimizing allocation strategy is critical for performance.

**Goal:** Minimize allocation overhead while maximizing memory efficiency through a three-phase optimization approach.

---

## In-Place Semantics Design

**Key Design:**
- **HIP dialect operations** use **in-place semantics**: Operations take output buffer as argument
  - Example: `hip.conv(%ctx, %input, %weights, %bias, %output)` - no return value
- **HIP dialect functions** use **destination-passing style**: Outputs passed as pointer arguments
  - Example: `func.func @inference_compute(%state: !llvm.ptr, %inputs: !llvm.ptr, %outputs: !llvm.ptr) -> i32`
  - Returns i32 status code (0 = success)
- **Final C interface** uses **destination-passing style**: Outputs passed via span_t*
  - Example: `int inference_compute(void* state, span_t* inputs, span_t* outputs)`

This design ensures consistent in-place semantics at all levels: operations write to pre-allocated buffers, functions receive output buffers from callers, and the C interface passes outputs via span_t. No memory is returned from functions - all outputs are written to caller-provided buffers.

---

## Memory Categories

| Type | Lifetime | Managed By | Example |
|------|----------|------------|---------|
| **Constants** | Session | Compiled code (weights embedded in DLL) | Conv weights: 37KB |
| **Intermediates** | Per-layer | Compiled code (pre-allocated in init) | Activation: 3MB |
| **Workspace** | Session | Compiled code (shared scratch space) | MIOpen: 8MB |
| **Input/Output** | Per-call | Caller (CustomOp) | Model input: 600KB |

**TODO - Input/Output GPU Buffer Allocation:**
- **Current design:** Allocate per-call in `inference_compute()` (see doc/mlir/passes/GenerateInterfacePass.md)
  - Simple implementation, no state needed
  - Cost: ~35ms/GB allocation overhead per inference
- **Alternative (Phase 2):** Pre-allocate in `inference_init()` with maximum expected size
  - One-time allocation overhead
  - Requires fixed maximum batch size
  - May waste memory if actual batch << max batch
- **Decision needed:** Choose strategy based on deployment scenario (fixed vs variable batch sizes)

---

## Three-Phase Optimization Strategy

### Phase 1: Naive Inline Allocation (Baseline)

**Implementation:** Allocate in `inference_compute()`, free immediately.

```mlir
func.func @inference_compute(%state: !llvm.ptr, ...) -> i32 {
  %buf = llvm.call @hipMalloc(%size) : (i64) -> !llvm.ptr
  llvm.call @miopenConvolutionForward(..., %buf, ...) : ...
  llvm.call @hipFree(%buf) : (!llvm.ptr) -> ()
  return %c0 : i32
}
```

**Performance:** ~20-65ms per inference (allocation overhead dominates)

**Use:** Initial implementation only, to get pipeline working.

---

### Phase 2: Allocation Hoisting ⭐ **CRITICAL**

**Key Idea:** Move allocations from `inference_compute()` to `inference_init()`. Allocate once, reuse many times.

**What is "Hoisting"?** Move code to execute earlier/less frequently (from every call → once per session).

**Implementation:**

1. **During HIP→LLVM lowering:** Track all `hip.alloc` operations
2. **Generate `inference_init()`:** Allocate all buffers, store in state struct
3. **Generate `inference_compute()`:** Load pre-allocated buffers from state
4. **Generate `inference_cleanup()`:** Free all buffers

**State struct:**
```c
struct InferenceState {
  hipStream_t stream;              // offset 0
  miopenHandle_t miopen_handle;    // offset 8
  hipblasLtHandle_t hipblas_handle;// offset 16

  // Pre-allocated buffers (offset 24)
  void* intermediate_0;  // 3MB
  void* intermediate_1;  // 1.5MB
  void* workspace;       // 8MB
  void* weight_0_gpu;    // Uploaded constants
  // ...
};
```

**Code generation:**
```mlir
// inference_init: Allocate once
func.func @inference_init(%out_state: !llvm.ptr<!llvm.ptr>) -> i32 {
  %state = llvm.call @malloc(%state_size) : ...

  // Pre-allocate all buffers
  %buf_0 = llvm.call @hipMalloc(%size_0) : (i64) -> !llvm.ptr
  %buf_0_field = llvm.getelementptr %state[0, 3, 0] : ...
  llvm.store %buf_0, %buf_0_field : !llvm.ptr

  // ... more buffers

  llvm.store %state, %out_state : ...
  return %c0 : i32
}

// inference_compute: Reuse pre-allocated buffers
func.func @inference_compute(%state: !llvm.ptr, ...) -> i32 {
  // Load pre-allocated buffer (no hipMalloc!)
  %buf_0_field = llvm.getelementptr %state[0, 3, 0] : ...
  %buf_0 = llvm.load %buf_0_field : !llvm.ptr

  llvm.call @miopenConvolutionForward(..., %buf_0, ...) : ...
  return %c0 : i32
}
```

**Performance:** ~5ms per inference (allocation overhead eliminated)

**Speedup: 4-12x faster than Phase 1**

**Priority:** HIGH - implement immediately after basic pipeline works

**Complexity:** Medium (⭐⭐⭐☆☆) - 1-2 weeks

---

### Phase 3: Memory Pooling (Future Optimization)

**Key Idea:** Allocate single memory pool. Reuse memory for intermediates with non-overlapping lifetimes.

**Observation:** Intermediate tensors are not all alive simultaneously → can share memory.

**Example:**
```
Layer 1: input → buf1 (3MB)     ← buf1 alive
Layer 2: buf1 → buf2 (1.5MB)    ← buf1+buf2 alive (peak: 4.5MB)
Layer 3: buf2 → buf3 (784KB)    ← buf2+buf3 alive, buf1 DEAD
                                ← buf3 can REUSE buf1's space!
```

**Memory layout:**
```
Pool (4.5MB total):
┌─────────────────┬──────────────┐
│ buf1 → buf3     │ buf2         │
│ (0..3MB)        │ (3MB..4.5MB) │
└─────────────────┴──────────────┘

Allocations:
  Phase 2: 3MB + 1.5MB + 784KB = 5.3MB
  Phase 3: 4.5MB (single pool)
  Savings: 800KB (15%)
```

**For ResNet50:**
- Phase 2: ~200MB allocated
- Phase 3: ~55MB pool
- **Savings: 73% (145MB)**

**Algorithm:**
1. **Liveness analysis:** Determine when each tensor is alive
2. **Layout optimization:** Assign offsets to minimize pool size (graph coloring)
3. **Code generation:** Replace allocations with offset calculations

**Priority:** Medium - implement when memory is constrained

**Complexity:** High (⭐⭐⭐⭐☆) - 3-4 weeks

---

## Implementation Roadmap

| Phase | Technique | Performance | Memory | Priority | Effort |
|-------|-----------|-------------|--------|----------|--------|
| **1** | Inline alloc | Baseline | 200MB | Testing only | ⭐☆☆☆☆ |
| **2** | Hoist to init | **4-12x faster** | 200MB | **HIGH** | ⭐⭐⭐☆☆ |
| **3** | Memory pool | Same | **55MB (73% savings)** | Medium | ⭐⭐⭐⭐☆ |

**Recommendation:**
1. ✅ Phase 1: Implement as part of basic HIP→LLVM lowering (get pipeline working)
2. 🎯 Phase 2: Implement immediately (critical for performance)
3. 📋 Phase 3: Implement when memory becomes bottleneck

---

## Performance Context

**GPU Memory Allocation Cost:**
- `hipMalloc(1GB)`: ~35ms
- `hipFree()`: 5-20ms (implicit synchronization)

**Typical Inference:**
- ResNet50 compute: ~5-10ms
- Without hoisting: +20-65ms allocation overhead = **3-6x slower**
- With hoisting: negligible overhead

**Industry Practice:** All production ML compilers (TensorRT, XLA, TVM) use allocation hoisting and memory pooling.

---

## Related Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system design
- [MLIR-COMPILATION-DESIGN.md](MLIR-COMPILATION-DESIGN.md) - MLIR lowering pipeline
- [notes/MEMORY_MANAGEMENT_DESIGN.md](../notes/MEMORY_MANAGEMENT_DESIGN.md) - Detailed technical notes
