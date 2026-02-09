# Memory Management Design - Three-Phase Optimization Strategy

**Date:** 2026-02-09
**Context:** Memory allocation optimization for ONNX→HIP→LLVM compiled inference

---

## Overview

This document describes a three-phase approach to memory management optimization in the compiled inference code. Each phase builds upon the previous one, progressively reducing memory allocation overhead and total memory usage.

**Key Insight:** GPU memory allocation (`hipMalloc`) is expensive (~35ms per GB). Optimizing allocation strategy is critical for performance.

---

## Background: GPU Memory Allocation Cost

| Operation | Overhead | Source |
|-----------|----------|---------|
| `hipMalloc` | ~35ms per GB | [HIP Issue #3809](https://github.com/ROCm/hip/issues/3809) |
| `hipFree` | Implicit sync (5-20ms) | [HIP Performance Guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html) |

**Context:** Typical ResNet50 inference on GPU takes ~5-10ms. Allocating/freeing 500MB on every inference would add **~22-37ms overhead** (3-4x slowdown)!

**Industry Best Practice:** "Allocate once, reuse many times" - all GPU programming guides recommend this.

---

## Memory Categories

Four types of memory with different characteristics:

### 1. Constants (Weights, Biases)
- **Size:** Fixed at compile time (known from ONNX model)
- **Location:** CPU (embedded in DLL) → GPU (uploaded once)
- **Lifetime:** Entire session (from init to cleanup)
- **Access pattern:** Read-only during inference
- **Example:** Conv weights [64×3×7×7] = 37KB

### 2. Intermediate Tensors (Activations)
- **Size:** Fixed for static shapes, variable for dynamic shapes
- **Location:** GPU memory only
- **Lifetime:** Per-layer (short-lived, can be reused)
- **Access pattern:** Write once, read once (usually)
- **Example:** Conv output [1×64×112×112] = 3MB

### 3. Workspace (Scratch Memory)
- **Size:** Maximum across all operations (known at compile time)
- **Location:** GPU memory
- **Lifetime:** Entire session (reused by all operations)
- **Access pattern:** Read-write (temporary)
- **Example:** MIOpen workspace = 8MB

### 4. Input/Output Tensors
- **Size:** Variable (user-controlled, batch size)
- **Location:** CPU (current), GPU (future)
- **Lifetime:** Per-inference call
- **Access pattern:** Managed by caller (CustomOp)
- **Example:** Model input [1×3×224×224] = 600KB

---

## Three-Phase Optimization Strategy

### Phase 1: Naive Inline Allocation (Baseline)

**Description:** Allocate and free memory inline in `inference_compute()`.

**Implementation:**
```mlir
// HIP IR
func.func @inference_compute(%ctx: !hip.context, ...) {
  // Allocate every time
  %buf = hip.alloc(%ctx) : memref<1x64x112x112xf32, 1>
  hip.conv(%ctx, %input, %weights, %bias, %buf) {...}
  hip.free(%ctx, %buf)
  return %buf
}

// Lowers to LLVM IR
func.func @inference_compute(%state: !llvm.ptr, ...) -> i32 {
  %buf = llvm.call @hipMalloc(%size) : (i64) -> !llvm.ptr
  llvm.call @miopenConvolutionForward(..., %buf, ...) : ...
  llvm.call @hipFree(%buf) : (!llvm.ptr) -> ()
  return %c0 : i32
}
```

**Performance:**
```
Per-inference cost:
  hipMalloc(3MB)     : ~0.1ms
  hipMalloc(1.5MB)   : ~0.05ms
  hipMalloc(784KB)   : ~0.03ms
  Actual compute     : 5ms
  hipFree × 3        : ~15-60ms (synchronization!)
  --------------------------------
  Total              : ~20-65ms
```

**Pros:**
- ✅ Simple to implement
- ✅ Correct (no memory management bugs)
- ✅ Easy to debug

**Cons:**
- ❌ **4-13x slower** than necessary
- ❌ Wastes GPU time on allocation/deallocation
- ❌ Allocation overhead dominates compute

**When to use:** Initial implementation only, to get pipeline working.

---

### Phase 2: Allocation Hoisting (Performance Fix)

**Description:** Move allocations from `inference_compute()` to `inference_init()`. Allocate once per session, reuse across all inference calls.

**What is "Hoisting"?**

**Hoist** = Compiler term meaning "move code to execute earlier/less frequently"

Example analogy:
```python
# Before hoisting (inefficient)
def process_images(images):
    for img in images:
        buffer = allocate_buffer()  # ← Allocates 1000 times!
        apply_filter(img, buffer)
        free_buffer(buffer)

# After hoisting (efficient)
def process_images(images):
    buffer = allocate_buffer()      # ← Hoisted: allocates once!
    for img in images:
        apply_filter(img, buffer)   # ← Reuses same buffer
    free_buffer(buffer)
```

**Implementation:**

**Step 1: Analyze allocations during HIP→LLVM pass**
```cpp
// During HIP→LLVM lowering
std::vector<BufferInfo> buffers;
func.walk([&](hip::AllocOp allocOp) {
  buffers.push_back({
    .index = buffers.size(),
    .type = allocOp.getMemref().getType(),
    .size_bytes = computeSizeInBytes(allocOp.getMemref().getType())
  });
});
```

**Step 2: Generate `inference_init()` with ALL allocations**
```mlir
func.func @inference_init(%out_state: !llvm.ptr<!llvm.ptr>) -> i32 {
  // Allocate state struct
  %state = llvm.call @malloc(%state_size) : (i64) -> !llvm.ptr

  // Create GPU handles
  %stream = llvm.call @hipStreamCreate() : () -> !llvm.ptr
  %miopen = llvm.call @miopenCreate() : () -> !llvm.ptr

  // Store handles in state
  %stream_field = llvm.getelementptr %state[0, 0] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %stream, %stream_field : !llvm.ptr

  // ============================================================
  // HOIST: Pre-allocate all intermediate buffers HERE
  // ============================================================

  // Buffer 0: intermediate_0 (1×64×112×112×f32 = 3,211,264 bytes)
  %size_0 = llvm.mlir.constant(3211264 : i64) : i64
  %buf_0 = llvm.call @hipMalloc(%size_0) : (i64) -> !llvm.ptr
  %buf_0_field = llvm.getelementptr %state[0, 3, 0] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %buf_0, %buf_0_field : !llvm.ptr

  // Buffer 1: intermediate_1 (1×128×56×56×f32 = 1,605,632 bytes)
  %size_1 = llvm.mlir.constant(1605632 : i64) : i64
  %buf_1 = llvm.call @hipMalloc(%size_1) : (i64) -> !llvm.ptr
  %buf_1_field = llvm.getelementptr %state[0, 3, 1] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %buf_1, %buf_1_field : !llvm.ptr

  // Workspace buffer (8MB)
  %workspace_size = llvm.mlir.constant(8388608 : i64) : i64
  %workspace = llvm.call @hipMalloc(%workspace_size) : (i64) -> !llvm.ptr
  %workspace_field = llvm.getelementptr %state[0, 3, 10] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %workspace, %workspace_field : !llvm.ptr

  // Upload constants to GPU
  %weight_cpu = llvm.mlir.addressof @conv_weight : !llvm.ptr
  %weight_size = llvm.mlir.constant(37632 : i64) : i64
  %weight_gpu = llvm.call @hipMalloc(%weight_size) : (i64) -> !llvm.ptr
  llvm.call @hipMemcpy(%weight_gpu, %weight_cpu, %weight_size, %H2D) : ...

  llvm.store %state, %out_state : !llvm.ptr<!llvm.ptr>
  return %c0 : i32
}
```

**Step 3: Generate `inference_compute()` that REUSES buffers**
```mlir
func.func @inference_compute(%state: !llvm.ptr, %inputs: !llvm.ptr, %outputs: !llvm.ptr) -> i32 {
  // Extract handles from state
  %miopen_field = llvm.getelementptr %state[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %miopen = llvm.load %miopen_field : !llvm.ptr

  // ============================================================
  // REUSE: Load pre-allocated buffers (NO hipMalloc here!)
  // ============================================================

  // Get buffer 0 (was allocated in init)
  %buf_0_field = llvm.getelementptr %state[0, 3, 0] : (!llvm.ptr) -> !llvm.ptr
  %buf_0 = llvm.load %buf_0_field : !llvm.ptr

  // Get buffer 1 (was allocated in init)
  %buf_1_field = llvm.getelementptr %state[0, 3, 1] : (!llvm.ptr) -> !llvm.ptr
  %buf_1 = llvm.load %buf_1_field : !llvm.ptr

  // Get workspace (was allocated in init)
  %workspace_field = llvm.getelementptr %state[0, 3, 10] : (!llvm.ptr) -> !llvm.ptr
  %workspace = llvm.load %workspace_field : !llvm.ptr

  // Execute operations using pre-allocated buffers
  %input_ptr = /* extract from inputs */

  llvm.call @miopenConvolutionForward(
    %miopen, %alpha,
    %input_desc, %input_ptr,
    %weight_desc, %weight_gpu,
    %conv_desc, %algo,
    %beta,
    %output_desc, %buf_0,        // ← Pre-allocated!
    %workspace, %workspace_size  // ← Pre-allocated!
  ) : ...

  return %c0 : i32
}
```

**Step 4: Free in `inference_cleanup()`**
```mlir
func.func @inference_cleanup(%state: !llvm.ptr) -> i32 {
  // Free all buffers
  %buf_0_field = llvm.getelementptr %state[0, 3, 0] : (!llvm.ptr) -> !llvm.ptr
  %buf_0 = llvm.load %buf_0_field : !llvm.ptr
  llvm.call @hipFree(%buf_0) : (!llvm.ptr) -> ()

  // ... free other buffers

  // Destroy handles
  %miopen_field = llvm.getelementptr %state[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %miopen = llvm.load %miopen_field : !llvm.ptr
  llvm.call @miopenDestroy(%miopen) : (!llvm.ptr) -> ()

  llvm.call @free(%state) : (!llvm.ptr) -> ()
  return %c0 : i32
}
```

**State Struct Layout:**
```c
struct InferenceState {
  // GPU Handles
  hipStream_t stream;              // field 0
  miopenHandle_t miopen_handle;    // field 1
  hipblasLtHandle_t hipblas_handle;// field 2

  // Pre-allocated Buffers (field 3 = array of pointers)
  struct {
    void* intermediate_0;          // [0] - 3MB
    void* intermediate_1;          // [1] - 1.5MB
    void* intermediate_2;          // [2] - 784KB
    // ...
    void* workspace;               // [10] - 8MB
    void* weight_0_gpu;            // [20] - weights
    void* weight_1_gpu;            // [21] - weights
  } buffers;
};
```

**Performance:**
```
Session lifetime:
  inference_init():
    hipMalloc × 5        : ~0.2ms (once per session)

  inference_compute() × 1000 calls:
    Load pointers        : negligible (just memory reads)
    Actual compute       : 5ms × 1000 = 5000ms

  inference_cleanup():
    hipFree × 5          : ~100ms (once per session)

Total for 1000 inferences: ~5.3 seconds (vs ~20-65 seconds in Phase 1)
```

**Speedup: 4-12x faster than Phase 1!**

**Pros:**
- ✅ **Massive performance improvement** (allocation overhead eliminated)
- ✅ Zero overhead in inference hot path
- ✅ Still relatively simple to implement
- ✅ Works with dynamic batch sizes (allocate for max size)

**Cons:**
- ⚠️ Wastes memory (each buffer has separate allocation, no reuse)
- ⚠️ For ResNet50: ~200MB allocated, but only ~50MB peak usage

**When to use:** Production baseline - always implement this.

**Implementation Complexity:** ⭐⭐⭐☆☆ (Medium)
- Track all hip.alloc operations
- Generate three-function code (init/compute/cleanup)
- Manage state struct layout
- **Estimate:** 1-2 weeks

---

### Phase 3: Memory Pooling with Liveness-Based Reuse (Memory Optimization)

**Description:** Allocate a single memory pool. Reuse memory for intermediate tensors with non-overlapping lifetimes.

**Key Insight:** Intermediate tensors are not all alive at the same time. We can overlay them in memory.

**Example: 3-Layer Network**

```mlir
// Naive Phase 2 approach:
func.func @model(%ctx: !hip.context, %input: memref<...>) {
  %buf1 = hip.alloc(%ctx) : memref<1x64x112x112xf32>   // 3MB
  hip.conv(%ctx, %input, %w1, %b1, %buf1)

  %buf2 = hip.alloc(%ctx) : memref<1x128x56x56xf32>    // 1.5MB
  hip.conv(%ctx, %buf1, %w2, %b2, %buf2)
  // ↑ buf1's last use - it's DEAD after this!

  %buf3 = hip.alloc(%ctx) : memref<1x256x28x28xf32>    // 784KB
  hip.conv(%ctx, %buf2, %w3, %b3, %buf3)
  // ↑ buf2's last use - it's DEAD after this!

  return %buf3
}
```

**Liveness Analysis:**

| Time Step | Operation | buf1 | buf2 | buf3 | Peak Memory |
|-----------|-----------|------|------|------|-------------|
| 0 | Conv1: write buf1 | ✅ ALIVE | ❌ | ❌ | 3 MB |
| 1 | Conv2: read buf1, write buf2 | ✅ ALIVE | ✅ ALIVE | ❌ | **4.5 MB** ← Peak! |
| 2 | Conv3: read buf2, write buf3 | ❌ DEAD | ✅ ALIVE | ✅ ALIVE | 2.3 MB |
| 3 | Return buf3 | ❌ DEAD | ❌ DEAD | ✅ ALIVE | 0.8 MB |

**Key Observation:**
- buf1 dies at step 1 (after Conv2 reads it)
- buf3 is born at step 2 (Conv3 writes it)
- buf1 and buf3 **never overlap** → can share memory!

**Memory Layout After Optimization:**

```
Memory Pool (4.5MB total):
┌──────────────────────────────────────────────────────────┐
│ Region A (0..3MB)      │ Region B (3MB..4.5MB)           │
│                        │                                  │
│ buf1 (3MB)             │ buf2 (1.5MB)                     │
│ [Conv1 output]         │ [Conv2 output]                   │
│                        │                                  │
│ ↓ buf1 dies            │ ↓ buf2 dies                      │
│                        │                                  │
│ buf3 (784KB) ← REUSE! │ <unused after Conv2>             │
│ [Conv3 output]         │                                  │
└──────────────────────────────────────────────────────────┘

Offset mapping:
  buf1 → pool + 0       (0..3MB)
  buf2 → pool + 3MB     (3MB..4.5MB)
  buf3 → pool + 0       (0..784KB) ← REUSES buf1's space!
```

**Implementation:**

**Step 1: Liveness Analysis (Compiler Algorithm)**
```cpp
// Pseudocode for liveness analysis pass
struct LivenessInfo {
  int first_def;   // When value is created (written)
  int last_use;    // When value is last read
};

std::map<Value, LivenessInfo> computeLiveness(Function func) {
  std::map<Value, LivenessInfo> liveness;

  // Number all operations sequentially
  int op_index = 0;
  for (Operation& op : func) {
    // Track when values are defined (created)
    for (Value result : op.results) {
      liveness[result].first_def = op_index;
    }

    // Track when values are used (read)
    for (Value operand : op.operands) {
      liveness[operand].last_use = max(liveness[operand].last_use, op_index);
    }

    op_index++;
  }

  return liveness;
}

// Build interference graph: two values interfere if they're alive at same time
bool interferes(LivenessInfo a, LivenessInfo b) {
  return !(a.last_use < b.first_def || b.last_use < a.first_def);
}
```

**Step 2: Memory Layout Optimization (Graph Coloring / Bin Packing)**
```cpp
struct MemoryLayout {
  std::map<Value, size_t> offsets;  // Value → offset in pool
  size_t total_size;                 // Total pool size needed
};

MemoryLayout optimizeLayout(
    std::vector<Value> values,
    std::map<Value, LivenessInfo> liveness) {

  // Sort by size (largest first - greedy heuristic)
  std::sort(values.begin(), values.end(), [](Value a, Value b) {
    return getSizeInBytes(a) > getSizeInBytes(b);
  });

  MemoryLayout layout;
  std::vector<Region> allocated_regions;

  for (Value v : values) {
    size_t size = getSizeInBytes(v);

    // Find lowest offset that doesn't conflict with live values
    size_t offset = 0;
    while (true) {
      bool conflicts = false;

      // Check if [offset, offset+size) overlaps with any live region
      for (Region& r : allocated_regions) {
        if (interferes(liveness[v], liveness[r.value]) &&
            rangesOverlap(offset, offset+size, r.start, r.end)) {
          conflicts = true;
          offset = r.end;  // Try next position
          break;
        }
      }

      if (!conflicts) break;
    }

    layout.offsets[v] = offset;
    allocated_regions.push_back({v, offset, offset + size});
    layout.total_size = max(layout.total_size, offset + size);
  }

  return layout;
}
```

**Step 3: Code Generation**
```mlir
// inference_init(): Allocate single pool
func.func @inference_init(%out_state: !llvm.ptr<!llvm.ptr>) -> i32 {
  %state = llvm.call @malloc(%state_size) : (i64) -> !llvm.ptr

  // Allocate SINGLE memory pool (peak usage size)
  %pool_size = llvm.mlir.constant(4718592 : i64) : i64  // 4.5MB
  %memory_pool = llvm.call @hipMalloc(%pool_size) : (i64) -> !llvm.ptr

  // Store pool base pointer in state
  %pool_field = llvm.getelementptr %state[0, 3, 0] : (!llvm.ptr) -> !llvm.ptr
  llvm.store %memory_pool, %pool_field : !llvm.ptr

  llvm.store %state, %out_state : !llvm.ptr<!llvm.ptr>
  return %c0 : i32
}

// inference_compute(): Compute offsets from pool
func.func @inference_compute(%state: !llvm.ptr, ...) -> i32 {
  // Load memory pool base
  %pool_field = llvm.getelementptr %state[0, 3, 0] : (!llvm.ptr) -> !llvm.ptr
  %pool = llvm.load %pool_field : !llvm.ptr

  // buf1 = pool + 0
  %buf1 = %pool
  llvm.call @miopenConvolutionForward(..., %buf1, ...) : ...

  // buf2 = pool + 3MB
  %offset_buf2 = llvm.mlir.constant(3145728 : i64) : i64
  %buf2 = llvm.getelementptr %pool[%offset_buf2] : (!llvm.ptr, i64) -> !llvm.ptr
  llvm.call @miopenConvolutionForward(..., %buf2, ...) : ...

  // buf3 = pool + 0 (REUSES buf1's space!)
  %buf3 = %pool  // Same pointer as buf1!
  llvm.call @miopenConvolutionForward(..., %buf3, ...) : ...

  return %c0 : i32
}

// inference_cleanup(): Free single pool
func.func @inference_cleanup(%state: !llvm.ptr) -> i32 {
  %pool_field = llvm.getelementptr %state[0, 3, 0] : (!llvm.ptr) -> !llvm.ptr
  %pool = llvm.load %pool_field : !llvm.ptr
  llvm.call @hipFree(%pool) : (!llvm.ptr) -> ()
  llvm.call @free(%state) : (!llvm.ptr) -> ()
  return %c0 : i32
}
```

**Performance (ResNet50 Example):**

| Approach | Memory Allocated | Peak Usage | Efficiency |
|----------|------------------|------------|------------|
| Phase 1: Inline Alloc | ~200MB (repeated) | ~50MB | N/A (reallocates) |
| Phase 2: Hoisted | ~200MB | ~50MB | 25% (wastes 150MB) |
| Phase 3: Pooled | **~55MB** | ~50MB | **91%** (wastes 5MB) |

**Savings: 145MB (73% reduction) compared to Phase 2!**

**Pros:**
- ✅ **Massive memory savings** (60-70% for typical models)
- ✅ Enables larger batch sizes or larger models on same GPU
- ✅ Better cache locality (all activations in contiguous memory)
- ✅ Single allocation = faster init

**Cons:**
- ⚠️ Complex implementation (liveness analysis + layout optimization)
- ⚠️ Dynamic shapes harder to handle (need conservative worst-case sizing)
- ⚠️ Debugging harder (pointer arithmetic, overlapping buffers)

**When to use:** Production optimization for memory-constrained scenarios.

**Implementation Complexity:** ⭐⭐⭐⭐☆ (High)
- Implement liveness analysis pass
- Implement memory layout optimization (graph coloring / bin packing)
- Handle dynamic shapes (worst-case analysis)
- Extensive testing (buffer overlaps can cause subtle bugs)
- **Estimate:** 3-4 weeks

---

## Can We Consolidate ALL Memory Types?

**Question:** Can constants/input/output/intermediate share the same pool?

**Analysis:**

| Memory Type | Can Share Pool? | Recommendation |
|-------------|-----------------|----------------|
| **Intermediates** | ✅ YES | Primary target - maximum savings here |
| **Workspace** | ✅ YES | Can overlap with intermediates that aren't live during operations |
| **Constants (weights)** | ⚠️ MAYBE | Technically yes, but usually kept separate for simplicity |
| **Input tensors** | ❌ NO | Owned by caller (CustomOp), not under our control |
| **Output tensors** | ❌ NO | Owned by caller (CustomOp), must be written to caller's buffer |

**Recommended Approach for Phase 3:**

```c
struct InferenceState {
  // Option A: Conservative (Simpler Implementation)
  void* constants_pool;      // Weights, biases (read-only, separate)
  void* workspace_pool;      // Scratch space (shared across ops)
  void* intermediate_pool;   // Activations (liveness-based reuse)

  // Option B: Aggressive (Maximum Memory Savings)
  void* unified_pool;        // Everything in one allocation!
  // Constants at offsets [0..C)
  // Workspace at offsets [C..C+W)
  // Intermediates at offsets [C+W..total)
};
```

**My Recommendation:**
- **Phase 3:** Option A (separate pools) - easier to implement, still 60-70% savings
- **Phase 4:** Option B (unified pool) - squeeze out last 5-10% if needed

---

## Implementation Roadmap

### Phase 1: Naive Baseline ✅
**Status:** Implement first to get pipeline working
**Duration:** Already done as part of basic lowering
**Goal:** Correctness, not performance

### Phase 2: Allocation Hoisting 🎯
**Status:** High priority - implement next
**Duration:** 1-2 weeks
**Goal:** Eliminate allocation overhead in hot path
**Success Criteria:** 4-12x speedup on repeated inference

### Phase 3: Memory Pooling 📋
**Status:** Future optimization
**Duration:** 3-4 weeks
**Goal:** Reduce memory footprint 60-70%
**Success Criteria:** Larger batches fit on same GPU, or support larger models

### Phase 4: Unified Pool (Optional) 💎
**Status:** Advanced optimization
**Duration:** 1-2 weeks
**Goal:** Squeeze out last 5-10% memory savings
**Success Criteria:** Maximum memory efficiency

---

## Industry Examples

### TensorRT (NVIDIA)
- Uses liveness-based memory pooling
- ResNet50: ~50MB peak usage (vs ~200MB naive)
- **4x memory savings**

### XLA (Google/TensorFlow)
- "Buffer Assignment" pass implements liveness analysis
- Automatically reuses buffers across operations
- Critical for TPU efficiency

### TVM (Apache)
- "Memory Planning" pass
- Graph coloring algorithm for buffer assignment
- **60-70% memory reduction** typical

**Conclusion:** Phase 3 optimization is industry-standard for production ML compilers.

---

## Summary

| Phase | Technique | Performance | Memory | Complexity | Priority |
|-------|-----------|-------------|--------|------------|----------|
| **1** | Inline alloc | Baseline | 200MB | ⭐☆☆☆☆ | Testing only |
| **2** | Hoist to init | **4-12x faster** | 200MB | ⭐⭐⭐☆☆ | **HIGH** |
| **3** | Memory pool | Same | **55MB** | ⭐⭐⭐⭐☆ | Medium |
| **4** | Unified pool | Same | **50MB** | ⭐⭐⭐⭐☆ | Low |

**Recommendation:**
1. Implement Phase 1 (done during basic lowering)
2. Implement Phase 2 immediately (critical for performance)
3. Implement Phase 3 when memory becomes a bottleneck
4. Consider Phase 4 only if 5-10% matters

---

## References

- [HIP Performance Guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)
- [CUDA Best Practices Guide - Memory Optimization](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- [TensorFlow XLA Buffer Assignment](https://www.tensorflow.org/xla/operation_semantics#buffer_assignment)
- [TVM Memory Planning Pass](https://tvm.apache.org/docs/dev/how_to/relay_bring_your_own_codegen.html#memory-planning)

---

**Next Steps:**
1. Document this in doc/MLIR-COMPILATION-DESIGN.md
2. Implement Phase 2 (allocation hoisting) in HipToLLVM.cpp
3. Design liveness analysis pass for future Phase 3
