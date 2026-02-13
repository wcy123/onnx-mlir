<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Native DLL vs LLVM IR Storage - Detailed Comparison

**Date:** 2026-02-11
**Document Type:** Design
**Review Status:** Draft

---

## Overview

This document compares two approaches for storing compiled models in [EPContext](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html):

1. **Native DLL:** Compile to platform-specific machine code
2. **LLVM IR:** Store LLVM Intermediate Representation, JIT compile at runtime

Both approaches are viable. This comparison helps inform the architectural decision.

---

## Approach 1: Native DLL

### How It Works

```
Compile-time:
  ONNX → MLIR → LLVM IR → Native x64 Machine Code → DLL
  Store DLL bytes in EPContext

Runtime:
  Extract DLL bytes → Load via MemoryModule → Execute directly
```

### Trade-offs

| Aspect | Native DLL |
|--------|-----------|
| **Startup time** | Fast (load from memory, ~5-10ms) |
| **Runtime deps** | [MemoryModule](https://github.com/fancycode/MemoryModule) only (~15KB) |
| **Portability** | GPU arch-specific (gfx1150, gfx1030, etc.) |
| **EPContext size** | Larger (includes code + weights + relocations) |
| **Validation** | Requires arch detection at runtime |
| **JIT overhead** | Zero (pre-compiled) |
| **Code optimization** | Full LLVM optimizations at compile-time |

### Strengths

- **Eliminates JIT overhead entirely** - Primary goal stated in ARCHITECTURE.md "Problem" section
- **Lightweight runtime** - No LLVM libraries needed (potentially 100+ MB saved)
- **Fast loading** - Direct memory mapping, no compilation delay
- **Maximum optimization** - All LLVM optimization passes run at compile-time

### Weaknesses

- **Architecture-specific** - Need separate DLLs for gfx1150, gfx1030, etc.
- **Larger storage** - Machine code is larger than IR
- **Deployment complexity** - Must detect GPU architecture and select correct DLL
- **Update inflexibility** - Changing weights requires full recompilation

### Sub-Decision: Memory DLL Loading vs Disk Files

If Native DLL is chosen, there's a secondary decision about how to load the DLL:

#### Option A: Memory Loading (via MemoryModule)

**How it works:**
- DLL stored as bytes in EPContext
- Load directly from memory buffer using [MemoryModule](https://github.com/fancycode/MemoryModule)
- No disk I/O required

**Trade-offs:**

| Aspect | Impact |
|--------|--------|
| **Deployment** | Single ONNX file (DLL embedded) |
| **Startup** | No disk I/O overhead |
| **Dependencies** | Requires MemoryModule library (~15KB) |
| **Security** | DLL embedded in ONNX (user must trust model file) |
| **WebNN compatibility** | Works with no-disk-access constraint |

**Strengths:**
- Cleaner deployment: No temporary files, no disk permissions needed
- Fast loading: Parse PE and map to memory
- WebNN compatibility: Required for no-disk-access constraint
- Simple integration: MemoryModule is lightweight, [MPL 2.0 license](https://github.com/fancycode/MemoryModule/blob/master/LICENSE.txt)

**Weaknesses:**
- Additional dependency (though small)
- DLL embedded in model file (harder for security scanning)

#### Option B: Disk Files

**How it works:**
- DLL stored as separate file on disk
- Load using standard OS loader
- ONNX model references DLL path

**Trade-offs:**

| Aspect | Impact |
|--------|--------|
| **Deployment** | ONNX file + separate DLL file(s) |
| **Startup** | Requires disk access |
| **Dependencies** | OS loader (zero additional deps) |
| **Security** | Separate DLL (easier to scan/verify) |
| **WebNN compatibility** | Violates no-disk-access constraint |

**Strengths:**
- Zero additional dependencies (use OS loader)
- Easier security scanning (separate DLL file)
- Simpler implementation (standard LoadLibrary)

**Weaknesses:**
- More complex deployment (multiple files)
- Requires disk permissions
- Path management complexity
- Incompatible with WebNN no-disk-access requirement

#### Recommendation

For Native DLL approach:
- **Use Memory Loading** if WebNN compatibility or deployment simplicity is important
- **Use Disk Files** if minimal dependencies or easier security scanning is prioritized
- Most implementations would choose Memory Loading to match EPContext's single-artifact philosophy

---

## Approach 2: LLVM IR

### How It Works

```
Compile-time:
  ONNX → MLIR → LLVM IR (bitcode)
  Store LLVM IR in EPContext

Runtime:
  Extract IR → JIT compile to native code → Execute
```

### Trade-offs

| Aspect | LLVM IR |
|--------|---------|
| **Startup time** | Slow (JIT compilation required, 100-500ms+ depending on model size) |
| **Runtime deps** | Large LLVM/MLIR libraries (50-150 MB) |
| **Portability** | Cross-architecture (compile at runtime for actual GPU) |
| **EPContext size** | Smaller (IR only, no architecture-specific code) |
| **Validation** | No architecture validation needed (compiles for actual hardware) |
| **JIT overhead** | Significant (recompiles on every process start) |
| **Code optimization** | JIT optimizations limited by time budget |

### Strengths

- **Cross-architecture portability** - Single EPContext works on any GPU
- **Smaller storage** - IR is more compact than native code
- **Simpler deployment** - No architecture detection needed
- **Theoretical flexibility** - Could apply runtime optimizations based on actual hardware

### Weaknesses

- **Reintroduces JIT overhead** - Contradicts primary goal of EPContext (eliminate recompilation)
- **Heavy runtime dependencies** - Must ship LLVM libraries
- **Slower startup** - Compilation delay on every process start
- **Limited JIT optimizations** - Time budget constraints prevent full optimization

---

## Hybrid Approach: Support Both

### How It Works

EPContext stores a format tag indicating DLL or IR:

```
Runtime:
  Read format tag from EPContext
  if (format == NATIVE_DLL):
    Load via MemoryModule
  else if (format == LLVM_IR):
    JIT compile then execute
```

### Trade-offs

| Aspect | Impact |
|--------|--------|
| **Flexibility** | Maximum - choose format per model |
| **Runtime complexity** | Doubled - maintain both code paths |
| **Testing burden** | Both paths need coverage |
| **Deployment size** | Full - ship MemoryModule + LLVM libraries |
| **Use cases** | Development (IR) vs Production (DLL) |

### Potential Use Cases

- **Development builds:** Use IR for flexibility during testing
- **Production builds:** Use Native DLL for performance
- **Cross-platform testing:** Use IR to test logic on different GPUs
- **Deployed applications:** Use Native DLL for optimal performance

---

## Decision Criteria

### Choose Native DLL if:

1. **Performance is critical** - Startup time matters
2. **Deployment size matters** - Want minimal runtime dependencies
3. **Target specific hardware** - Know GPU architecture at deployment time
4. **EPContext philosophy alignment** - Goal is to eliminate recompilation overhead

### Choose LLVM IR if:

1. **Portability is critical** - Need single artifact across GPU architectures
2. **Storage size matters** - EPContext size is constrained
3. **Hardware unknown at compile-time** - Don't know target GPU architecture
4. **JIT overhead acceptable** - Startup delay is tolerable

### Choose Hybrid if:

1. **Different use cases exist** - Development vs production have different needs
2. **Migration path needed** - Want to transition gradually
3. **Maximum flexibility required** - Let users choose per-model

---

## Recommendation Framework

Consider your primary goal:

| Primary Goal | Recommendation |
|--------------|----------------|
| Eliminate JIT overhead (per ARCHITECTURE.md "Problem") | **Native DLL** |
| Cross-GPU portability | **LLVM IR** |
| Support both development and production | **Hybrid** |
| Minimize EPContext size | **LLVM IR** |
| Minimize runtime dependencies | **Native DLL** |
| Fastest inference startup | **Native DLL** |

---

## Implementation Impact

### If Native DLL Chosen:

**Required Components:**
- MemoryModule integration for in-memory DLL loading
- GPU architecture detection at runtime
- Architecture-specific compilation pipeline

**Optional Enhancements:**
- Multi-architecture DLL packaging (single EPContext with multiple DLLs)
- Fallback to CPU if architecture mismatch detected

### If LLVM IR Chosen:

**Required Components:**
- LLVM JIT infrastructure at runtime
- LLVM library deployment strategy
- JIT compilation error handling

**Optional Enhancements:**
- JIT compilation caching to disk
- Incremental compilation to reduce startup time

### If Hybrid Chosen:

**Required Components:**
- Format detection in EPContext loading
- Both Native DLL and LLVM IR code paths
- Dual testing infrastructure

**Optional Enhancements:**
- Auto-selection based on environment (dev vs prod)
- Performance telemetry to guide format choice

---

## Open Questions

1. **Storage budget:** What is the acceptable EPContext size limit?
2. **Startup time budget:** What is the acceptable inference initialization delay?
3. **Deployment scenarios:** Single GPU type or multiple?
4. **Development workflow:** How important is cross-architecture testing?
5. **Runtime environment:** Can we assume LLVM libraries available, or must we minimize dependencies?

---

## References

- [ARCHITECTURE.md](ARCHITECTURE.md) - Main architecture document
- [ONNX Runtime EP Context Design](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html)
- [MemoryModule](https://github.com/fancycode/MemoryModule) - In-memory DLL loading
- [LLVM JIT Tutorial](https://llvm.org/docs/tutorial/) - LLVM JIT compilation

---

**Document History:**
- v1.1 (2026-02-11): Added "Memory DLL Loading vs Disk Files" as Native DLL sub-decision
- v1.0 (2026-02-11): Initial comparison document extracted from ARCHITECTURE.md
