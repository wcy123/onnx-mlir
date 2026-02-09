# CLI Option Conflict Analysis

## Problem Statement

When attempting to run `hip-opt` tool with ONNX-MLIR integration, we encounter:

```
CommandLine Error: Option 'o' registered more than once!
LLVM ERROR: inconsistency in registered CommandLine options
```

## Root Cause: Transitive Dependency Chain

The conflicting `-o` option is pulled in through this dependency chain:

```
hip-opt (executable)
  └─> HipDialect (library)
       └─> OMONNXOps (onnx-mlir ONNX dialect)
            └─> OMMlirDialects (onnx-mlir MLIR utilities)
                 └─> OMCompilerOptions (contains CLI options)
                      └─> CompilerOptions.cpp (registers -o option)
```

## Evidence

### 1. Conflicting Option Location

**File:** `3rd-party/onnx-mlir/src/Compiler/CompilerOptions.cpp`

```cpp
static llvm::cl::opt<std::string, true> outputBaseNameOpt("o",
    llvm::cl::desc("For onnx-mlir, specify the base path for output file..."));
```

This conflicts with `mlir-opt`'s own `-o` option for output files.

### 2. Library Dependencies

**`OMCompilerOptions` library** (`3rd-party/onnx-mlir/src/Compiler/CMakeLists.txt`):
```cmake
add_onnx_mlir_library(OMCompilerOptions
  CompilerOptions.cpp
  EXCLUDE_FROM_OM_LIBS
  ...
)
```

**`OMMlirDialects` library** (`3rd-party/onnx-mlir/src/Dialect/Mlir/CMakeLists.txt`):
```cmake
add_onnx_mlir_library(OMMlirDialects
  IndexExpr.cpp
  IndexExprBuilder.cpp
  DialectBuilder.cpp
  VectorMachineSupport.cpp

  LINK_LIBS PUBLIC
  OMCompilerOptions  # <-- Links to OMCompilerOptions
  ...
)
```

**`OMONNXOps` library** (`3rd-party/onnx-mlir/src/Dialect/ONNX/CMakeLists.txt`):
```cmake
add_onnx_mlir_library(OMONNXOps
  ...
  LINK_LIBS PUBLIC
  OMMlirDialects  # <-- Links to OMMlirDialects
  OMShapeHelperOpInterface
  OMONNXElementsAttr
  ...
)
```

**`HipDialect` library** (`lib/HipDialect/CMakeLists.txt`):
```cmake
target_link_libraries(HipDialect PUBLIC
  ...
  OMONNXOps  # <-- Links to OMONNXOps (required for ONNXConvOp)
  ...
)
```

### 3. Why the Dependency Exists

**Q: Why does `HipDialect` need `OMONNXOps`?**

A: For the ONNX→HIP conversion pass in `lib/HipDialect/OnnxToHip.cpp`:

```cpp
#include "src/Dialect/ONNX/ONNXOps.hpp"

struct ConvToHipPattern : public OpConversionPattern<ONNXConvOp> {
  // Pattern matches ONNXConvOp type - requires ONNX dialect
  LogicalResult matchAndRewrite(...) { ... }
};
```

**Q: Why does `OMMlirDialects` need `OMCompilerOptions`?**

A: `OMMlirDialects` contains utilities like `IndexExpr.cpp` and `DialectBuilder.cpp` that may reference global compiler options (e.g., `onnx_mlir::mcpu`, `onnx_mlir::march`) defined in `CompilerOptions.cpp`.

However, this dependency might not be strictly necessary for our use case.

## LLVM Command-Line Option Mechanism

LLVM uses **static global constructors** for option registration:

```cpp
// In CompilerOptions.cpp (onnx-mlir)
static llvm::cl::opt<std::string, true> outputBaseNameOpt("o", ...);
//     ^^^^^^ static global - constructor runs BEFORE main()

// In mlir-opt tool (LLVM/MLIR)
static llvm::cl::opt<std::string> outputFilename("o", ...);
//     ^^^^^^ static global - constructor runs BEFORE main()
```

**Execution flow:**
1. Program starts
2. Static constructors run (register all `llvm::cl::opt` options)
3. LLVM detects two `-o` registrations
4. LLVM aborts: "Option 'o' registered more than once!"
5. `main()` never executes

**Result:** No way to "disable" options at runtime - they're already registered before `main()`.

## Potential Solutions

### Option 1: Remove OMCompilerOptions Dependency (Needs Investigation)

**Approach:** Check if `OMMlirDialects` actually uses anything from `OMCompilerOptions`.

**Steps:**
1. Search `OMMlirDialects` source files for references to `onnx_mlir::` namespace
2. If no references found, try removing the dependency
3. Rebuild and test

**Risk:** May break onnx-mlir build if dependency is actually needed.

### Option 2: Split OMMlirDialects (Invasive)

**Approach:** Create a lighter version of `OMMlirDialects` without `OMCompilerOptions` dependency.

**Risk:** Requires modifying onnx-mlir CMake structure - high maintenance burden.

### Option 3: Programmatic API Usage (Current Workaround)

**Approach:** Don't use `hip-opt` CLI tool - use PassManager API directly.

**Implementation:**
```cpp
// Standalone test program
int main(int argc, char **argv) {
  MLIRContext context;
  auto module = parseMLIR(argv[1], &context);

  PassManager pm(&context);
  pm.addPass(createConvertOnnxToHipPass());
  pm.addPass(createConvertHipToLLVMPass());
  pm.run(module);

  module->print(llvm::outs());
}
```

**Advantages:**
- Avoids CLI option registration conflicts
- Standard approach for production use (Level-1 Pass will use this)
- No modification to onnx-mlir needed

**Disadvantages:**
- Cannot use `hip-opt` CLI tool for debugging
- Requires separate test executable

### Option 4: Conditional Option Registration (Requires onnx-mlir Changes)

**Approach:** Modify onnx-mlir to conditionally register options.

**Example:**
```cpp
// In onnx-mlir CompilerOptions.cpp
#ifndef ONNX_MLIR_DISABLE_CLI_OPTIONS
static llvm::cl::opt<std::string, true> outputBaseNameOpt("o", ...);
#endif
```

**Risk:** Requires upstreaming changes to onnx-mlir project.

## Current Status

**Documented in:** `tools/hip-opt/TEST_RESULTS.md`

**Conclusion:**
- Both conversion passes compile successfully ✅
- CLI tool conflict prevents command-line testing ⚠️
- Programmatic API usage (Option 3) is the recommended workaround
- Level-1 Pass will use PassManager API directly (no CLI dependency)

**Next Steps:**
1. Investigate if `OMMlirDialects` truly needs `OMCompilerOptions` (Option 1)
2. If yes, implement Option 3 (standalone test program)
3. Update DEMO.md with actual transformation outputs
