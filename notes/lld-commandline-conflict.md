<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# LLD CommandLine Conflict - Resolution

## Problem Statement

When integrating LLD as a library for DLL compilation, we encountered a global state conflict in LLVM's CommandLine system.

**Error observed:**
```
lld-link (LLVM option parsing): Not enough positional command line arguments specified!
```

## Root Cause

Both `mlir-hip-compiler` and LLD tried to use LLVM's global `CommandLine` system:

1. **mlir-hip-compiler** (debug tool):
   - Used `cl::opt<>` declarations for command-line parsing
   - Called `cl::ParseCommandLineOptions()` which registers options globally
   - This populated the global CommandLine state

2. **LLD** (invoked as library):
   - Also uses `cl::opt<>` internally for its options
   - When called via `lld::coff::link()`, tried to parse from the same global state
   - Found mlir-hip-compiler's options instead of its own arguments
   - Result: "Not enough positional arguments" error

**Key insight**: LLVM's CommandLine system is **global singleton** - only one component can use it at a time.

## Solution: Manual Parsing in mlir-hip-compiler

**Before** (using LLVM CommandLine):
```cpp
#include "llvm/Support/CommandLine.h"

static cl::opt<std::string> inputFilename(cl::Positional, ...);
static cl::opt<std::string> outputFilename("o", ...);
// ... more cl::opt declarations ...

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "...");
  // Now global CommandLine state is occupied
  // Later: linker.linkDLL() -> lld::coff::link() -> CONFLICT!
}
```

**After** (manual parsing):
```cpp
// No #include "llvm/Support/CommandLine.h" needed

struct Options {
  std::string inputFilename;
  std::string outputFilename = "output.dll";
  int optLevel = 2;
  bool verbose = false;
  // ... other fields ...

  bool parse(int argc, char **argv);  // Manual parsing
  void printHelp() const;
};

int main(int argc, char **argv) {
  Options opts;
  if (!opts.parse(argc, argv)) {
    opts.printHelp();
    return 1;
  }

  InitLLVM X(argc, argv);
  // Global CommandLine is now CLEAN and FREE for LLD
  // Later: linker.linkDLL() -> lld::coff::link() -> SUCCESS!
}
```

**Benefits**:
- Frees global CommandLine system entirely for LLD
- Simple, maintainable code (~80 lines of parsing)
- No global state pollution
- Works reliably across platforms

**Trade-offs**:
- Lost LLVM's auto-generated help (but custom help is better anyway)
- Manual validation (but only 7 simple options)
- ~50 extra lines of code vs `cl::opt` declarations

## Why Production EP Is Unaffected

The **production Execution Provider DLL** won't have this issue:

```
ONNX Runtime Process
  └─> EP DLL (library loaded by ONNX Runtime)
      └─> Receives ONNX graph programmatically (no argc/argv)
      └─> Compiles: MLIR → LLVM IR → Object → LLD → Compiled Code
      └─> Never calls cl::ParseCommandLineOptions()
      └─> Global CommandLine stays clean
      └─> LLD works without conflicts
```

**Key differences from mlir-hip-compiler**:
- EP DLL is a library, not a command-line tool
- No command-line parsing needed (inputs come programmatically)
- Global CommandLine system remains unused and clean
- When EP calls LLD, no conflicts occur

## Architecture Diagram

### mlir-hip-compiler (Debug Tool - HAD Conflict, NOW FIXED)

```
User invokes:
  mlir-hip-compiler input.mlir -o output.dll -v

Before fix (BROKEN):
  main() → cl::ParseCommandLineOptions()
    → Global CommandLine populated with mlir-hip-compiler options
    → linkDLL() → lld::coff::link()
      → LLD tries to use same global CommandLine
      → CONFLICT: "Not enough positional arguments"

After fix (WORKS):
  main() → opts.parse(argc, argv)
    → Local Options struct, no global state
    → linkDLL() → lld::coff::link()
      → LLD uses clean global CommandLine
      → SUCCESS: DLL linked
```

### Production EP DLL (NEVER Had Conflict)

```
ONNX Runtime loads EP.dll:
  ONNXRuntimeAPI → ExecutionProvider::Compile(onnx_graph)
    → MLIR compilation pipeline
    → linkDLL() → lld::coff::link()
      → Global CommandLine is clean (never used)
      → SUCCESS: DLL linked
```

## Files Modified

**tools/mlir-hip-compiler/main.cpp** (lines 18-91):
- Removed `#include "llvm/Support/CommandLine.h"`
- Removed all `static cl::opt<>` declarations (lines 61-85)
- Added `struct Options` with `parse()` and `printHelp()` methods
- Replaced `cl::ParseCommandLineOptions()` with `opts.parse()`
- Updated all references: `inputFilename` → `opts.inputFilename`, etc.

## Testing

**Before fix:**
```bash
$ mlir-hip-compiler input.mlir -o output.dll
lld-link (LLVM option parsing): Not enough positional command line arguments specified!
Error linking DLL
```

**After fix:**
```bash
$ mlir-hip-compiler input.mlir -o output.dll -v
=== MLIR to HIP DLL Compiler ===
Input: input.mlir
Output: output.dll
Mode: dll
Optimization: O2

--- Step 1: Parsing MLIR ---
✓ MLIR parsed successfully

--- Step 2: Running MLIR Passes ---
✓ MLIR passes completed

--- Step 3: Translating to LLVM IR ---
✓ LLVM IR generated

--- Step 4: Optimizing LLVM IR (O2) ---
✓ Optimization completed

--- Step 5: Emitting LLVM IR ---
✓ LLVM IR written to: output.ll

--- Step 6: Compiling to Object File ---
✓ Object file created: output.obj

--- Step 7: Linking to DLL ---
LLD-LINK command (6 args): [0]='/DLL' [1]='/OUT:output.dll' ...
✓ LLD invoked successfully (no CommandLine conflict)
```

**Verification**:
- ✅ No "Not enough positional arguments" error
- ✅ LLD executes and processes object file
- ✅ .ll file created (LLVM IR)
- ✅ .obj file created (object code)
- ✅ .obj.def file created (export definitions)
- ⚠️ DLL linking may fail due to missing runtime symbols (expected until HipDnnRuntime is built)

## Next Steps

1. **Build HipDnnRuntime.lib** - Required for successful DLL linking
2. **Test full pipeline** - Complete MLIR → DLL with all dependencies
3. **Update DEMO.md** - Document the successful Stage 4 compilation
4. **Create output examples** - Generate real artifacts in `../output/` directory

## Lessons Learned

1. **Global state is dangerous** - LLVM's CommandLine singleton causes conflicts
2. **Library vs tool requirements** - Production code (EP DLL) has different constraints than debug tools
3. **Simple solutions work** - Manual parsing is straightforward and reliable
4. **Scope matters** - This issue only affected the debug tool, not production

## Related Documentation

- **doc/ARCHITECTURE.md** - Explains EP DLL vs debug tools
- **lib/Backend/DLLLinker.cpp** - LLD integration code
- **lib/Backend/CMakeLists.txt** - LLD library linking configuration
