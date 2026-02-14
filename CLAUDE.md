<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

**ONNX HIP/DNN Execution Provider**: ONNX Runtime Execution Provider with MLIR-based compilation targeting AMD GPUs via HIP/MIOpen.

**Architecture**: MLIR dialect compilation → HIP kernels → DLL generation → ONNX Runtime integration.

## Build System

**CRITICAL**: Non-standard build paths:
- Build: `../../build/$(basename $PWD)` (NOT `./build`)
- Install: `../../local`
- Runtime: `/MTd` via `CMAKE_MSVC_RUNTIME_LIBRARY`

**Configure**:
```bash
# CRITICAL: CMAKE_PREFIX_PATH must be absolute path (relative paths fail)
# CRITICAL: CMAKE_PROGRAM_PATH must point to clang/llvm-link (adjust to your LLVM install)
LOCAL_DIR=$(cd ../../local && pwd)
cmake -S . -B ../../build/$(basename $PWD) -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_PROGRAM_PATH="C:/LLVM20/bin" \
  -DONNX_MLIR_BUILD_TESTS=OFF \
  -DBUILD_HIP_OPT_TOOL=ON \
  -DONNX_HIP_INCLUDE_LIT_TESTS=ON \
  --fresh
```

**Note**: `CMAKE_PROGRAM_PATH` specifies where to find `clang` and `llvm-link` executables.
Adjust path based on your system (see README.md for full build instructions).

**Note**: `ONNX_MLIR_BUILD_TESTS=OFF` disables onnx-mlir tests (requires LLVM test utilities not installed).

**Build**: `cmake --build ../../build/$(basename $PWD) --config Debug --parallel`

**Test**: `ctest --test-dir ../../build/$(basename $PWD) --verbose`

**MSVC Setup** (if headers missing):
```bash
cmd /c "call \"\"C:\\msvsn2022\\VC\\Auxiliary\\Build\\vcvars64.bat\"\" && cd /d %CD% && cmake --build \"\"../../build/$(basename $PWD)\"\" --config Debug --parallel"
```

## Architecture

**Layers**: MLIR dialects → mlir-hip-compiler → HIP backend → DLL generation → ONNX Runtime EP

**Design Patterns**:
1. **MLIR Compilation**: Transform MLIR dialects to HIP/LLVM IR
2. **Backend Integration**: HIP runtime, MIOpen library integration
3. **Testing**: End-to-end tests with CTest integration

**Key Directories**: `lib/Backend/`, `lib/Conversion/`, `tools/mlir-hip-compiler/`, `test/`

## Testing

**Test Structure**:
- `test/lit/` - LIT-based unit tests for MLIR passes (fast, focused)
- `test/e2e/` - End-to-end integration tests (slow, compile + execute)
- `test/runtime/` - Runtime unit tests (optional, requires GTest)

**LIT Tests** (MLIR pass testing):
```bash
# All LIT tests
ctest --test-dir ../../build/$(basename $PWD) -R LitTests --verbose

# Direct LIT execution (more detailed output)
llvm-lit -v test/lit/

# Specific category
llvm-lit -v test/lit/Conversion/onnx-to-hip/

# Single test
llvm-lit -v test/lit/Conversion/onnx-to-hip/test_gemm_basic.mlir

# Via CMake target
cmake --build ../../build/$(basename $PWD) --target check-onnx-hip-lit
```

**E2E Tests** (integration testing):
```bash
# All E2E tests
ctest --test-dir ../../build/$(basename $PWD) -R "CompileDemoConvDLL|TestDemoConvDLL" --verbose

# All tests (LIT + E2E)
ctest --test-dir ../../build/$(basename $PWD) --verbose
```

**Prerequisites**:
- `llvm-lit`: Install via `pip install lit` (required for LIT tests)
- `FileCheck`: Provided by LLVM installation
- `hip-opt`: Built by project (`BUILD_HIP_OPT_TOOL=ON`)

**Writing New Tests**: See `test/lit/README.md` for LIT test writing guide

## Git Workflow

**Remotes**:
- `origin` / `fork` = `https://github.com/wcy123/onnx-hipdnn-ep.git` (your fork, push here)
- `upstream` = `https://github.com/ROCm/onnx-hipdnn-ep.git` (main repository, read-only)

**Push Policy**: After creating commits, push to origin/fork when ready.

**Required Steps**:
1. Sync: `git checkout main && git pull upstream main`
2. Branch: `git checkout -b feature/<name>` (BEFORE changes)
3. Commit: After file changes
4. Push: `git push origin <branch>` (or `git push fork <branch>`, they're the same)
5. Create PR manually when ready

**CRITICAL - Before Marking PR Ready**:
Before marking PR ready for review, MUST run pre-commit to fix formatting issues:
```bash
pre-commit run --all-files
```
If pre-commit makes changes (formatting, linting), commit and push them BEFORE marking PR ready. This prevents CI pre-commit check failures.

**PR Title Format**:
- **For backlog issues**: `Issue #NNN: <type>: <description>`
  - Example: `Issue #022: refactor: remove fix_info dead code`
  - Example: `Issue #023: feat: migrate v1 to v2 execution provider API`
- **For new features/other work**: `<type>: <description>` (no issue number)
  - Example: `feat: add new optimization pass`

**PR Operations**:
- ✅ `gh pr view <number>` (most reliable)
- ✅ `gh pr list` (to find PR number first)

**Commit Rules**:
- ❌ NO AI mentions (Co-Authored-By: Claude, etc.)
- ✅ Conventional commits (`feat:`, `fix:`, `docs:`, etc.)
- ✅ Stage specific files: `git add <file>` (NOT `-A` or `.`)

## Setup

**Pre-commit** (required): `scripts/setup-dev-env.ps1` (Windows) or `scripts/setup-dev-env.sh` (Linux/Mac)

## Common Pitfalls

1. Build dir: `../../build/$(basename $PWD)`, NOT `./build`
2. Install prefix: `../../local`
3. Git push: `origin` or `fork` (they're the same)
4. Never work on `main` branch
5. Run `scripts/setup-dev-env.*` before contributing
6. Launch bash from MSVC Developer Command Prompt (Windows)
7. NO AI/tool mentions in commits/PRs
8. Pre-commit hooks don't enforce when committing via Claude Code - run `pre-commit run --all-files` after commits to verify

## Dependencies

**Required**: LLVM/MLIR, ONNX Runtime, HIP Runtime, MIOpen
**Optional**: GTest

### Building LLVM/MLIR with LLD Support

**CRITICAL**: LLVM must be built with matching runtime library and LLD support for DLL compilation.

**Configuration** (from llvm-project directory):
```bash
cmake -S llvm -B ../../build/llvm-project \
  -DLLVM_ENABLE_PROJECTS="mlir;lld" \
  -DCMAKE_INSTALL_PREFIX=/c/Develop/m/local \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=OFF \
  -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded\$<\$<CONFIG:Debug>:Debug>"
```

**Build**:
```bash
# Try parallel build first (much faster)
cmake --build ../../build/llvm-project --config Debug

# If link errors occur (C1041 PDB conflicts), use --parallel 1 for linking phase only
# Then resume parallel build for remaining targets
cmake --build ../../build/llvm-project --config Debug --parallel 1  # if needed
cmake --build ../../build/llvm-project --config Debug              # resume parallel
```

**Install**:
```bash
cmake --install ../../build/llvm-project --config Debug
```

**Critical Settings**:
- `LLVM_ENABLE_PROJECTS="mlir;lld"` - Required for DLL linking (lldCOFF, lldELF, lldCommon)
- `CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>"` - Must match project's /MTd runtime
- `BUILD_SHARED_LIBS=OFF` - Static libraries only
- Parallel build preferred; only use `--parallel 1` if C1041 PDB file conflicts occur during linking
