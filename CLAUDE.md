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
LOCAL_DIR=$(cd ../../local && pwd)
cmake -S . -B ../../build/$(basename $PWD) -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON --fresh
```

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

**Key Directories**: `lib/Backend/`, `lib/Conversion/`, `tools/mlir-hip-compiler/`, `test/e2e/`

## Testing

**Framework**: CTest, end-to-end tests
**Run**: `ctest --test-dir ../../build/$(basename $PWD) -R TestName --verbose`

## Git Workflow

**Remotes**: `origin` (main repository, read-only) | `fork` (your fork, push here)

**CRITICAL**: Always push to `fork`, never `origin`

**Push Policy**: After creating commits, ALWAYS push to fork immediately unless the user says otherwise.

**Auto-PR Policy**: After successfully pushing to fork, IMMEDIATELY check if a PR exists for the branch. If not, create a draft PR automatically with `gh pr create --draft`.

**Required Steps**:
1. Sync: `git checkout main && git pull origin main`
2. Branch: `git checkout -b feature/<name>` (BEFORE changes)
3. Commit: After file changes, BEFORE testing
4. First Push: `git push -u fork <branch>` (sets upstream tracking)
5. Subsequent Pushes: `git push fork <branch>`
6. PR: `gh pr create --draft` (IMMEDIATELY after first push, auto-create if no PR exists)

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

**PR Operations** (fork-based workflow):
- ❌ `gh pr view` (fails - branch not in origin)
- ✅ `gh pr view <number>` (most reliable)
- ✅ `gh pr view <owner>:<branch>` (e.g., `gh pr view your-username:feature/name`)
- ✅ `gh pr list` (to find PR number first)

**Why `gh pr view` fails**: PRs are cross-repository (fork → origin). `gh pr view` searches for the branch in the current repo (origin), but the branch only exists in the fork.

**Commit Rules**:
- ❌ NO AI mentions (Co-Authored-By: Claude, etc.)
- ✅ Conventional commits (`feat:`, `fix:`, `docs:`, etc.)
- ✅ Stage specific files: `git add <file>` (NOT `-A` or `.`)

## Setup

**Pre-commit** (required): `scripts/setup-dev-env.ps1` (Windows) or `scripts/setup-dev-env.sh` (Linux/Mac)

## Common Pitfalls

1. Build dir: `../../build/$(basename $PWD)`, NOT `./build`
2. Install prefix: `../../local`
3. Git push: `fork`, NEVER `origin`
4. Never work on `main` branch
5. Run `scripts/setup-dev-env.*` before contributing
6. Launch bash from MSVC Developer Command Prompt (Windows)
7. NO AI/tool mentions in commits/PRs
8. Pre-commit hooks don't enforce when committing via Claude Code - run `pre-commit run --all-files` after commits to verify

## Dependencies

**Required**: LLVM/MLIR, ONNX Runtime, HIP Runtime, MIOpen
**Optional**: GTest
