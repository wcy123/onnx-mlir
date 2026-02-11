# MLIR Test Files

This directory contains MLIR test inputs for the compilation pipeline.

## Files

### `identity_llvm.mlir`
A minimal MLIR module in LLVM dialect that tests the full compilation pipeline:
- MLIR → LLVM IR translation
- LLVM IR → Object file compilation
- Object file → DLL linking

The module exports three interface functions:
- `inference_init` - Creates runtime state
- `inference_compute` - Runs inference (identity/no-op in this test)
- `inference_cleanup` - Destroys runtime state

## Usage

These files are compiled by `mlir-hip-compiler` during CTest:

```bash
# Build and run tests
cmake -S . -B ../../build/onnx-hipdnn-ep -G Ninja
cmake --build ../../build/onnx-hipdnn-ep
ctest --test-dir ../../build/onnx-hipdnn-ep --output-on-failure
```

## Adding New Tests

1. Create a new `.mlir` file in this directory
2. Ensure it's in LLVM dialect with the interface functions
3. Add corresponding CTest entries in `test/CMakeLists.txt`
