<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# onnx-hipdnn-ep

**ONNX Runtime Execution Provider with MLIR-based AOT compilation for AMD ROCm**

An MLIR-based ahead-of-time (AOT) compilation pipeline for ONNX Runtime, targeting AMD GPUs via HIP and MIOpen. This project integrates ONNX-to-MLIR conversion with HIP dialect compilation to enable zero-cost abstraction and EPContext caching.

## Current Status

This is an active integration branch (`mlir-integration-1`) combining:
- **PR #1**: ONNX → MLIR conversion (MorphiZen framework)
- **PR #4**: HIP MLIR dialect (hip-opt)

**Progress:**
- ✅ PR #1 baseline integrated (ONNX → MLIR parsing)
- ⏳ PR #4 hip-opt integration (in progress)
- ⏳ Pattern-based lowering (ONNX-MLIR → HIP dialect)
- ⏳ Native compilation pipeline

## Key Features

- **AOT Compilation**: Compile ONNX graphs to optimized HIP kernels ahead-of-time
- **Zero-Cost Abstraction**: MLIR-based transformations with no runtime overhead
- **EPContext Caching**: Cache compiled artifacts for fast startup
- **Minimal Design**: No pattern matching or protobuf dependencies in core pipeline
- **Clean Architecture**: Level-1 pass structure based on MorphiZen framework

## Quick Start

### Prerequisites

- CMake 3.28+
- Visual Studio 2022 (Windows) or C++17 compiler
- Python 3 with onnx package: `pip install onnx`
- Git
- Clang (LLVM 19+ with llvm-link)

### Build

From workspace root directory:

```bash
# 1. Build ONNXRuntime (required dependency)
git clone https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
./build.bat --config Release --build_shared_lib --parallel \
  --skip_submodule_sync --build_dir ../build/onnxruntime \
  --skip_tests --cmake_extra_defines CMAKE_INSTALL_PREFIX=$PWD/../local
cmake --build ../build/onnxruntime/Release/ --target install

# 2. Build onnx-hipdnn-ep
cd ../onnx-hipdnn-ep
git submodule update --init --recursive
cmake -S . -B ../build/onnx-hipdnn-ep \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL \
  -DCMAKE_INSTALL_PREFIX=../local \
  -DCMAKE_PREFIX_PATH=$PWD/../local
cmake --build ../build/onnx-hipdnn-ep --config Release --parallel
```

**Note:** First build takes 1-3 hours (LLVM/MLIR auto-fetch). Subsequent builds are much faster.

### Test

```bash
# Generate test models
cd test
python gen_conv_model.py && python gen_conv_gemm_model.py
cp *.onnx ../../build/onnx-hipdnn-ep/bin/Release/

# Run tests
cd ../../build/onnx-hipdnn-ep/bin/Release/
./ort_integration_test
```

## Documentation

- **[Building & Testing Guide](doc/BUILDING.md)** - Comprehensive build instructions, configuration options, and testing
- **[Architecture](doc/design/ARCHITECTURE.md)** - System architecture and design decisions

## Project Structure

```
onnx-hipdnn-ep/
├── CMakeLists.txt              # Root build configuration
├── level-1-pass-mlir/          # MLIR compiler (ONNX → MLIR → HIP → LLVM)
├── test/                       # Integration tests
├── doc/                        # Documentation
│   ├── BUILDING.md             # Build & testing guide
│   └── design/                 # Architecture docs
├── 3rd-party/morphizen/        # MorphiZen framework (submodule)
└── etc/morphizen_config.json   # MorphiZen pass configuration
```

## License

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

Licensed under the MIT License. See [LICENSE](LICENSE) file for details.
