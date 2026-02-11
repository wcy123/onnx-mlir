# MLIR to DLL Pipeline - Quick Reference

## 🚀 Quick Start

```bash
# Build
cmake -B build -S .
cmake --build build --config Release

# Set mode
export COMPILATION_MODE=native  # or "ir"

# Test
./build/bin/test/EndToEndTest inference.dll
```

## 🔧 Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `COMPILATION_MODE` | `ir`, `native` | `native` | Compilation mode |
| `OUTPUT_PATH` | any string | `inference` | Output file name |
| `MLIR_PRINT_WITH_VERBOSE` | `0`, `1` | `0` | Verbose MLIR output |

## 📁 File Locations

```
lib/
├── Runtime/           # GPU runtime wrappers
├── Backend/           # LLVM Backend + DLL Linker
└── HipDialect/        # MLIR passes (GenerateInterfacePass)

level-1-pass-mlir-compiler/src/
└── pass_main.cpp      # Compiler driver

test/integration/
├── EndToEndTest.cpp           # Integration test
├── verify_dll_exports.bat     # Windows verification
└── verify_dll_exports.sh      # Linux verification
```

## 🎯 Exported Functions

```c
// Initialize GPU resources (returns 0 on success, 1-5 on error)
int inference_init(void** out_state);

// Execute inference (returns 0 on success, 5+ on error)
int inference_compute(void* state, span_t* inputs, span_t* outputs);

// Cleanup GPU resources (returns 0 always - best effort)
int inference_cleanup(void* state);
```

## 📊 Data Structures

```c
struct tensor_t {
    void* data;        // CPU pointer
    int64_t* shape;    // Dimension array
    size_t rank;       // Number of dimensions
};

struct span_t {
    tensor_t* data;    // Tensor array
    size_t count;      // Number of tensors
};
```

## 🏗️ Build Dependencies

### Required
- CMake 3.18+
- LLVM 18+ (with MLIR and LLD)
- ROCm 5.7+ (HIP, MIOpen, hipBLASLt)
- C++17 compiler

### Components Built
- `libHipDnnRuntime.a` - Runtime library
- `libLLVMBackend.a` - LLVM backend
- `morphizen-level1-pass-mlir-compiler` - Compiler

## 🧪 Testing Commands

```bash
# Verify exports (Windows)
test/integration/verify_dll_exports.bat inference.dll

# Verify exports (Linux)
test/integration/verify_dll_exports.sh inference.so

# Run integration test
build/bin/test/EndToEndTest inference.dll

# Manual DLL inspection (Windows)
dumpbin /EXPORTS inference.dll

# Manual SO inspection (Linux)
nm -D inference.so | grep inference
```

## 📈 Compilation Flow

### IR Mode
```
ONNX → MLIR → LLVM IR (.ll file)
```

### Native Mode
```
ONNX → MLIR → LLVM IR → Object File → DLL
```

## 🎨 Key Components

| Component | Purpose | LOC |
|-----------|---------|-----|
| Runtime Library | GPU wrappers | 480 |
| LLVM Backend | IR + Native compilation | 570 |
| GenerateInterfacePass | C interface | 900 |
| DLL Linker | LLD integration | 310 |
| Compiler Driver | Orchestration | 150 |

## 🔍 Debugging

```bash
# Print MLIR at each stage
export MLIR_PRINT_WITH_VERBOSE=1

# Generate IR for inspection
export COMPILATION_MODE=ir

# Check LLVM IR output
cat inference.ll

# Verify DLL symbols
dumpbin /EXPORTS inference.dll  # Windows
nm -D inference.so               # Linux
```

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| Cannot find LLD | Build LLVM with `-DLLVM_ENABLE_PROJECTS=lld` |
| ROCm not found | Add ROCm to `CMAKE_PREFIX_PATH` |
| DLL won't load | Add ROCm bin to PATH |
| Symbols missing | Check .def file generation |

## 📚 Documentation

- `IMPLEMENTATION_GUIDE.md` - Full implementation guide
- `IMPLEMENTATION_SUMMARY.md` - Implementation summary
- `QUICK_REFERENCE.md` - This file

## 🔧 CMake Options

```bash
cmake -B build \
  -DBUILD_HIP_DIALECT=ON \
  -DBUILD_MLIR_COMPILER=ON \
  -DCMAKE_BUILD_TYPE=Release
```

## 🎯 Error Codes

### inference_init
- `0` - Success
- `1` - Allocation failed
- `2` - Stream creation failed
- `3` - MIOpen creation failed
- `4` - Set stream failed
- `5` - hipBLAS creation failed

### inference_compute
- `0` - Success
- `5` - Validation failed
- `6+` - Runtime errors

### inference_cleanup
- `0` - Success (always)

## 🚀 Performance Tips

```bash
# Optimization levels
-O0  # No optimization (debug)
-O1  # Basic optimization
-O2  # Default (recommended)
-O3  # Aggressive optimization

# Set in LLVM backend
optimizeLLVMIR(module, 2);  // O2
```

## 📝 Example Usage

```cpp
// Load DLL
HMODULE dll = LoadLibrary("inference.dll");
auto init = (inference_init_fn)GetProcAddress(dll, "inference_init");

// Initialize
void* state;
init(&state);

// Prepare data
tensor_t input = {data, shape, rank};
span_t inputs = {&input, 1};

// Compute
compute(state, &inputs, &outputs);

// Cleanup
cleanup(state);
FreeLibrary(dll);
```

## 🎓 Learning Path

1. Read `IMPLEMENTATION_GUIDE.md` for architecture
2. Study `EndToEndTest.cpp` for usage
3. Examine `GenerateInterfacePass.cpp` for MLIR
4. Review `LLVMBackend.cpp` for compilation
5. Check `DLLLinker.cpp` for linking

## 📊 Project Stats

- **Total Files:** 21
- **Total LOC:** ~3,500
- **Completion:** 95%
- **Testing:** Infrastructure ready

## 🎉 Quick Win

```bash
# Generate IR in 3 commands
export COMPILATION_MODE=ir
cmake --build build
# Run your ONNX model
# Output: inference.ll
```

---

**Quick Help:** See `IMPLEMENTATION_GUIDE.md` for detailed documentation
