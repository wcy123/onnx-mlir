// MLIR test: Identity function in LLVM dialect
// Tests the MLIR → LLVM IR → Object → DLL pipeline
//
// This module defines the inference interface functions that will be exported
// from the generated DLL. It's a minimal test case that exercises the full
// compilation pipeline without complex computation.

module {
  // Declare runtime helper functions (provided by lib/Runtime)
  llvm.func @runtime_state_init(!llvm.ptr) -> i32
  llvm.func @runtime_state_cleanup(!llvm.ptr) -> i32
  llvm.func @runtime_prepare_inference(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @runtime_cleanup_inference(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

  // Main computation function (no-op for this test)
  llvm.func @main(%input: !llvm.ptr, %output: !llvm.ptr) {
    llvm.return
  }

  // Constant management (no constants in this test)
  llvm.func @get_constant_count() -> i64 {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    llvm.return %c0 : i64
  }

  llvm.func @initialize_constants(%state: !llvm.ptr) -> i32 {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %c0 : i32
  }

  llvm.func @release_constants(%state: !llvm.ptr) -> i32 {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %c0 : i32
  }

  // === Exported Interface Functions ===

  // inference_init: Creates runtime state
  llvm.func @inference_init(%out_state: !llvm.ptr) -> i32 {
    %result = llvm.call @runtime_state_init(%out_state) : (!llvm.ptr) -> i32
    llvm.return %result : i32
  }

  // inference_cleanup: Destroys runtime state
  llvm.func @inference_cleanup(%state: !llvm.ptr) -> i32 {
    %result = llvm.call @runtime_state_cleanup(%state) : (!llvm.ptr) -> i32
    llvm.return %result : i32
  }

  // inference_compute: Runs the model
  llvm.func @inference_compute(%state: !llvm.ptr, %inputs_ptr: !llvm.ptr, %outputs_ptr: !llvm.ptr) -> i32 {
    // Allocate space for InferenceData pointer
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %data_alloc = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr

    // Prepare inference (allocates GPU buffers, copies inputs)
    %prep_result = llvm.call @runtime_prepare_inference(%state, %inputs_ptr, %outputs_ptr, %data_alloc)
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

    // Check for errors
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
    %prep_failed = llvm.icmp "ne" %prep_result, %c0_i32 : i32
    llvm.cond_br %prep_failed, ^error, ^continue

  ^error:
    llvm.return %prep_result : i32

  ^continue:
    // Load InferenceData pointer
    %data = llvm.load %data_alloc : !llvm.ptr -> !llvm.ptr

    // For this test, skip calling @main (identity = no-op)

    // Cleanup inference (copies outputs, frees GPU buffers)
    %cleanup_result = llvm.call @runtime_cleanup_inference(%state, %data, %outputs_ptr)
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

    llvm.return %cleanup_result : i32
  }
}
