// Manually generated interface for test_identity
// This bypasses the GenerateInterfacePass for testing purposes
// In production, these functions would be auto-generated

module {
  // Declare runtime helper functions
  llvm.func @runtime_state_init(!llvm.ptr) -> i32
  llvm.func @runtime_state_cleanup(!llvm.ptr) -> i32
  llvm.func @runtime_prepare_inference(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @runtime_cleanup_inference(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @hip_upload_constant(!llvm.ptr, i64, !llvm.ptr, i64) -> i32

  // Main computation function (no-op for testing)
  llvm.func @main(%input: !llvm.ptr, %output: !llvm.ptr) {
    llvm.return
  }

  // Constant management functions
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

  // Interface function: inference_init
  // Creates runtime state and initializes constants
  llvm.func @inference_init(%out_state: !llvm.ptr) -> i32 {
    %result = llvm.call @runtime_state_init(%out_state) : (!llvm.ptr) -> i32
    llvm.return %result : i32
  }

  // Interface function: inference_cleanup
  // Releases constants and destroys runtime state
  llvm.func @inference_cleanup(%state: !llvm.ptr) -> i32 {
    %result = llvm.call @runtime_state_cleanup(%state) : (!llvm.ptr) -> i32
    llvm.return %result : i32
  }

  // Interface function: inference_compute
  // Simplified version that calls runtime helpers and @main
  llvm.func @inference_compute(%state: !llvm.ptr, %inputs_ptr: !llvm.ptr, %outputs_ptr: !llvm.ptr) -> i32 {
    // Allocate space for InferenceData pointer
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %data_alloc = llvm.alloca %c1 x !llvm.ptr : (i64) -> !llvm.ptr

    // Call runtime_prepare_inference
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

    // For this test, we skip calling @main since we don't have GPU buffers extraction
    // In a real implementation, we would extract buffers from InferenceData and call @main

    // Call runtime_cleanup_inference
    %cleanup_result = llvm.call @runtime_cleanup_inference(%state, %data, %outputs_ptr)
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

    llvm.return %cleanup_result : i32
  }
}
