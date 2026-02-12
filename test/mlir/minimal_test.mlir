// Minimal test: Simple DLL with exports matching current runtime API
// Tests the MLIR → LLVM IR → Object → DLL pipeline
//
// This module defines minimal inference functions that use the actual
// runtime API (hipdnn_ep_* functions) instead of the old runtime_* names.

module {
  // Declare actual runtime functions (provided by lib/Runtime/hipdnn_ep_runtime.cpp)
  llvm.func @hipdnn_ep_state_init(!llvm.ptr) -> i32
  llvm.func @hipdnn_ep_state_cleanup(!llvm.ptr) -> i32

  // === Exported Interface Functions ===

  // inference_init: Creates runtime state
  llvm.func @inference_init(%out_state: !llvm.ptr) -> i32 attributes { sym_visibility = "public" } {
    %result = llvm.call @hipdnn_ep_state_init(%out_state) : (!llvm.ptr) -> i32
    llvm.return %result : i32
  }

  // inference_cleanup: Destroys runtime state
  llvm.func @inference_cleanup(%state: !llvm.ptr) -> i32 attributes { sym_visibility = "public" } {
    %result = llvm.call @hipdnn_ep_state_cleanup(%state) : (!llvm.ptr) -> i32
    llvm.return %result : i32
  }

  // inference_compute: Minimal computation (just returns success)
  llvm.func @inference_compute(%state: !llvm.ptr, %inputs: !llvm.ptr, %outputs: !llvm.ptr) -> i32 attributes { sym_visibility = "public" } {
    // For this minimal test, just return success (0)
    %c0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %c0 : i32
  }
}
