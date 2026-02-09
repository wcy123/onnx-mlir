#ifndef HIP_OPS
#define HIP_OPS

include "HipDialect.td"
include "HipTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class Hip_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hip_Dialect, mnemonic, traits>;

def Hip_CreateHandleOp : Hip_Op<"create_handle", []> {
  let summary = "Create a HIP runtime handle";
  let description = [{
    Creates and returns a new HIP runtime handle.
    
    Example:
    ```mlir
    %handle = hip.create_handle() : !hip.handle
    ```
  }];
  
  let results = (outs Hip_HandleType:$handle);
  let assemblyFormat = "`(` `)` attr-dict `:` type($handle)";
}

def Hip_DestroyHandleOp : Hip_Op<"destroy_handle", []> {
  let summary = "Destroy a HIP runtime handle";
  let description = [{
    Destroys a HIP runtime handle.
    
    Example:
    ```mlir
    hip.destroy_handle(%handle) : !hip.handle
    ```
  }];
  
  let arguments = (ins Hip_HandleType:$handle);
  let assemblyFormat = "`(` $handle `)` attr-dict `:` type($handle)";
}

def Hip_AllocOp : Hip_Op<"alloc", []> {
  let summary = "Allocate device memory";
  let description = [{
    Allocates device memory using HIP and returns a memref.
    
    Example:
    ```mlir
    %x = hip.alloc(%handle, %N) : memref<?x128xf32, 1>
    ```
  }];
  
  let arguments = (ins 
    Hip_HandleType:$handle,
    Variadic<Index>:$dynamicSizes
  );
  let results = (outs AnyMemRef:$memref);
  
  let assemblyFormat = [{
    `(` $handle (`,` $dynamicSizes^ )? `)` attr-dict `:` type($memref)
  }];
}

def Hip_FreeOp : Hip_Op<"free", []> {
  let summary = "Free device memory";
  let description = [{
    Frees device memory allocated with hip.alloc.

    Example:
    ```mlir
    hip.free(%handle, %x) : memref<?x128xf32, 1>
    ```
  }];

  let arguments = (ins
    Hip_HandleType:$handle,
    AnyMemRef:$memref
  );

  let assemblyFormat = [{
    `(` $handle `,` $memref `)` attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// MIOpen DNN Operations
//===----------------------------------------------------------------------===//

def Hip_ConvOp : Hip_Op<"conv", []> {
  let summary = "2D convolution operation using MIOpen";
  let description = [{
    Performs 2D convolution using MIOpen (miopenConvolutionForward).

    Attributes:
    - kernel_shape: Array of kernel dimensions [H, W]
    - strides: Array of stride values [H, W]
    - pads: Array of padding values [top, left, bottom, right]
    - dilations: Array of dilation values [H, W]
    - group: Number of groups for grouped convolution

    Example:
    ```mlir
    %output = hip.conv(%handle, %input, %weights, %bias)
              {kernel_shape = [3, 3], strides = [1, 1],
               pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
              : (memref<?x?x?x?xf32, 1>, memref<?x?x?x?xf32, 1>, memref<?xf32, 1>)
              -> memref<?x?x?x?xf32, 1>
    ```
  }];

  let arguments = (ins
    Hip_HandleType:$handle,
    AnyMemRef:$input,
    AnyMemRef:$weights,
    Optional<AnyMemRef>:$bias,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    I64ArrayAttr:$dilations,
    I64Attr:$group
  );
  let results = (outs AnyMemRef:$output);

  let assemblyFormat = [{
    `(` $handle `,` $input `,` $weights (`,` $bias^)? `)` attr-dict
    `:` `(` type($input) `,` type($weights) (`,` type($bias)^)? `)` `->` type($output)
  }];
}

def Hip_GemmOp : Hip_Op<"gemm", []> {
  let summary = "General matrix multiplication using hipBLASLt";
  let description = [{
    Performs matrix multiplication: C = alpha * A * B + beta * C
    using hipBLASLt (miopenGEMM or hipblasLtMatmul).

    Attributes:
    - transA: Whether to transpose matrix A (0 or 1)
    - transB: Whether to transpose matrix B (0 or 1)
    - alpha: Scalar multiplier for A*B (default 1.0)
    - beta: Scalar multiplier for C (default 1.0)

    Example:
    ```mlir
    %C = hip.gemm(%handle, %A, %B)
         {transA = 0, transB = 0, alpha = 1.0, beta = 0.0}
         : (memref<?x?xf32, 1>, memref<?x?xf32, 1>)
         -> memref<?x?xf32, 1>
    ```
  }];

  let arguments = (ins
    Hip_HandleType:$handle,
    AnyMemRef:$A,
    AnyMemRef:$B,
    Optional<AnyMemRef>:$C,
    I64Attr:$transA,
    I64Attr:$transB,
    F32Attr:$alpha,
    F32Attr:$beta
  );
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    `(` $handle `,` $A `,` $B (`,` $C^)? `)` attr-dict
    `:` `(` type($A) `,` type($B) (`,` type($C)^)? `)` `->` type($result)
  }];
}

def Hip_MaxPoolOp : Hip_Op<"maxpool", []> {
  let summary = "Max pooling operation using MIOpen";
  let description = [{
    Performs max pooling using MIOpen (miopenPoolingForward with max mode).

    Attributes:
    - kernel_shape: Array of pooling window dimensions [H, W]
    - strides: Array of stride values [H, W]
    - pads: Array of padding values [top, left, bottom, right]

    Example:
    ```mlir
    %output = hip.maxpool(%handle, %input)
              {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]}
              : memref<?x?x?x?xf32, 1> -> memref<?x?x?x?xf32, 1>
    ```
  }];

  let arguments = (ins
    Hip_HandleType:$handle,
    AnyMemRef:$input,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads
  );
  let results = (outs AnyMemRef:$output);

  let assemblyFormat = [{
    `(` $handle `,` $input `)` attr-dict
    `:` type($input) `->` type($output)
  }];
}

def Hip_AvgPoolOp : Hip_Op<"avgpool", []> {
  let summary = "Average pooling operation using MIOpen";
  let description = [{
    Performs average pooling using MIOpen (miopenPoolingForward with average mode).

    Attributes:
    - kernel_shape: Array of pooling window dimensions [H, W]
    - strides: Array of stride values [H, W]
    - pads: Array of padding values [top, left, bottom, right]

    Example:
    ```mlir
    %output = hip.avgpool(%handle, %input)
              {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]}
              : memref<?x?x?x?xf32, 1> -> memref<?x?x?x?xf32, 1>
    ```
  }];

  let arguments = (ins
    Hip_HandleType:$handle,
    AnyMemRef:$input,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads
  );
  let results = (outs AnyMemRef:$output);

  let assemblyFormat = [{
    `(` $handle `,` $input `)` attr-dict
    `:` type($input) `->` type($output)
  }];
}

#endif // HIP_OPS
