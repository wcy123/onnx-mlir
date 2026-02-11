module attributes {hipdnn.input_count = 1 : i64, hipdnn.input_ranks = array<i64: 4>, hipdnn.output_count = 1 : i64, hipdnn.output_ranks = array<i64: 4>} {
  llvm.func @hip_release_constant(!llvm.ptr, i64)
  llvm.func @hip_upload_constant(!llvm.ptr, i64, !llvm.ptr, i64)
  llvm.func @miopenConvolutionForward(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
  llvm.func @hipMalloc(i64) -> !llvm.ptr
  llvm.func @hip_get_constant(!llvm.ptr, i64) -> !llvm.ptr
  llvm.mlir.global internal constant @constant_2(dense<2.000000e+00> : tensor<64x64x3x3xf32>) {addr_space = 0 : i32} : !llvm.array<36864 x f32>
  llvm.mlir.global internal constant @constant_1(dense<5.000000e-01> : tensor<64xf32>) {addr_space = 0 : i32} : !llvm.array<64 x f32>
  llvm.mlir.global internal constant @constant_3(dense<1.000000e-01> : tensor<64xf32>) {addr_space = 0 : i32} : !llvm.array<64 x f32>
  llvm.mlir.global internal constant @constant_0(dense<1.000000e+00> : tensor<64x3x3x3xf32>) {addr_space = 0 : i32} : !llvm.array<1728 x f32>
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr<1>, %arg13: !llvm.ptr<1>, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64) -> i32 {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)>
    %1 = llvm.insertvalue %arg12, %0[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %2 = llvm.insertvalue %arg13, %1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %3 = llvm.insertvalue %arg14, %2[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %4 = llvm.insertvalue %arg15, %3[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %5 = llvm.insertvalue %arg19, %4[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %6 = llvm.insertvalue %arg16, %5[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %7 = llvm.insertvalue %arg20, %6[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %8 = llvm.insertvalue %arg17, %7[3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %9 = llvm.insertvalue %arg21, %8[4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %10 = llvm.insertvalue %arg18, %9[3, 3] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %11 = llvm.insertvalue %arg22, %10[4, 3] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %12 = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)>
    %13 = llvm.insertvalue %arg1, %12[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %14 = llvm.insertvalue %arg2, %13[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %15 = llvm.insertvalue %arg3, %14[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %16 = llvm.insertvalue %arg4, %15[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %17 = llvm.insertvalue %arg8, %16[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %18 = llvm.insertvalue %arg5, %17[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %19 = llvm.insertvalue %arg9, %18[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %20 = llvm.insertvalue %arg6, %19[3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %21 = llvm.insertvalue %arg10, %20[4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %22 = llvm.insertvalue %arg7, %21[3, 3] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %23 = llvm.insertvalue %arg11, %22[4, 3] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %24 = llvm.mlir.constant(0 : i64) : i64
    %25 = llvm.call @hip_get_constant(%arg0, %24) : (!llvm.ptr, i64) -> !llvm.ptr
    %26 = llvm.addrspacecast %25 : !llvm.ptr to !llvm.ptr<1>
    %27 = llvm.mlir.constant(64 : i64) : i64
    %28 = llvm.mlir.constant(3 : i64) : i64
    %29 = llvm.mlir.constant(3 : i64) : i64
    %30 = llvm.mlir.constant(3 : i64) : i64
    %31 = llvm.mlir.constant(1 : i64) : i64
    %32 = llvm.mlir.constant(3 : i64) : i64
    %33 = llvm.mlir.constant(9 : i64) : i64
    %34 = llvm.mlir.constant(27 : i64) : i64
    %35 = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)>
    %36 = llvm.insertvalue %26, %35[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %37 = llvm.insertvalue %26, %36[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %38 = llvm.mlir.constant(0 : index) : i64
    %39 = llvm.insertvalue %38, %37[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %40 = llvm.insertvalue %27, %39[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %41 = llvm.insertvalue %28, %40[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %42 = llvm.insertvalue %29, %41[3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %43 = llvm.insertvalue %30, %42[3, 3] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %44 = llvm.insertvalue %34, %43[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %45 = llvm.insertvalue %33, %44[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %46 = llvm.insertvalue %32, %45[4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %47 = llvm.insertvalue %31, %46[4, 3] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %48 = llvm.mlir.constant(1 : i64) : i64
    %49 = llvm.call @hip_get_constant(%arg0, %48) : (!llvm.ptr, i64) -> !llvm.ptr
    %50 = llvm.addrspacecast %49 : !llvm.ptr to !llvm.ptr<1>
    %51 = llvm.mlir.constant(64 : i64) : i64
    %52 = llvm.mlir.constant(1 : i64) : i64
    %53 = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %54 = llvm.insertvalue %50, %53[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %55 = llvm.insertvalue %50, %54[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %56 = llvm.mlir.constant(0 : index) : i64
    %57 = llvm.insertvalue %56, %55[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %58 = llvm.insertvalue %51, %57[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %59 = llvm.insertvalue %52, %58[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %60 = llvm.mlir.constant(1 : index) : i64
    %61 = llvm.mlir.constant(64 : index) : i64
    %62 = llvm.mlir.constant(224 : index) : i64
    %63 = llvm.mlir.constant(224 : index) : i64
    %64 = llvm.mlir.constant(1 : index) : i64
    %65 = llvm.mlir.constant(50176 : index) : i64
    %66 = llvm.mlir.constant(3211264 : index) : i64
    %67 = llvm.mlir.constant(3211264 : index) : i64
    %68 = llvm.mlir.zero : !llvm.ptr
    %69 = llvm.getelementptr %68[%67] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %70 = llvm.ptrtoint %69 : !llvm.ptr to i64
    %71 = llvm.call @hipMalloc(%70) : (i64) -> !llvm.ptr
    %72 = llvm.addrspacecast %71 : !llvm.ptr to !llvm.ptr<1>
    %73 = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)>
    %74 = llvm.insertvalue %72, %73[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %75 = llvm.insertvalue %72, %74[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %76 = llvm.mlir.constant(0 : index) : i64
    %77 = llvm.insertvalue %76, %75[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %78 = llvm.insertvalue %60, %77[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %79 = llvm.insertvalue %61, %78[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %80 = llvm.insertvalue %62, %79[3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %81 = llvm.insertvalue %63, %80[3, 3] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %82 = llvm.insertvalue %66, %81[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %83 = llvm.insertvalue %65, %82[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %84 = llvm.insertvalue %63, %83[4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %85 = llvm.insertvalue %64, %84[4, 3] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %86 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %87 = llvm.addrspacecast %86 : !llvm.ptr<1> to !llvm.ptr
    %88 = llvm.extractvalue %47[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %89 = llvm.addrspacecast %88 : !llvm.ptr<1> to !llvm.ptr
    %90 = llvm.extractvalue %85[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %91 = llvm.addrspacecast %90 : !llvm.ptr<1> to !llvm.ptr
    %92 = llvm.extractvalue %59[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %93 = llvm.addrspacecast %92 : !llvm.ptr<1> to !llvm.ptr
    %94 = llvm.mlir.constant(3 : i64) : i64
    %95 = llvm.mlir.constant(3 : i64) : i64
    %96 = llvm.mlir.constant(1 : i64) : i64
    %97 = llvm.mlir.constant(1 : i64) : i64
    %98 = llvm.mlir.constant(1 : i64) : i64
    %99 = llvm.mlir.constant(1 : i64) : i64
    %100 = llvm.mlir.constant(1 : i64) : i64
    %101 = llvm.mlir.constant(1 : i64) : i64
    %102 = llvm.mlir.constant(1 : i64) : i64
    %103 = llvm.mlir.constant(1 : i64) : i64
    %104 = llvm.mlir.constant(1 : i64) : i64
    %105 = llvm.call @miopenConvolutionForward(%arg0, %87, %89, %93, %91, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %106 = llvm.mlir.constant(2 : i64) : i64
    %107 = llvm.call @hip_get_constant(%arg0, %106) : (!llvm.ptr, i64) -> !llvm.ptr
    %108 = llvm.addrspacecast %107 : !llvm.ptr to !llvm.ptr<1>
    %109 = llvm.mlir.constant(64 : i64) : i64
    %110 = llvm.mlir.constant(64 : i64) : i64
    %111 = llvm.mlir.constant(3 : i64) : i64
    %112 = llvm.mlir.constant(3 : i64) : i64
    %113 = llvm.mlir.constant(1 : i64) : i64
    %114 = llvm.mlir.constant(3 : i64) : i64
    %115 = llvm.mlir.constant(9 : i64) : i64
    %116 = llvm.mlir.constant(576 : i64) : i64
    %117 = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)>
    %118 = llvm.insertvalue %108, %117[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %119 = llvm.insertvalue %108, %118[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %120 = llvm.mlir.constant(0 : index) : i64
    %121 = llvm.insertvalue %120, %119[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %122 = llvm.insertvalue %109, %121[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %123 = llvm.insertvalue %110, %122[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %124 = llvm.insertvalue %111, %123[3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %125 = llvm.insertvalue %112, %124[3, 3] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %126 = llvm.insertvalue %116, %125[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %127 = llvm.insertvalue %115, %126[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %128 = llvm.insertvalue %114, %127[4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %129 = llvm.insertvalue %113, %128[4, 3] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %130 = llvm.mlir.constant(3 : i64) : i64
    %131 = llvm.call @hip_get_constant(%arg0, %130) : (!llvm.ptr, i64) -> !llvm.ptr
    %132 = llvm.addrspacecast %131 : !llvm.ptr to !llvm.ptr<1>
    %133 = llvm.mlir.constant(64 : i64) : i64
    %134 = llvm.mlir.constant(1 : i64) : i64
    %135 = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %136 = llvm.insertvalue %132, %135[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %137 = llvm.insertvalue %132, %136[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %138 = llvm.mlir.constant(0 : index) : i64
    %139 = llvm.insertvalue %138, %137[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %140 = llvm.insertvalue %133, %139[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %141 = llvm.insertvalue %134, %140[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %142 = llvm.mlir.constant(1 : index) : i64
    %143 = llvm.mlir.constant(64 : index) : i64
    %144 = llvm.mlir.constant(112 : index) : i64
    %145 = llvm.mlir.constant(112 : index) : i64
    %146 = llvm.mlir.constant(1 : index) : i64
    %147 = llvm.mlir.constant(12544 : index) : i64
    %148 = llvm.mlir.constant(802816 : index) : i64
    %149 = llvm.mlir.constant(802816 : index) : i64
    %150 = llvm.mlir.zero : !llvm.ptr
    %151 = llvm.getelementptr %150[%149] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %152 = llvm.ptrtoint %151 : !llvm.ptr to i64
    %153 = llvm.call @hipMalloc(%152) : (i64) -> !llvm.ptr
    %154 = llvm.addrspacecast %153 : !llvm.ptr to !llvm.ptr<1>
    %155 = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)>
    %156 = llvm.insertvalue %154, %155[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %157 = llvm.insertvalue %154, %156[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %158 = llvm.mlir.constant(0 : index) : i64
    %159 = llvm.insertvalue %158, %157[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %160 = llvm.insertvalue %142, %159[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %161 = llvm.insertvalue %143, %160[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %162 = llvm.insertvalue %144, %161[3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %163 = llvm.insertvalue %145, %162[3, 3] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %164 = llvm.insertvalue %148, %163[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %165 = llvm.insertvalue %147, %164[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %166 = llvm.insertvalue %145, %165[4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %167 = llvm.insertvalue %146, %166[4, 3] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %168 = llvm.extractvalue %85[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %169 = llvm.addrspacecast %168 : !llvm.ptr<1> to !llvm.ptr
    %170 = llvm.extractvalue %129[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %171 = llvm.addrspacecast %170 : !llvm.ptr<1> to !llvm.ptr
    %172 = llvm.extractvalue %167[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %173 = llvm.addrspacecast %172 : !llvm.ptr<1> to !llvm.ptr
    %174 = llvm.extractvalue %141[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %175 = llvm.addrspacecast %174 : !llvm.ptr<1> to !llvm.ptr
    %176 = llvm.mlir.constant(3 : i64) : i64
    %177 = llvm.mlir.constant(3 : i64) : i64
    %178 = llvm.mlir.constant(2 : i64) : i64
    %179 = llvm.mlir.constant(2 : i64) : i64
    %180 = llvm.mlir.constant(1 : i64) : i64
    %181 = llvm.mlir.constant(1 : i64) : i64
    %182 = llvm.mlir.constant(1 : i64) : i64
    %183 = llvm.mlir.constant(1 : i64) : i64
    %184 = llvm.mlir.constant(1 : i64) : i64
    %185 = llvm.mlir.constant(1 : i64) : i64
    %186 = llvm.mlir.constant(1 : i64) : i64
    %187 = llvm.call @miopenConvolutionForward(%arg0, %169, %171, %175, %173, %176, %177, %178, %179, %180, %181, %182, %183, %184, %185, %186) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %188 = llvm.mlir.constant(1 : index) : i64
    %189 = llvm.extractvalue %167[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %190 = llvm.mul %188, %189 : i64
    %191 = llvm.extractvalue %167[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %192 = llvm.mul %190, %191 : i64
    %193 = llvm.extractvalue %167[3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %194 = llvm.mul %192, %193 : i64
    %195 = llvm.extractvalue %167[3, 3] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %196 = llvm.mul %194, %195 : i64
    %197 = llvm.mlir.zero : !llvm.ptr
    %198 = llvm.getelementptr %197[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %199 = llvm.ptrtoint %198 : !llvm.ptr to i64
    %200 = llvm.mul %196, %199 : i64
    %201 = llvm.extractvalue %167[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %202 = llvm.extractvalue %167[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %203 = llvm.getelementptr %201[%202] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %204 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %205 = llvm.extractvalue %11[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)> 
    %206 = llvm.getelementptr %204[%205] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    "llvm.intr.memcpy"(%206, %203, %200) <{isVolatile = false}> : (!llvm.ptr<1>, !llvm.ptr<1>, i64) -> ()
    %207 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %207 : i32
  }
  llvm.func @get_constant_count() -> i64 {
    %0 = llvm.mlir.constant(4 : i64) : i64
    llvm.return %0 : i64
  }
  llvm.func @initialize_constants(%arg0: !llvm.ptr) -> i32 {
    %0 = llvm.mlir.addressof @constant_2 : !llvm.ptr
    %1 = llvm.mlir.constant(2 : i64) : i64
    %2 = llvm.mlir.constant(147456 : i64) : i64
    llvm.call @hip_upload_constant(%arg0, %1, %0, %2) : (!llvm.ptr, i64, !llvm.ptr, i64) -> ()
    %3 = llvm.mlir.addressof @constant_1 : !llvm.ptr
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.mlir.constant(256 : i64) : i64
    llvm.call @hip_upload_constant(%arg0, %4, %3, %5) : (!llvm.ptr, i64, !llvm.ptr, i64) -> ()
    %6 = llvm.mlir.addressof @constant_3 : !llvm.ptr
    %7 = llvm.mlir.constant(3 : i64) : i64
    %8 = llvm.mlir.constant(256 : i64) : i64
    llvm.call @hip_upload_constant(%arg0, %7, %6, %8) : (!llvm.ptr, i64, !llvm.ptr, i64) -> ()
    %9 = llvm.mlir.addressof @constant_0 : !llvm.ptr
    %10 = llvm.mlir.constant(0 : i64) : i64
    %11 = llvm.mlir.constant(6912 : i64) : i64
    llvm.call @hip_upload_constant(%arg0, %10, %9, %11) : (!llvm.ptr, i64, !llvm.ptr, i64) -> ()
    %12 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %12 : i32
  }
  llvm.func @release_constants(%arg0: !llvm.ptr) -> i32 {
    %0 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @hip_release_constant(%arg0, %0) : (!llvm.ptr, i64) -> ()
    %1 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @hip_release_constant(%arg0, %1) : (!llvm.ptr, i64) -> ()
    %2 = llvm.mlir.constant(3 : i64) : i64
    llvm.call @hip_release_constant(%arg0, %2) : (!llvm.ptr, i64) -> ()
    %3 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @hip_release_constant(%arg0, %3) : (!llvm.ptr, i64) -> ()
    %4 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %4 : i32
  }
}

