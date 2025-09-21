module {
  llvm.func @matmul(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32, %arg5: i32) attributes {nvvm.kernel = true} {
    %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(0 : index) : i64
    %3 = nvvm.read.ptx.sreg.ntid.x : i32
    %4 = nvvm.read.ptx.sreg.ctaid.x : i32
    %5 = nvvm.read.ptx.sreg.ctaid.y : i32
    %6 = nvvm.read.ptx.sreg.tid.x : i32
    %7 = nvvm.read.ptx.sreg.tid.y : i32
    %8 = llvm.mul %4, %3 : i32
    %9 = llvm.add %8, %6 : i32
    %10 = llvm.mul %5, %3 : i32
    %11 = llvm.add %10, %7 : i32
    %12 = llvm.sext %arg5 : i32 to i64
    llvm.br ^bb1(%2, %0 : i64, f32)
  ^bb1(%13: i64, %14: f32):  // 2 preds: ^bb0, ^bb2
    %15 = llvm.icmp "slt" %13, %12 : i64
    llvm.cond_br %15, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %16 = llvm.trunc %13 : i64 to i32
    %17 = llvm.mul %11, %arg5 : i32
    %18 = llvm.add %17, %16 : i32
    %19 = llvm.mul %16, %arg4 : i32
    %20 = llvm.add %19, %9 : i32
    %21 = llvm.getelementptr %arg0[%18] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %22 = llvm.load %21 : !llvm.ptr -> f32
    %23 = llvm.getelementptr %arg1[%20] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %24 = llvm.load %23 : !llvm.ptr -> f32
    %25 = llvm.fmul %22, %24 : f32
    %26 = llvm.fadd %14, %25 : f32
    %27 = llvm.add %13, %1 : i64
    llvm.br ^bb1(%27, %26 : i64, f32)
  ^bb3:  // pred: ^bb1
    %28 = llvm.mul %11, %arg4 : i32
    %29 = llvm.add %28, %9 : i32
    %30 = llvm.getelementptr %arg2[%29] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %14, %30 : f32, !llvm.ptr
    llvm.return
  }
}
