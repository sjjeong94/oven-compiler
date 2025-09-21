module {
  llvm.func @sigmoid(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %1 = llvm.mlir.constant(1.44269502 : f32) : f32
    %2 = nvvm.read.ptx.sreg.ntid.x : i32
    %3 = nvvm.read.ptx.sreg.ctaid.x : i32
    %4 = nvvm.read.ptx.sreg.tid.x : i32
    %5 = llvm.mul %3, %2 : i32
    %6 = llvm.add %5, %4 : i32
    %7 = llvm.getelementptr %arg0[%6] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %8 = llvm.load %7 : !llvm.ptr -> f32
    %9 = llvm.fneg %8 : f32
    %10 = llvm.fmul %9, %1 : f32
    %11 = llvm.intr.exp2(%10) : (f32) -> f32
    %12 = llvm.fadd %11, %0 : f32
    %13 = llvm.fdiv %0, %12 : f32
    %14 = llvm.getelementptr %arg1[%6] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %13, %14 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @exp(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = llvm.mlir.constant(1.44269502 : f32) : f32
    %1 = nvvm.read.ptx.sreg.ntid.x : i32
    %2 = nvvm.read.ptx.sreg.ctaid.x : i32
    %3 = nvvm.read.ptx.sreg.tid.x : i32
    %4 = llvm.mul %2, %1 : i32
    %5 = llvm.add %4, %3 : i32
    %6 = llvm.getelementptr %arg0[%5] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %7 = llvm.load %6 : !llvm.ptr -> f32
    %8 = llvm.fmul %7, %0 : f32
    %9 = llvm.intr.exp2(%8) : (f32) -> f32
    %10 = llvm.getelementptr %arg1[%5] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %9, %10 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @sqrt(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = nvvm.read.ptx.sreg.ntid.x : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = llvm.mul %1, %0 : i32
    %4 = llvm.add %3, %2 : i32
    %5 = llvm.getelementptr %arg0[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %6 = llvm.load %5 : !llvm.ptr -> f32
    %7 = llvm.intr.sqrt(%6) : (f32) -> f32
    %8 = llvm.getelementptr %arg1[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %7, %8 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @abs(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = nvvm.read.ptx.sreg.ntid.x : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = llvm.mul %1, %0 : i32
    %4 = llvm.add %3, %2 : i32
    %5 = llvm.getelementptr %arg0[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %6 = llvm.load %5 : !llvm.ptr -> f32
    %7 = llvm.intr.fabs(%6) : (f32) -> f32
    %8 = llvm.getelementptr %arg1[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %7, %8 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @ceil(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = nvvm.read.ptx.sreg.ntid.x : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = llvm.mul %1, %0 : i32
    %4 = llvm.add %3, %2 : i32
    %5 = llvm.getelementptr %arg0[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %6 = llvm.load %5 : !llvm.ptr -> f32
    %7 = llvm.intr.ceil(%6) : (f32) -> f32
    %8 = llvm.getelementptr %arg1[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %7, %8 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @floor(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = nvvm.read.ptx.sreg.ntid.x : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = llvm.mul %1, %0 : i32
    %4 = llvm.add %3, %2 : i32
    %5 = llvm.getelementptr %arg0[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %6 = llvm.load %5 : !llvm.ptr -> f32
    %7 = llvm.intr.floor(%6) : (f32) -> f32
    %8 = llvm.getelementptr %arg1[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %7, %8 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @rsqrt(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %1 = nvvm.read.ptx.sreg.ntid.x : i32
    %2 = nvvm.read.ptx.sreg.ctaid.x : i32
    %3 = nvvm.read.ptx.sreg.tid.x : i32
    %4 = llvm.mul %2, %1 : i32
    %5 = llvm.add %4, %3 : i32
    %6 = llvm.getelementptr %arg0[%5] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %7 = llvm.load %6 : !llvm.ptr -> f32
    %8 = llvm.intr.sqrt(%7) : (f32) -> f32
    %9 = llvm.fdiv %0, %8 : f32
    %10 = llvm.getelementptr %arg1[%5] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %9, %10 : f32, !llvm.ptr
    llvm.return
  }
}

