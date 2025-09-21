module {
  llvm.func @add(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = nvvm.read.ptx.sreg.ntid.x : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = llvm.mul %1, %0 : i32
    %4 = llvm.add %3, %2 : i32
    %5 = llvm.getelementptr %arg0[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %6 = llvm.load %5 : !llvm.ptr -> f32
    %7 = llvm.getelementptr %arg1[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %8 = llvm.load %7 : !llvm.ptr -> f32
    %9 = llvm.fadd %6, %8 : f32
    %10 = llvm.getelementptr %arg2[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %9, %10 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @mul(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = nvvm.read.ptx.sreg.ntid.x : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = llvm.mul %1, %0 : i32
    %4 = llvm.add %3, %2 : i32
    %5 = llvm.getelementptr %arg0[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %6 = llvm.load %5 : !llvm.ptr -> f32
    %7 = llvm.getelementptr %arg1[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %8 = llvm.load %7 : !llvm.ptr -> f32
    %9 = llvm.fmul %6, %8 : f32
    %10 = llvm.getelementptr %arg2[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %9, %10 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @sub(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = nvvm.read.ptx.sreg.ntid.x : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = llvm.mul %1, %0 : i32
    %4 = llvm.add %3, %2 : i32
    %5 = llvm.getelementptr %arg0[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %6 = llvm.load %5 : !llvm.ptr -> f32
    %7 = llvm.getelementptr %arg1[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %8 = llvm.load %7 : !llvm.ptr -> f32
    %9 = llvm.fsub %6, %8 : f32
    %10 = llvm.getelementptr %arg2[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %9, %10 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @div(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = nvvm.read.ptx.sreg.ntid.x : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = llvm.mul %1, %0 : i32
    %4 = llvm.add %3, %2 : i32
    %5 = llvm.getelementptr %arg0[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %6 = llvm.load %5 : !llvm.ptr -> f32
    %7 = llvm.getelementptr %arg1[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %8 = llvm.load %7 : !llvm.ptr -> f32
    %9 = llvm.fdiv %6, %8 : f32
    %10 = llvm.getelementptr %arg2[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %9, %10 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @vadd(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = nvvm.read.ptx.sreg.ntid.x : i32
    %2 = nvvm.read.ptx.sreg.ctaid.x : i32
    %3 = nvvm.read.ptx.sreg.tid.x : i32
    %4 = llvm.mul %2, %1 : i32
    %5 = llvm.add %4, %3 : i32
    %6 = llvm.mul %5, %0 : i32
    %7 = llvm.getelementptr %arg0[%6] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %8 = llvm.load %7 : !llvm.ptr -> vector<4xf32>
    %9 = llvm.getelementptr %arg1[%6] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %10 = llvm.load %9 : !llvm.ptr -> vector<4xf32>
    %11 = llvm.fadd %8, %10 : vector<4xf32>
    %12 = llvm.getelementptr %arg2[%6] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %11, %12 : vector<4xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func @vmul(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = nvvm.read.ptx.sreg.ntid.x : i32
    %2 = nvvm.read.ptx.sreg.ctaid.x : i32
    %3 = nvvm.read.ptx.sreg.tid.x : i32
    %4 = llvm.mul %2, %1 : i32
    %5 = llvm.add %4, %3 : i32
    %6 = llvm.mul %5, %0 : i32
    %7 = llvm.getelementptr %arg0[%6] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %8 = llvm.load %7 : !llvm.ptr -> vector<4xf32>
    %9 = llvm.getelementptr %arg1[%6] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %10 = llvm.load %9 : !llvm.ptr -> vector<4xf32>
    %11 = llvm.fmul %8, %10 : vector<4xf32>
    %12 = llvm.getelementptr %arg2[%6] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %11, %12 : vector<4xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func @vsub(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = nvvm.read.ptx.sreg.ntid.x : i32
    %2 = nvvm.read.ptx.sreg.ctaid.x : i32
    %3 = nvvm.read.ptx.sreg.tid.x : i32
    %4 = llvm.mul %2, %1 : i32
    %5 = llvm.add %4, %3 : i32
    %6 = llvm.mul %5, %0 : i32
    %7 = llvm.getelementptr %arg0[%6] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %8 = llvm.load %7 : !llvm.ptr -> vector<4xf32>
    %9 = llvm.getelementptr %arg1[%6] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %10 = llvm.load %9 : !llvm.ptr -> vector<4xf32>
    %11 = llvm.fsub %8, %10 : vector<4xf32>
    %12 = llvm.getelementptr %arg2[%6] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %11, %12 : vector<4xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func @vdiv(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = nvvm.read.ptx.sreg.ntid.x : i32
    %2 = nvvm.read.ptx.sreg.ctaid.x : i32
    %3 = nvvm.read.ptx.sreg.tid.x : i32
    %4 = llvm.mul %2, %1 : i32
    %5 = llvm.add %4, %3 : i32
    %6 = llvm.mul %5, %0 : i32
    %7 = llvm.getelementptr %arg0[%6] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %8 = llvm.load %7 : !llvm.ptr -> vector<4xf32>
    %9 = llvm.getelementptr %arg1[%6] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %10 = llvm.load %9 : !llvm.ptr -> vector<4xf32>
    %11 = llvm.fdiv %8, %10 : vector<4xf32>
    %12 = llvm.getelementptr %arg2[%6] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %11, %12 : vector<4xf32>, !llvm.ptr
    llvm.return
  }
}

