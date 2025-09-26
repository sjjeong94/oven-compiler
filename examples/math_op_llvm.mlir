module {
  llvm.func @__nv_log2f(f32) -> f32 attributes {nvvm.kernel = true}
  llvm.func @__nv_logf(f32) -> f32 attributes {nvvm.kernel = true}
  llvm.func @__nv_tanf(f32) -> f32 attributes {nvvm.kernel = true}
  llvm.func @__nv_sinf(f32) -> f32 attributes {nvvm.kernel = true}
  llvm.func @__nv_cosf(f32) -> f32 attributes {nvvm.kernel = true}
  llvm.func @cos(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = nvvm.read.ptx.sreg.ntid.x : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = llvm.mul %1, %0 : i32
    %4 = llvm.add %3, %2 : i32
    %5 = llvm.getelementptr %arg0[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %6 = llvm.load %5 : !llvm.ptr -> f32
    %7 = llvm.call @__nv_cosf(%6) : (f32) -> f32
    %8 = llvm.getelementptr %arg1[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %7, %8 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @sin(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = nvvm.read.ptx.sreg.ntid.x : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = llvm.mul %1, %0 : i32
    %4 = llvm.add %3, %2 : i32
    %5 = llvm.getelementptr %arg0[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %6 = llvm.load %5 : !llvm.ptr -> f32
    %7 = llvm.call @__nv_sinf(%6) : (f32) -> f32
    %8 = llvm.getelementptr %arg1[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %7, %8 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @tan(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = nvvm.read.ptx.sreg.ntid.x : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = llvm.mul %1, %0 : i32
    %4 = llvm.add %3, %2 : i32
    %5 = llvm.getelementptr %arg0[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %6 = llvm.load %5 : !llvm.ptr -> f32
    %7 = llvm.call @__nv_tanf(%6) : (f32) -> f32
    %8 = llvm.getelementptr %arg1[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %7, %8 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @log(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = nvvm.read.ptx.sreg.ntid.x : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = llvm.mul %1, %0 : i32
    %4 = llvm.add %3, %2 : i32
    %5 = llvm.getelementptr %arg0[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %6 = llvm.load %5 : !llvm.ptr -> f32
    %7 = llvm.call @__nv_logf(%6) : (f32) -> f32
    %8 = llvm.getelementptr %arg1[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %7, %8 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @log2(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {nvvm.kernel = true} {
    %0 = nvvm.read.ptx.sreg.ntid.x : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = llvm.mul %1, %0 : i32
    %4 = llvm.add %3, %2 : i32
    %5 = llvm.getelementptr %arg0[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %6 = llvm.load %5 : !llvm.ptr -> f32
    %7 = llvm.call @__nv_log2f(%6) : (f32) -> f32
    %8 = llvm.getelementptr %arg1[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %7, %8 : f32, !llvm.ptr
    llvm.return
  }
}

