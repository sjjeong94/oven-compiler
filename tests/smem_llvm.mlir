
module {
  llvm.mlir.global external @global_smem() { addr_space = 3 : i32, alignment = 16 : i64 } : !llvm.array<0 x i8>
  llvm.func @function(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {nvvm.kernel = true} {
    %shared_buf_ptr = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %0 = llvm.mlir.constant(128 : i32) : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = llvm.mul %0, %1 : i32
    %4 = llvm.add %2, %3 : i32
    %5 = llvm.getelementptr %arg0[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %6 = llvm.getelementptr %shared_buf_ptr[%2] : (!llvm.ptr<3>, i32) -> !llvm.ptr, f32
    %7 = llvm.getelementptr %arg1[%4] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %8 = llvm.load %5 : !llvm.ptr -> f32
    llvm.store %8, %6 : f32, !llvm.ptr
    nvvm.barrier0
    %9 = llvm.load %6 : !llvm.ptr -> f32
    llvm.store %9, %7 : f32, !llvm.ptr
    llvm.return
  }
}