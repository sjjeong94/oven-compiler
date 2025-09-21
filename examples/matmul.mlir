// Generated MLIR code from Python source

func.func @matmul(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr, %c_ptr: !llvm.ptr, %m: i32, %n: i32, %k: i32) {
  %0 = nvvm.read.ptx.sreg.ntid.x : i32
  %1 = nvvm.read.ptx.sreg.ctaid.x : i32
  %2 = nvvm.read.ptx.sreg.ctaid.y : i32
  %3 = nvvm.read.ptx.sreg.tid.x : i32
  %4 = nvvm.read.ptx.sreg.tid.y : i32
  %5 = arith.muli %1, %0 : i32
  %6 = arith.addi %5, %3 : i32
  %7 = arith.muli %2, %0 : i32
  %8 = arith.addi %7, %4 : i32
  %9 = arith.constant 0.0 : f32
  %10 = arith.constant 0 : index
  %11 = arith.index_cast %k : i32 to index
  %12 = arith.constant 1 : index
  %13 = scf.for %i_index = %10 to %11 step %12 iter_args(%sum = %9) -> (f32) {
    %14 = arith.index_cast %i_index : index to i32
    %15 = arith.muli %8, %k : i32
    %16 = arith.addi %15, %14 : i32
    %17 = arith.muli %14, %n : i32
    %18 = arith.addi %17, %6 : i32
    %19 = oven.load %a_ptr, %16 : (!llvm.ptr, i32) -> f32
    %20 = oven.load %b_ptr, %18 : (!llvm.ptr, i32) -> f32
    %21 = arith.mulf %19, %20 : f32
    %22 = arith.addf %sum, %21 : f32
    scf.yield %22 : f32
    }
  %23 = arith.muli %8, %n : i32
  %24 = arith.addi %23, %6 : i32
  oven.store %13, %c_ptr, %24 : (f32, !llvm.ptr, i32)
  return
}

