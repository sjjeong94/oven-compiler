func.func @matmul(%0: !llvm.ptr, %1: !llvm.ptr, %2: !llvm.ptr, %3: i32, %4: i32, %5: i32) {
  %6 = nvvm.read.ptx.sreg.ntid.x : i32
  %7 = nvvm.read.ptx.sreg.ctaid.x : i32
  %8 = nvvm.read.ptx.sreg.ctaid.y : i32
  %9 = nvvm.read.ptx.sreg.tid.x : i32
  %10 = nvvm.read.ptx.sreg.tid.y : i32
  %11 = arith.muli %7, %6 : i32
  %12 = arith.addi %11, %9 : i32
  %13 = arith.muli %8, %6 : i32
  %14 = arith.addi %13, %10 : i32
  %15 = arith.constant 0.0 : f32
  %17 = arith.constant 0 : i32
  %18 = arith.constant 1 : i32
  %19 = arith.index_cast %17 : i32 to index
  %20 = arith.index_cast %5 : i32 to index
  %21 = arith.index_cast %18 : i32 to index
  %23 = scf.for %16 = %19 to %20 step %21 iter_args(%22 = %15) -> (f32) {
    %24 = arith.muli %14, %5 : i32
    %25 = arith.index_cast %16 : index to i32
    %26 = arith.addi %24, %25 : i32
    %27 = arith.index_cast %16 : index to i32
    %28 = arith.muli %27, %4 : i32
    %29 = arith.addi %28, %12 : i32
    %30 = oven.load %0, %26 : (!llvm.ptr, i32) -> f32
    %31 = oven.load %1, %29 : (!llvm.ptr, i32) -> f32
    %32 = arith.mulf %30, %31 : f32
    %33 = arith.addf %22, %32 : f32
    scf.yield %33 : f32
  }
  %34 = arith.muli %14, %4 : i32
  %35 = arith.addi %34, %12 : i32
  oven.store %23, %2, %35 : (f32, !llvm.ptr, i32)
  return
}