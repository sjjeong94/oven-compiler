func.func @add(%0: !llvm.ptr, %1: !llvm.ptr, %2: !llvm.ptr) {
  %3 = nvvm.read.ptx.sreg.ntid.x : i32
  %4 = nvvm.read.ptx.sreg.ctaid.x : i32
  %5 = nvvm.read.ptx.sreg.tid.x : i32
  %6 = arith.muli %4, %3 : i32
  %7 = arith.addi %6, %5 : i32
  %8 = oven.load %0, %7 : (!llvm.ptr, i32) -> f32
  %9 = oven.load %1, %7 : (!llvm.ptr, i32) -> f32
  %10 = arith.addf %8, %9 : f32
  oven.store %10, %2, %7 : (f32, !llvm.ptr, i32)
  return
}
func.func @mul(%11: !llvm.ptr, %12: !llvm.ptr, %13: !llvm.ptr) {
  %14 = nvvm.read.ptx.sreg.ntid.x : i32
  %15 = nvvm.read.ptx.sreg.ctaid.x : i32
  %16 = nvvm.read.ptx.sreg.tid.x : i32
  %17 = arith.muli %15, %14 : i32
  %18 = arith.addi %17, %16 : i32
  %19 = oven.load %11, %18 : (!llvm.ptr, i32) -> f32
  %20 = oven.load %12, %18 : (!llvm.ptr, i32) -> f32
  %21 = arith.mulf %19, %20 : f32
  oven.store %21, %13, %18 : (f32, !llvm.ptr, i32)
  return
}
func.func @sub(%22: !llvm.ptr, %23: !llvm.ptr, %24: !llvm.ptr) {
  %25 = nvvm.read.ptx.sreg.ntid.x : i32
  %26 = nvvm.read.ptx.sreg.ctaid.x : i32
  %27 = nvvm.read.ptx.sreg.tid.x : i32
  %28 = arith.muli %26, %25 : i32
  %29 = arith.addi %28, %27 : i32
  %30 = oven.load %22, %29 : (!llvm.ptr, i32) -> f32
  %31 = oven.load %23, %29 : (!llvm.ptr, i32) -> f32
  %32 = arith.subf %30, %31 : f32
  oven.store %32, %24, %29 : (f32, !llvm.ptr, i32)
  return
}
func.func @div(%33: !llvm.ptr, %34: !llvm.ptr, %35: !llvm.ptr) {
  %36 = nvvm.read.ptx.sreg.ntid.x : i32
  %37 = nvvm.read.ptx.sreg.ctaid.x : i32
  %38 = nvvm.read.ptx.sreg.tid.x : i32
  %39 = arith.muli %37, %36 : i32
  %40 = arith.addi %39, %38 : i32
  %41 = oven.load %33, %40 : (!llvm.ptr, i32) -> f32
  %42 = oven.load %34, %40 : (!llvm.ptr, i32) -> f32
  %43 = arith.divf %41, %42 : f32
  oven.store %43, %35, %40 : (f32, !llvm.ptr, i32)
  return
}
func.func @vadd(%44: !llvm.ptr, %45: !llvm.ptr, %46: !llvm.ptr) {
  %47 = nvvm.read.ptx.sreg.ntid.x : i32
  %48 = nvvm.read.ptx.sreg.ctaid.x : i32
  %49 = nvvm.read.ptx.sreg.tid.x : i32
  %50 = arith.muli %48, %47 : i32
  %51 = arith.addi %50, %49 : i32
  %52 = arith.constant 4 : i32
  %53 = arith.muli %51, %52 : i32
  %54 = oven.vload %44, %53 : (!llvm.ptr, i32) -> vector<4xf32>
  %55 = oven.vload %45, %53 : (!llvm.ptr, i32) -> vector<4xf32>
  %56 = arith.addf %54, %55 : vector<4xf32>
  oven.vstore %56, %46, %53 : (vector<4xf32>, !llvm.ptr, i32)
  return
}
func.func @vmul(%57: !llvm.ptr, %58: !llvm.ptr, %59: !llvm.ptr) {
  %60 = nvvm.read.ptx.sreg.ntid.x : i32
  %61 = nvvm.read.ptx.sreg.ctaid.x : i32
  %62 = nvvm.read.ptx.sreg.tid.x : i32
  %63 = arith.muli %61, %60 : i32
  %64 = arith.addi %63, %62 : i32
  %65 = arith.constant 4 : i32
  %66 = arith.muli %64, %65 : i32
  %67 = oven.vload %57, %66 : (!llvm.ptr, i32) -> vector<4xf32>
  %68 = oven.vload %58, %66 : (!llvm.ptr, i32) -> vector<4xf32>
  %69 = arith.mulf %67, %68 : vector<4xf32>
  oven.vstore %69, %59, %66 : (vector<4xf32>, !llvm.ptr, i32)
  return
}
func.func @vsub(%70: !llvm.ptr, %71: !llvm.ptr, %72: !llvm.ptr) {
  %73 = nvvm.read.ptx.sreg.ntid.x : i32
  %74 = nvvm.read.ptx.sreg.ctaid.x : i32
  %75 = nvvm.read.ptx.sreg.tid.x : i32
  %76 = arith.muli %74, %73 : i32
  %77 = arith.addi %76, %75 : i32
  %78 = arith.constant 4 : i32
  %79 = arith.muli %77, %78 : i32
  %80 = oven.vload %70, %79 : (!llvm.ptr, i32) -> vector<4xf32>
  %81 = oven.vload %71, %79 : (!llvm.ptr, i32) -> vector<4xf32>
  %82 = arith.subf %80, %81 : vector<4xf32>
  oven.vstore %82, %72, %79 : (vector<4xf32>, !llvm.ptr, i32)
  return
}
func.func @vdiv(%83: !llvm.ptr, %84: !llvm.ptr, %85: !llvm.ptr) {
  %86 = nvvm.read.ptx.sreg.ntid.x : i32
  %87 = nvvm.read.ptx.sreg.ctaid.x : i32
  %88 = nvvm.read.ptx.sreg.tid.x : i32
  %89 = arith.muli %87, %86 : i32
  %90 = arith.addi %89, %88 : i32
  %91 = arith.constant 4 : i32
  %92 = arith.muli %90, %91 : i32
  %93 = oven.vload %83, %92 : (!llvm.ptr, i32) -> vector<4xf32>
  %94 = oven.vload %84, %92 : (!llvm.ptr, i32) -> vector<4xf32>
  %95 = arith.divf %93, %94 : vector<4xf32>
  oven.vstore %95, %85, %92 : (vector<4xf32>, !llvm.ptr, i32)
  return
}