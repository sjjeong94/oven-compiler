func.func @sigmoid(%0: !llvm.ptr, %1: !llvm.ptr) {
  %2 = nvvm.read.ptx.sreg.ntid.x : i32
  %3 = nvvm.read.ptx.sreg.ctaid.x : i32
  %4 = nvvm.read.ptx.sreg.tid.x : i32
  %5 = arith.muli %3, %2 : i32
  %6 = arith.addi %5, %4 : i32
  %7 = oven.load %0, %6 : (!llvm.ptr, i32) -> f32
  %8 = oven.sigmoid %7 : f32
  oven.store %8, %1, %6 : (f32, !llvm.ptr, i32)
  return
}
func.func @exp(%9: !llvm.ptr, %10: !llvm.ptr) {
  %11 = nvvm.read.ptx.sreg.ntid.x : i32
  %12 = nvvm.read.ptx.sreg.ctaid.x : i32
  %13 = nvvm.read.ptx.sreg.tid.x : i32
  %14 = arith.muli %12, %11 : i32
  %15 = arith.addi %14, %13 : i32
  %16 = oven.load %9, %15 : (!llvm.ptr, i32) -> f32
  %17 = math.exp %16 : f32
  oven.store %17, %10, %15 : (f32, !llvm.ptr, i32)
  return
}
func.func @sqrt(%18: !llvm.ptr, %19: !llvm.ptr) {
  %20 = nvvm.read.ptx.sreg.ntid.x : i32
  %21 = nvvm.read.ptx.sreg.ctaid.x : i32
  %22 = nvvm.read.ptx.sreg.tid.x : i32
  %23 = arith.muli %21, %20 : i32
  %24 = arith.addi %23, %22 : i32
  %25 = oven.load %18, %24 : (!llvm.ptr, i32) -> f32
  %26 = math.sqrt %25 : f32
  oven.store %26, %19, %24 : (f32, !llvm.ptr, i32)
  return
}
func.func @abs(%27: !llvm.ptr, %28: !llvm.ptr) {
  %29 = nvvm.read.ptx.sreg.ntid.x : i32
  %30 = nvvm.read.ptx.sreg.ctaid.x : i32
  %31 = nvvm.read.ptx.sreg.tid.x : i32
  %32 = arith.muli %30, %29 : i32
  %33 = arith.addi %32, %31 : i32
  %34 = oven.load %27, %33 : (!llvm.ptr, i32) -> f32
  %35 = math.absf %34 : f32
  oven.store %35, %28, %33 : (f32, !llvm.ptr, i32)
  return
}
func.func @ceil(%36: !llvm.ptr, %37: !llvm.ptr) {
  %38 = nvvm.read.ptx.sreg.ntid.x : i32
  %39 = nvvm.read.ptx.sreg.ctaid.x : i32
  %40 = nvvm.read.ptx.sreg.tid.x : i32
  %41 = arith.muli %39, %38 : i32
  %42 = arith.addi %41, %40 : i32
  %43 = oven.load %36, %42 : (!llvm.ptr, i32) -> f32
  %44 = math.ceil %43 : f32
  oven.store %44, %37, %42 : (f32, !llvm.ptr, i32)
  return
}
func.func @floor(%45: !llvm.ptr, %46: !llvm.ptr) {
  %47 = nvvm.read.ptx.sreg.ntid.x : i32
  %48 = nvvm.read.ptx.sreg.ctaid.x : i32
  %49 = nvvm.read.ptx.sreg.tid.x : i32
  %50 = arith.muli %48, %47 : i32
  %51 = arith.addi %50, %49 : i32
  %52 = oven.load %45, %51 : (!llvm.ptr, i32) -> f32
  %53 = math.floor %52 : f32
  oven.store %53, %46, %51 : (f32, !llvm.ptr, i32)
  return
}
func.func @rsqrt(%54: !llvm.ptr, %55: !llvm.ptr) {
  %56 = nvvm.read.ptx.sreg.ntid.x : i32
  %57 = nvvm.read.ptx.sreg.ctaid.x : i32
  %58 = nvvm.read.ptx.sreg.tid.x : i32
  %59 = arith.muli %57, %56 : i32
  %60 = arith.addi %59, %58 : i32
  %61 = oven.load %54, %60 : (!llvm.ptr, i32) -> f32
  %62 = math.rsqrt %61 : f32
  oven.store %62, %55, %60 : (f32, !llvm.ptr, i32)
  return
}