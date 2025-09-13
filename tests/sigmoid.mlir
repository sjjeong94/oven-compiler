
func.func @function(%a: !llvm.ptr, %b: !llvm.ptr) {
  %0 = nvvm.read.ptx.sreg.ntid.x : i32
  %1 = nvvm.read.ptx.sreg.ctaid.x : i32
  %2 = nvvm.read.ptx.sreg.tid.x : i32
  %3 = arith.muli %0, %1 : i32
  %4 = arith.addi %2, %3 : i32
  %5 = oven.load %a, %4 : (!llvm.ptr, i32) -> f32
  %6 = oven.sigmoid %5 : f32 -> f32
  oven.store %6, %b, %4 : (f32, !llvm.ptr, i32)
  return
}