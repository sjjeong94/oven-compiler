
func.func @function(%a: !llvm.ptr, %b: !llvm.ptr) {
  %smem = oven.smem : !llvm.ptr<3>
  %smem2 = oven.smem : !llvm.ptr<3>
  %0 = nvvm.read.ptx.sreg.ntid.x : i32
  %1 = nvvm.read.ptx.sreg.ctaid.x : i32
  %2 = nvvm.read.ptx.sreg.tid.x : i32
  %3 = arith.muli %1, %0 : i32
  %4 = arith.addi %3, %2 : i32
  %5 = oven.load %a, %4 : (!llvm.ptr, i32) -> f32
  oven.store %5, %smem, %2 : (f32, !llvm.ptr<3>, i32)
  nvvm.barrier0
  %6 = oven.load %smem, %2 : (!llvm.ptr<3>, i32) -> f32
  oven.store %6, %b, %4 : (f32, !llvm.ptr, i32)
  return
}
