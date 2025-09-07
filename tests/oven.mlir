
func.func @test_load_store(%a: !llvm.ptr, %b: !llvm.ptr) {
  %0 = arith.constant 128 : i32
  %1 = nvvm.read.ptx.sreg.ctaid.x : i32
  %2 = nvvm.read.ptx.sreg.tid.x : i32
  %3 = arith.muli %1, %0 : i32
  %4 = arith.addi %3, %2 : i32
  %5 = oven.load %a, %4 : (!llvm.ptr, i32) -> f32
  oven.store %5, %b, %4 : (f32, !llvm.ptr, i32)
  return
}

func.func @test_exp(%a: f32) -> f32 {
  %0 = math.exp %a : f32
  return %0 : f32
}

func.func @test_sigmoid(%a: f32) -> f32 {
  %0 = oven.sigmoid %a : f32 -> f32
  return %0 : f32
}
