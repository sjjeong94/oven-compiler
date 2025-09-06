module {
  func.func @test_llvm(%a: f32, %b: f32) -> (f32, f32) {
    return %a, %b : f32, f32
  }

  func.func @test_load_store(%a: !llvm.ptr, %b: !llvm.ptr) {
    %0 = llvm.mlir.constant(1024 : i32) : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = llvm.mul %1, %0 : i32
    %4 = llvm.add %3, %2 : i32
    %9 = oven.load %a, %4 : (!llvm.ptr, i32) -> f32
    oven.store %9, %b, %4 : (f32, !llvm.ptr, i32)
    return
  }
}