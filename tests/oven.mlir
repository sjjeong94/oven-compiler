
// RUN: oven-opt %s --oven-to-llvm | FileCheck %s

func.func @test_load_store(%a: !llvm.ptr, %b: !llvm.ptr) {
  // CHECK-LABEL: llvm.func @test_load_store
  // CHECK: nvvm.read.ptx.sreg.ntid.x : i32
  %0 = nvvm.read.ptx.sreg.ntid.x : i32
  // CHECK: nvvm.read.ptx.sreg.ctaid.x : i32
  %1 = nvvm.read.ptx.sreg.ctaid.x : i32
  // CHECK: nvvm.read.ptx.sreg.tid.x : i32
  %2 = nvvm.read.ptx.sreg.tid.x : i32
  // CHECK: llvm.mul
  %3 = arith.muli %1, %0 : i32
  // CHECK: llvm.add
  %4 = arith.addi %3, %2 : i32
  // CHECK: llvm.getelementptr
  // CHECK: llvm.load
  %5 = oven.load %a, %4 : (!llvm.ptr, i32) -> f32
  // CHECK: llvm.getelementptr
  // CHECK: llvm.store
  oven.store %5, %b, %4 : (f32, !llvm.ptr, i32)
  return
}

func.func @test_exp(%a: f32) -> f32 {
  // CHECK-LABEL: llvm.func @test_exp
  // CHECK: llvm.mlir.constant(1.44269502 : f32)
  // CHECK: llvm.fmul
  // CHECK: llvm.intr.exp2
  %0 = math.exp %a : f32
  return %0 : f32
}

func.func @test_sigmoid(%a: f32) -> f32 {
  // CHECK-LABEL: llvm.func @test_sigmoid
  // CHECK: llvm.fneg
  // CHECK: llvm.fmul
  // CHECK: llvm.intr.exp2
  // CHECK: llvm.fadd
  // CHECK: llvm.fdiv
  %0 = oven.sigmoid %a : f32 -> f32
  return %0 : f32
}
