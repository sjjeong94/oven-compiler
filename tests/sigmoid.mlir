
// RUN: oven-opt %s --oven-to-llvm | FileCheck %s

func.func @function(%a: !llvm.ptr, %b: !llvm.ptr) {
  // CHECK-LABEL: llvm.func @function
  // CHECK: nvvm.read.ptx.sreg.ntid.x : i32
  %0 = nvvm.read.ptx.sreg.ntid.x : i32
  // CHECK: nvvm.read.ptx.sreg.ctaid.x : i32
  %1 = nvvm.read.ptx.sreg.ctaid.x : i32
  // CHECK: nvvm.read.ptx.sreg.tid.x : i32
  %2 = nvvm.read.ptx.sreg.tid.x : i32
  // CHECK: llvm.mul
  %3 = arith.muli %0, %1 : i32
  // CHECK: llvm.add
  %4 = arith.addi %2, %3 : i32
  // CHECK: llvm.getelementptr
  // CHECK: llvm.load
  %5 = oven.load %a, %4 : (!llvm.ptr, i32) -> f32
  // CHECK: llvm.fneg
  // CHECK: llvm.fmul
  // CHECK: llvm.intr.exp2
  // CHECK: llvm.fadd
  // CHECK: llvm.fdiv
  %6 = oven.sigmoid %5 : f32 -> f32
  // CHECK: llvm.getelementptr
  // CHECK: llvm.store
  oven.store %6, %b, %4 : (f32, !llvm.ptr, i32)
  return
}