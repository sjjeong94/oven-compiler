
// RUN: oven-opt %s --oven-to-llvm | FileCheck %s

func.func @function(%a: !llvm.ptr, %b: !llvm.ptr) {
  // CHECK-LABEL: llvm.func @function
  // CHECK: llvm.mlir.addressof @smem0 : !llvm.ptr<3>
  %smem = oven.smem : !llvm.ptr<3>
  %smem2 = oven.smem : !llvm.ptr<3>
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
  oven.store %5, %smem, %2 : (f32, !llvm.ptr<3>, i32)
  // CHECK: nvvm.barrier0
  nvvm.barrier0
  // CHECK: llvm.getelementptr
  // CHECK: llvm.load
  %6 = oven.load %smem, %2 : (!llvm.ptr<3>, i32) -> f32
  // CHECK: llvm.getelementptr
  // CHECK: llvm.store
  oven.store %6, %b, %4 : (f32, !llvm.ptr, i32)
  return
}
