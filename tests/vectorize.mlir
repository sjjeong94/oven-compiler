
// RUN: oven-opt %s --oven-to-llvm | FileCheck %s

func.func @function(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  // CHECK-LABEL: llvm.func @function
  // CHECK: llvm.mlir.constant(4 : i32) : i32
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
  %size = arith.constant 4 : i32
  // CHECK: llvm.mul
  %offset = arith.muli %4, %size : i32
  // CHECK: llvm.getelementptr {{.*}} -> !llvm.ptr, f32
  // CHECK: llvm.load {{.*}} -> vector<4xf32>
  %5 = oven.vload %arg0, %offset, 4 : (!llvm.ptr, i32) -> vector<4xf32>
  // CHECK: llvm.getelementptr {{.*}} -> !llvm.ptr, f32
  // CHECK: llvm.store {{.*}} : vector<4xf32>, !llvm.ptr
  oven.vstore %5, %arg1, %offset, 4 : (vector<4xf32>, !llvm.ptr, i32)
  return
}