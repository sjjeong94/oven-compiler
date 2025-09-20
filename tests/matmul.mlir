
// RUN: oven-opt %s --oven-to-llvm | FileCheck %s

func.func @function(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr, %c_ptr: !llvm.ptr, %m: i32, %n: i32, %k: i32) {
  // CHECK-LABEL: llvm.func @function
  // CHECK: nvvm.read.ptx.sreg.ntid.x : i32
  %block_size = nvvm.read.ptx.sreg.ntid.x : i32
  // CHECK: nvvm.read.ptx.sreg.ctaid.x : i32
  %cCol = nvvm.read.ptx.sreg.ctaid.x : i32
  // CHECK: nvvm.read.ptx.sreg.ctaid.y : i32
  %cRow = nvvm.read.ptx.sreg.ctaid.y : i32
  // CHECK: nvvm.read.ptx.sreg.tid.x : i32
  %tCol = nvvm.read.ptx.sreg.tid.x : i32
  // CHECK: nvvm.read.ptx.sreg.tid.y : i32
  %tRow = nvvm.read.ptx.sreg.tid.y : i32

  // CHECK: llvm.mul
  %col0 = arith.muli %cCol, %block_size : i32
  // CHECK: llvm.add
  %col = arith.addi %col0, %tCol : i32
  // CHECK: llvm.mul
  %row0 = arith.muli %cRow, %block_size : i32
  // CHECK: llvm.add
  %row = arith.addi %row0, %tRow : i32

  %start = arith.constant 0 : index
  %end = arith.index_cast %k : i32 to index
  %step = arith.constant 1 : index
  %zerof = arith.constant 0.0 : f32
  // CHECK: llvm.br ^bb1
  %sum_final = scf.for %i_index = %start to %end step %step iter_args(%sum = %zerof) -> (f32) {
    %i = arith.index_cast %i_index : index to i32
    // CHECK: llvm.mul
    %a_offset0 = arith.muli %row, %k : i32
    // CHECK: llvm.add
    %a_offset = arith.addi %a_offset0, %i : i32
    // CHECK: llvm.mul
    %b_offset0 = arith.muli %i, %n : i32
    // CHECK: llvm.add
    %b_offset = arith.addi %b_offset0, %col : i32
    // CHECK: llvm.getelementptr
    // CHECK: llvm.load
    %a = oven.load %a_ptr, %a_offset : (!llvm.ptr, i32) -> f32
    // CHECK: llvm.getelementptr
    // CHECK: llvm.load
    %b = oven.load %b_ptr, %b_offset : (!llvm.ptr, i32) -> f32
    // CHECK: llvm.fmul
    %prod = arith.mulf %a, %b : f32
    // CHECK: llvm.fadd
    %sum_new = arith.addf %sum, %prod : f32
    scf.yield %sum_new : f32
  }
  
  // CHECK: llvm.mul
  %c_offset0 = arith.muli %row, %n : i32
  // CHECK: llvm.add
  %c_offset = arith.addi %c_offset0, %col : i32
  // CHECK: llvm.getelementptr
  // CHECK: llvm.store
  oven.store %sum_final, %c_ptr, %c_offset : (f32, !llvm.ptr, i32)
  return
}