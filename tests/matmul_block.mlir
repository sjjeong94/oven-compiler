
// RUN: oven-opt %s --oven-to-llvm | FileCheck %s

func.func @function(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr, %c_ptr: !llvm.ptr, %m: i32, %n: i32, %k: i32) {
  // CHECK-LABEL: llvm.func @function
  // CHECK: llvm.mlir.addressof @smem0 : !llvm.ptr<3>
  %smem = oven.smem : !llvm.ptr<3>

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

  // CHECK: llvm.mul
  %smem_offset0 = arith.muli %tRow, %block_size : i32
  // CHECK: llvm.add
  %smem_offset = arith.addi %smem_offset0, %tCol : i32
  // CHECK: llvm.mul
  %bsmem_offset0 = arith.muli %block_size, %block_size : i32
  // CHECK: llvm.add
  %bsmem_offset = arith.addi %bsmem_offset0, %smem_offset : i32

  %1 = arith.constant 1 : index
  %start = arith.constant 0 : index
  %end = arith.index_cast %k : i32 to index
  %step = arith.index_cast %block_size : i32 to index
  %zerof = arith.constant 0.0 : f32
  // CHECK: llvm.br
  %sum_final = scf.for %i_index = %start to %end step %step iter_args(%sum = %zerof) -> (f32) {
    %i = arith.index_cast %i_index : index to i32
    
    %i_c = arith.addi %i, %tCol : i32
    %i_r = arith.addi %i, %tRow : i32

    %a_offset0 = arith.muli %row, %k : i32
    %a_offset = arith.addi %a_offset0, %i_c : i32
    %b_offset0 = arith.muli %i_r, %n : i32
    %b_offset = arith.addi %b_offset0, %col : i32

    // CHECK: llvm.getelementptr
    // CHECK: llvm.load
    %a = oven.load %a_ptr, %a_offset : (!llvm.ptr, i32) -> f32
    // CHECK: llvm.getelementptr
    // CHECK: llvm.load
    %b = oven.load %b_ptr, %b_offset : (!llvm.ptr, i32) -> f32

    // CHECK: llvm.getelementptr
    // CHECK: llvm.store
    oven.store %a, %smem, %smem_offset : (f32, !llvm.ptr<3>, i32)
    // CHECK: llvm.getelementptr
    // CHECK: llvm.store
    oven.store %b, %smem, %bsmem_offset : (f32, !llvm.ptr<3>, i32)
    // CHECK: nvvm.barrier0
    nvvm.barrier0

    // CHECK: llvm.br
    %partial_sum = scf.for %j_index = %start to %step step %1 iter_args(%sum_inner = %zerof) -> (f32) {
      %j = arith.index_cast %j_index : index to i32

      %as_offset0 = arith.muli %tRow, %block_size : i32
      %as_offset = arith.addi %as_offset0, %j : i32
      %bs_offset0 = arith.muli %j, %block_size : i32
      %bs_offset1 = arith.addi %bs_offset0, %bsmem_offset0 : i32
      %bs_offset = arith.addi %bs_offset1, %tCol : i32

      // CHECK: llvm.getelementptr
      // CHECK: llvm.load
      %a_smem = oven.load %smem, %as_offset : (!llvm.ptr<3>, i32) -> f32
      // CHECK: llvm.getelementptr
      // CHECK: llvm.load
      %b_smem = oven.load %smem, %bs_offset : (!llvm.ptr<3>, i32) -> f32
      // CHECK: llvm.fmul
      %prod = arith.mulf %a_smem, %b_smem : f32
      // CHECK: llvm.fadd
      %sum_inner_new = arith.addf %sum_inner, %prod : f32
      scf.yield %sum_inner_new : f32
    }
    // CHECK: nvvm.barrier0
    nvvm.barrier0
    %sum_new = arith.addf %sum, %partial_sum : f32
    scf.yield %sum_new : f32
  }
  
  %c_offset0 = arith.muli %row, %n : i32
  %c_offset = arith.addi %c_offset0, %col : i32
  // CHECK: llvm.getelementptr
  // CHECK: llvm.store
  oven.store %sum_final, %c_ptr, %c_offset : (f32, !llvm.ptr, i32)
  return
}
