
func.func @function(%aptr: !llvm.ptr, %bptr: !llvm.ptr, %cptr: !llvm.ptr, %m: i32, %n: i32, %k: i32) {
  %dim_x = nvvm.read.ptx.sreg.ntid.x : i32
  %dim_y = nvvm.read.ptx.sreg.ntid.y : i32
  %bid_x = nvvm.read.ptx.sreg.ctaid.x : i32
  %bid_y = nvvm.read.ptx.sreg.ctaid.y : i32
  %tid_x = nvvm.read.ptx.sreg.tid.x : i32
  %tid_y = nvvm.read.ptx.sreg.tid.y : i32

  %0 = arith.muli %bid_x, %dim_x : i32
  %x = arith.addi %0, %tid_x : i32
  %1 = arith.muli %bid_y, %dim_y : i32
  %y = arith.addi %1, %tid_y : i32
  %2 = arith.muli %x, %n : i32
  %offset = arith.addi %2, %y : i32

  %zerof = arith.constant 0.0 : f32
  %one = arith.constant 1 : index
  %zero = arith.constant 0 : index
  %k_index = arith.index_cast %k : i32 to index

  %sum_final = scf.for %i_index = %zero to %k_index step %one iter_args(%sum = %zerof) -> (f32) {
    %i = arith.index_cast %i_index : index to i32
    %a_offset_ = arith.muli %x, %k : i32
    %a_offset = arith.addi %a_offset_, %i : i32
    %b_offset_ = arith.muli %i, %n : i32
    %b_offset = arith.addi %b_offset_, %y : i32
    %a = oven.load %aptr, %a_offset : (!llvm.ptr, i32) -> f32
    %b = oven.load %bptr, %b_offset : (!llvm.ptr, i32) -> f32
    %prod = arith.mulf %a, %b : f32
    %sum_new = arith.addf %sum, %prod : f32
    scf.yield %sum_new : f32
  }
  oven.store %sum_final, %cptr, %offset : (f32, !llvm.ptr, i32)
  return
}