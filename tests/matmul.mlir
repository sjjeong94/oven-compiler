
func.func @function(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr, %c_ptr: !llvm.ptr, %m: i32, %n: i32, %k: i32) {
  %block_size = nvvm.read.ptx.sreg.ntid.x : i32
  %cCol = nvvm.read.ptx.sreg.ctaid.x : i32
  %cRow = nvvm.read.ptx.sreg.ctaid.y : i32
  %tCol = nvvm.read.ptx.sreg.tid.x : i32
  %tRow = nvvm.read.ptx.sreg.tid.y : i32

  %col0 = arith.muli %cCol, %block_size : i32
  %col = arith.addi %col0, %tCol : i32
  %row0 = arith.muli %cRow, %block_size : i32
  %row = arith.addi %row0, %tRow : i32

  %start = arith.constant 0 : index
  %end = arith.index_cast %k : i32 to index
  %step = arith.constant 1 : index
  %zerof = arith.constant 0.0 : f32
  %sum_final = scf.for %i_index = %start to %end step %step iter_args(%sum = %zerof) -> (f32) {
    %i = arith.index_cast %i_index : index to i32
    %a_offset0 = arith.muli %row, %k : i32
    %a_offset = arith.addi %a_offset0, %i : i32
    %b_offset0 = arith.muli %i, %n : i32
    %b_offset = arith.addi %b_offset0, %col : i32
    %a = oven.load %a_ptr, %a_offset : (!llvm.ptr, i32) -> f32
    %b = oven.load %b_ptr, %b_offset : (!llvm.ptr, i32) -> f32
    %prod = arith.mulf %a, %b : f32
    %sum_new = arith.addf %sum, %prod : f32
    scf.yield %sum_new : f32
  }
  
  %c_offset0 = arith.muli %row, %n : i32
  %c_offset = arith.addi %c_offset0, %col : i32
  oven.store %sum_final, %c_ptr, %c_offset : (f32, !llvm.ptr, i32)
  return
}