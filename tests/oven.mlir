module {
  func.func @test_type_syntax(%a: f32, %b: f32) -> i32 {
    %0 = oven.block_idx.x : i32
    %1 = oven.block_idx.y : i32
    %2 = oven.block_idx.z : i32
    %3 = oven.thread_idx.x : i32
    %4 = oven.thread_idx.y : i32
    %5 = oven.thread_idx.z : i32

    %6 = arith.constant 1024 : i32 // context
    %7 = arith.muli %0, %6 : i32
    %8 = arith.addi %7, %3 : i32
    return %8 : i32
  }
}