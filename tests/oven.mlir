module {
  func.func @test_type_syntax(%a: i32, %b: i32) -> i32 {
    %c = oven.add %a, %b : (i32, i32) -> i32
    return %c : i32
  }
}