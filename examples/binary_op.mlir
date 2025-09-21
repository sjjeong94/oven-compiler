// Generated MLIR code from Python source

func.func @add(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr, %c_ptr: !llvm.ptr) {
  %0 = nvvm.read.ptx.sreg.ntid.x : i32
  %1 = nvvm.read.ptx.sreg.ctaid.x : i32
  %2 = nvvm.read.ptx.sreg.tid.x : i32
  %3 = arith.muli %1, %0 : i32
  %4 = arith.addi %3, %2 : i32
  %5 = oven.load %a_ptr, %4 : (!llvm.ptr, i32) -> f32
  %6 = oven.load %b_ptr, %4 : (!llvm.ptr, i32) -> f32
  %7 = arith.addf %5, %6 : f32
  oven.store %7, %c_ptr, %4 : (f32, !llvm.ptr, i32)
  return
}

func.func @mul(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr, %c_ptr: !llvm.ptr) {
  %8 = nvvm.read.ptx.sreg.ntid.x : i32
  %9 = nvvm.read.ptx.sreg.ctaid.x : i32
  %10 = nvvm.read.ptx.sreg.tid.x : i32
  %11 = arith.muli %9, %8 : i32
  %12 = arith.addi %11, %10 : i32
  %13 = oven.load %a_ptr, %12 : (!llvm.ptr, i32) -> f32
  %14 = oven.load %b_ptr, %12 : (!llvm.ptr, i32) -> f32
  %15 = arith.mulf %13, %14 : f32
  oven.store %15, %c_ptr, %12 : (f32, !llvm.ptr, i32)
  return
}

func.func @sub(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr, %c_ptr: !llvm.ptr) {
  %16 = nvvm.read.ptx.sreg.ntid.x : i32
  %17 = nvvm.read.ptx.sreg.ctaid.x : i32
  %18 = nvvm.read.ptx.sreg.tid.x : i32
  %19 = arith.muli %17, %16 : i32
  %20 = arith.addi %19, %18 : i32
  %21 = oven.load %a_ptr, %20 : (!llvm.ptr, i32) -> f32
  %22 = oven.load %b_ptr, %20 : (!llvm.ptr, i32) -> f32
  %23 = arith.subf %21, %22 : f32
  oven.store %23, %c_ptr, %20 : (f32, !llvm.ptr, i32)
  return
}

func.func @div(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr, %c_ptr: !llvm.ptr) {
  %24 = nvvm.read.ptx.sreg.ntid.x : i32
  %25 = nvvm.read.ptx.sreg.ctaid.x : i32
  %26 = nvvm.read.ptx.sreg.tid.x : i32
  %27 = arith.muli %25, %24 : i32
  %28 = arith.addi %27, %26 : i32
  %29 = oven.load %a_ptr, %28 : (!llvm.ptr, i32) -> f32
  %30 = oven.load %b_ptr, %28 : (!llvm.ptr, i32) -> f32
  %31 = arith.divf %29, %30 : f32
  oven.store %31, %c_ptr, %28 : (f32, !llvm.ptr, i32)
  return
}

func.func @vadd(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr, %c_ptr: !llvm.ptr) {
  %32 = nvvm.read.ptx.sreg.ntid.x : i32
  %33 = nvvm.read.ptx.sreg.ctaid.x : i32
  %34 = nvvm.read.ptx.sreg.tid.x : i32
  %35 = arith.muli %33, %32 : i32
  %36 = arith.addi %35, %34 : i32
  %37 = arith.constant 4 : i32
  %38 = arith.muli %36, %37 : i32
  %39 = oven.vload %a_ptr, %38, 4 : (!llvm.ptr, i32) -> vector<4xf32>
  %40 = oven.vload %b_ptr, %38, 4 : (!llvm.ptr, i32) -> vector<4xf32>
  %41 = arith.addf %39, %40 : vector<4xf32>
  oven.vstore %41, %c_ptr, %38, 4 : (vector<4xf32>, !llvm.ptr, i32)
  return
}

func.func @vmul(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr, %c_ptr: !llvm.ptr) {
  %42 = nvvm.read.ptx.sreg.ntid.x : i32
  %43 = nvvm.read.ptx.sreg.ctaid.x : i32
  %44 = nvvm.read.ptx.sreg.tid.x : i32
  %45 = arith.muli %43, %42 : i32
  %46 = arith.addi %45, %44 : i32
  %47 = arith.constant 4 : i32
  %48 = arith.muli %46, %47 : i32
  %49 = oven.vload %a_ptr, %48, 4 : (!llvm.ptr, i32) -> vector<4xf32>
  %50 = oven.vload %b_ptr, %48, 4 : (!llvm.ptr, i32) -> vector<4xf32>
  %51 = arith.mulf %49, %50 : vector<4xf32>
  oven.vstore %51, %c_ptr, %48, 4 : (vector<4xf32>, !llvm.ptr, i32)
  return
}

func.func @vsub(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr, %c_ptr: !llvm.ptr) {
  %52 = nvvm.read.ptx.sreg.ntid.x : i32
  %53 = nvvm.read.ptx.sreg.ctaid.x : i32
  %54 = nvvm.read.ptx.sreg.tid.x : i32
  %55 = arith.muli %53, %52 : i32
  %56 = arith.addi %55, %54 : i32
  %57 = arith.constant 4 : i32
  %58 = arith.muli %56, %57 : i32
  %59 = oven.vload %a_ptr, %58, 4 : (!llvm.ptr, i32) -> vector<4xf32>
  %60 = oven.vload %b_ptr, %58, 4 : (!llvm.ptr, i32) -> vector<4xf32>
  %61 = arith.subf %59, %60 : vector<4xf32>
  oven.vstore %61, %c_ptr, %58, 4 : (vector<4xf32>, !llvm.ptr, i32)
  return
}

func.func @vdiv(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr, %c_ptr: !llvm.ptr) {
  %62 = nvvm.read.ptx.sreg.ntid.x : i32
  %63 = nvvm.read.ptx.sreg.ctaid.x : i32
  %64 = nvvm.read.ptx.sreg.tid.x : i32
  %65 = arith.muli %63, %62 : i32
  %66 = arith.addi %65, %64 : i32
  %67 = arith.constant 4 : i32
  %68 = arith.muli %66, %67 : i32
  %69 = oven.vload %a_ptr, %68, 4 : (!llvm.ptr, i32) -> vector<4xf32>
  %70 = oven.vload %b_ptr, %68, 4 : (!llvm.ptr, i32) -> vector<4xf32>
  %71 = arith.divf %69, %70 : vector<4xf32>
  oven.vstore %71, %c_ptr, %68, 4 : (vector<4xf32>, !llvm.ptr, i32)
  return
}

