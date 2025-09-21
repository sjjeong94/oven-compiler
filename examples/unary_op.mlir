// Generated MLIR code from Python source

func.func @sigmoid(%x_ptr: !llvm.ptr, %y_ptr: !llvm.ptr) {
  %0 = nvvm.read.ptx.sreg.ntid.x : i32
  %1 = nvvm.read.ptx.sreg.ctaid.x : i32
  %2 = nvvm.read.ptx.sreg.tid.x : i32
  %3 = arith.muli %1, %0 : i32
  %4 = arith.addi %3, %2 : i32
  %5 = oven.load %x_ptr, %4 : (!llvm.ptr, i32) -> f32
  %6 = oven.sigmoid %5 : f32 -> f32
  oven.store %6, %y_ptr, %4 : (f32, !llvm.ptr, i32)
  return
}

func.func @exp(%x_ptr: !llvm.ptr, %y_ptr: !llvm.ptr) {
  %7 = nvvm.read.ptx.sreg.ntid.x : i32
  %8 = nvvm.read.ptx.sreg.ctaid.x : i32
  %9 = nvvm.read.ptx.sreg.tid.x : i32
  %10 = arith.muli %8, %7 : i32
  %11 = arith.addi %10, %9 : i32
  %12 = oven.load %x_ptr, %11 : (!llvm.ptr, i32) -> f32
  %13 = math.exp %12 : f32
  oven.store %13, %y_ptr, %11 : (f32, !llvm.ptr, i32)
  return
}

func.func @sqrt(%x_ptr: !llvm.ptr, %y_ptr: !llvm.ptr) {
  %14 = nvvm.read.ptx.sreg.ntid.x : i32
  %15 = nvvm.read.ptx.sreg.ctaid.x : i32
  %16 = nvvm.read.ptx.sreg.tid.x : i32
  %17 = arith.muli %15, %14 : i32
  %18 = arith.addi %17, %16 : i32
  %19 = oven.load %x_ptr, %18 : (!llvm.ptr, i32) -> f32
  %20 = math.sqrt %19 : f32
  oven.store %20, %y_ptr, %18 : (f32, !llvm.ptr, i32)
  return
}

func.func @abs(%x_ptr: !llvm.ptr, %y_ptr: !llvm.ptr) {
  %21 = nvvm.read.ptx.sreg.ntid.x : i32
  %22 = nvvm.read.ptx.sreg.ctaid.x : i32
  %23 = nvvm.read.ptx.sreg.tid.x : i32
  %24 = arith.muli %22, %21 : i32
  %25 = arith.addi %24, %23 : i32
  %26 = oven.load %x_ptr, %25 : (!llvm.ptr, i32) -> f32
  %27 = math.absf %26 : f32
  oven.store %27, %y_ptr, %25 : (f32, !llvm.ptr, i32)
  return
}

func.func @ceil(%x_ptr: !llvm.ptr, %y_ptr: !llvm.ptr) {
  %28 = nvvm.read.ptx.sreg.ntid.x : i32
  %29 = nvvm.read.ptx.sreg.ctaid.x : i32
  %30 = nvvm.read.ptx.sreg.tid.x : i32
  %31 = arith.muli %29, %28 : i32
  %32 = arith.addi %31, %30 : i32
  %33 = oven.load %x_ptr, %32 : (!llvm.ptr, i32) -> f32
  %34 = math.ceil %33 : f32
  oven.store %34, %y_ptr, %32 : (f32, !llvm.ptr, i32)
  return
}

func.func @floor(%x_ptr: !llvm.ptr, %y_ptr: !llvm.ptr) {
  %35 = nvvm.read.ptx.sreg.ntid.x : i32
  %36 = nvvm.read.ptx.sreg.ctaid.x : i32
  %37 = nvvm.read.ptx.sreg.tid.x : i32
  %38 = arith.muli %36, %35 : i32
  %39 = arith.addi %38, %37 : i32
  %40 = oven.load %x_ptr, %39 : (!llvm.ptr, i32) -> f32
  %41 = math.floor %40 : f32
  oven.store %41, %y_ptr, %39 : (f32, !llvm.ptr, i32)
  return
}

func.func @rsqrt(%x_ptr: !llvm.ptr, %y_ptr: !llvm.ptr) {
  %42 = nvvm.read.ptx.sreg.ntid.x : i32
  %43 = nvvm.read.ptx.sreg.ctaid.x : i32
  %44 = nvvm.read.ptx.sreg.tid.x : i32
  %45 = arith.muli %43, %42 : i32
  %46 = arith.addi %45, %44 : i32
  %47 = oven.load %x_ptr, %46 : (!llvm.ptr, i32) -> f32
  %48 = math.rsqrt %47 : f32
  oven.store %48, %y_ptr, %46 : (f32, !llvm.ptr, i32)
  return
}

