; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define ptx_kernel void @matmul(ptr %0, ptr %1, ptr %2, i32 %3, i32 %4, i32 %5) {
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %9 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %10 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %11 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %12 = mul i32 %8, %7
  %13 = add i32 %12, %10
  %14 = mul i32 %9, %7
  %15 = add i32 %14, %11
  %16 = sext i32 %5 to i64
  br label %17

17:                                               ; preds = %21, %6
  %18 = phi i64 [ %33, %21 ], [ 0, %6 ]
  %19 = phi float [ %32, %21 ], [ 0.000000e+00, %6 ]
  %20 = icmp slt i64 %18, %16
  br i1 %20, label %21, label %34

21:                                               ; preds = %17
  %22 = trunc i64 %18 to i32
  %23 = mul i32 %15, %5
  %24 = add i32 %23, %22
  %25 = mul i32 %22, %4
  %26 = add i32 %25, %13
  %27 = getelementptr float, ptr %0, i32 %24
  %28 = load float, ptr %27, align 4
  %29 = getelementptr float, ptr %1, i32 %26
  %30 = load float, ptr %29, align 4
  %31 = fmul float %28, %30
  %32 = fadd float %19, %31
  %33 = add i64 %18, 1
  br label %17

34:                                               ; preds = %17
  %35 = mul i32 %15, %4
  %36 = add i32 %35, %13
  %37 = getelementptr float, ptr %2, i32 %36
  store float %19, ptr %37, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
