; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define ptx_kernel void @add(ptr %0, ptr %1, ptr %2) {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %7 = mul i32 %5, %4
  %8 = add i32 %7, %6
  %9 = getelementptr float, ptr %0, i32 %8
  %10 = load float, ptr %9, align 4
  %11 = getelementptr float, ptr %1, i32 %8
  %12 = load float, ptr %11, align 4
  %13 = fadd float %10, %12
  %14 = getelementptr float, ptr %2, i32 %8
  store float %13, ptr %14, align 4
  ret void
}

define ptx_kernel void @mul(ptr %0, ptr %1, ptr %2) {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %7 = mul i32 %5, %4
  %8 = add i32 %7, %6
  %9 = getelementptr float, ptr %0, i32 %8
  %10 = load float, ptr %9, align 4
  %11 = getelementptr float, ptr %1, i32 %8
  %12 = load float, ptr %11, align 4
  %13 = fmul float %10, %12
  %14 = getelementptr float, ptr %2, i32 %8
  store float %13, ptr %14, align 4
  ret void
}

define ptx_kernel void @sub(ptr %0, ptr %1, ptr %2) {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %7 = mul i32 %5, %4
  %8 = add i32 %7, %6
  %9 = getelementptr float, ptr %0, i32 %8
  %10 = load float, ptr %9, align 4
  %11 = getelementptr float, ptr %1, i32 %8
  %12 = load float, ptr %11, align 4
  %13 = fsub float %10, %12
  %14 = getelementptr float, ptr %2, i32 %8
  store float %13, ptr %14, align 4
  ret void
}

define ptx_kernel void @div(ptr %0, ptr %1, ptr %2) {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %7 = mul i32 %5, %4
  %8 = add i32 %7, %6
  %9 = getelementptr float, ptr %0, i32 %8
  %10 = load float, ptr %9, align 4
  %11 = getelementptr float, ptr %1, i32 %8
  %12 = load float, ptr %11, align 4
  %13 = fdiv float %10, %12
  %14 = getelementptr float, ptr %2, i32 %8
  store float %13, ptr %14, align 4
  ret void
}

define ptx_kernel void @vadd(ptr %0, ptr %1, ptr %2) {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %7 = mul i32 %5, %4
  %8 = add i32 %7, %6
  %9 = mul i32 %8, 4
  %10 = getelementptr float, ptr %0, i32 %9
  %11 = load <4 x float>, ptr %10, align 16
  %12 = getelementptr float, ptr %1, i32 %9
  %13 = load <4 x float>, ptr %12, align 16
  %14 = fadd <4 x float> %11, %13
  %15 = getelementptr float, ptr %2, i32 %9
  store <4 x float> %14, ptr %15, align 16
  ret void
}

define ptx_kernel void @vmul(ptr %0, ptr %1, ptr %2) {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %7 = mul i32 %5, %4
  %8 = add i32 %7, %6
  %9 = mul i32 %8, 4
  %10 = getelementptr float, ptr %0, i32 %9
  %11 = load <4 x float>, ptr %10, align 16
  %12 = getelementptr float, ptr %1, i32 %9
  %13 = load <4 x float>, ptr %12, align 16
  %14 = fmul <4 x float> %11, %13
  %15 = getelementptr float, ptr %2, i32 %9
  store <4 x float> %14, ptr %15, align 16
  ret void
}

define ptx_kernel void @vsub(ptr %0, ptr %1, ptr %2) {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %7 = mul i32 %5, %4
  %8 = add i32 %7, %6
  %9 = mul i32 %8, 4
  %10 = getelementptr float, ptr %0, i32 %9
  %11 = load <4 x float>, ptr %10, align 16
  %12 = getelementptr float, ptr %1, i32 %9
  %13 = load <4 x float>, ptr %12, align 16
  %14 = fsub <4 x float> %11, %13
  %15 = getelementptr float, ptr %2, i32 %9
  store <4 x float> %14, ptr %15, align 16
  ret void
}

define ptx_kernel void @vdiv(ptr %0, ptr %1, ptr %2) {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %7 = mul i32 %5, %4
  %8 = add i32 %7, %6
  %9 = mul i32 %8, 4
  %10 = getelementptr float, ptr %0, i32 %9
  %11 = load <4 x float>, ptr %10, align 16
  %12 = getelementptr float, ptr %1, i32 %9
  %13 = load <4 x float>, ptr %12, align 16
  %14 = fdiv <4 x float> %11, %13
  %15 = getelementptr float, ptr %2, i32 %9
  store <4 x float> %14, ptr %15, align 16
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
