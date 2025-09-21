; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define ptx_kernel void @sigmoid(ptr %0, ptr %1) {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = mul i32 %4, %3
  %7 = add i32 %6, %5
  %8 = getelementptr float, ptr %0, i32 %7
  %9 = load float, ptr %8, align 4
  %10 = fneg float %9
  %11 = fmul float %10, 0x3FF7154760000000
  %12 = call float @llvm.exp2.f32(float %11)
  %13 = fadd float %12, 1.000000e+00
  %14 = fdiv float 1.000000e+00, %13
  %15 = getelementptr float, ptr %1, i32 %7
  store float %14, ptr %15, align 4
  ret void
}

define ptx_kernel void @exp(ptr %0, ptr %1) {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = mul i32 %4, %3
  %7 = add i32 %6, %5
  %8 = getelementptr float, ptr %0, i32 %7
  %9 = load float, ptr %8, align 4
  %10 = fmul float %9, 0x3FF7154760000000
  %11 = call float @llvm.exp2.f32(float %10)
  %12 = getelementptr float, ptr %1, i32 %7
  store float %11, ptr %12, align 4
  ret void
}

define ptx_kernel void @sqrt(ptr %0, ptr %1) {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = mul i32 %4, %3
  %7 = add i32 %6, %5
  %8 = getelementptr float, ptr %0, i32 %7
  %9 = load float, ptr %8, align 4
  %10 = call float @llvm.sqrt.f32(float %9)
  %11 = getelementptr float, ptr %1, i32 %7
  store float %10, ptr %11, align 4
  ret void
}

define ptx_kernel void @abs(ptr %0, ptr %1) {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = mul i32 %4, %3
  %7 = add i32 %6, %5
  %8 = getelementptr float, ptr %0, i32 %7
  %9 = load float, ptr %8, align 4
  %10 = call float @llvm.fabs.f32(float %9)
  %11 = getelementptr float, ptr %1, i32 %7
  store float %10, ptr %11, align 4
  ret void
}

define ptx_kernel void @ceil(ptr %0, ptr %1) {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = mul i32 %4, %3
  %7 = add i32 %6, %5
  %8 = getelementptr float, ptr %0, i32 %7
  %9 = load float, ptr %8, align 4
  %10 = call float @llvm.ceil.f32(float %9)
  %11 = getelementptr float, ptr %1, i32 %7
  store float %10, ptr %11, align 4
  ret void
}

define ptx_kernel void @floor(ptr %0, ptr %1) {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = mul i32 %4, %3
  %7 = add i32 %6, %5
  %8 = getelementptr float, ptr %0, i32 %7
  %9 = load float, ptr %8, align 4
  %10 = call float @llvm.floor.f32(float %9)
  %11 = getelementptr float, ptr %1, i32 %7
  store float %10, ptr %11, align 4
  ret void
}

define ptx_kernel void @rsqrt(ptr %0, ptr %1) {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = mul i32 %4, %3
  %7 = add i32 %6, %5
  %8 = getelementptr float, ptr %0, i32 %7
  %9 = load float, ptr %8, align 4
  %10 = call float @llvm.sqrt.f32(float %9)
  %11 = fdiv float 1.000000e+00, %10
  %12 = getelementptr float, ptr %1, i32 %7
  store float %11, ptr %12, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp2.f32(float) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.ceil.f32(float) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.floor.f32(float) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
