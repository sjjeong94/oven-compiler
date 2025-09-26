; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptx_kernel float @__nv_log2f(float)

declare ptx_kernel float @__nv_logf(float)

declare ptx_kernel float @__nv_tanf(float)

declare ptx_kernel float @__nv_sinf(float)

declare ptx_kernel float @__nv_cosf(float)

define ptx_kernel void @cos(ptr %0, ptr %1) {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = mul i32 %4, %3
  %7 = add i32 %6, %5
  %8 = getelementptr float, ptr %0, i32 %7
  %9 = load float, ptr %8, align 4
  %10 = call float @__nv_cosf(float %9)
  %11 = getelementptr float, ptr %1, i32 %7
  store float %10, ptr %11, align 4
  ret void
}

define ptx_kernel void @sin(ptr %0, ptr %1) {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = mul i32 %4, %3
  %7 = add i32 %6, %5
  %8 = getelementptr float, ptr %0, i32 %7
  %9 = load float, ptr %8, align 4
  %10 = call float @__nv_sinf(float %9)
  %11 = getelementptr float, ptr %1, i32 %7
  store float %10, ptr %11, align 4
  ret void
}

define ptx_kernel void @tan(ptr %0, ptr %1) {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = mul i32 %4, %3
  %7 = add i32 %6, %5
  %8 = getelementptr float, ptr %0, i32 %7
  %9 = load float, ptr %8, align 4
  %10 = call float @__nv_tanf(float %9)
  %11 = getelementptr float, ptr %1, i32 %7
  store float %10, ptr %11, align 4
  ret void
}

define ptx_kernel void @log(ptr %0, ptr %1) {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = mul i32 %4, %3
  %7 = add i32 %6, %5
  %8 = getelementptr float, ptr %0, i32 %7
  %9 = load float, ptr %8, align 4
  %10 = call float @__nv_logf(float %9)
  %11 = getelementptr float, ptr %1, i32 %7
  store float %10, ptr %11, align 4
  ret void
}

define ptx_kernel void @log2(ptr %0, ptr %1) {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = mul i32 %4, %3
  %7 = add i32 %6, %5
  %8 = getelementptr float, ptr %0, i32 %7
  %9 = load float, ptr %8, align 4
  %10 = call float @__nv_log2f(float %9)
  %11 = getelementptr float, ptr %1, i32 %7
  store float %10, ptr %11, align 4
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

