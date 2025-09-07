add_mlir_library(AddNVVMKernel
  AddNVVMKernel.cpp

  ${PROJECT_SOURCE_DIR}/lib/Conversion/AddNVVMKernel/
  ADDITIONAL_HEADER_DIRS

  DEPENDS
  AddNVVMKernelPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRPass
  MLIRTransforms
  )

set(LLVM_TARGET_DEFINITIONS AddNVVMKernel.td)
mlir_tablegen(AddNVVMKernel.h.inc -gen-pass-decls -name AddNVVMKernel)
add_dependencies(mlir-headers MLIROvenOpsIncGen)
add_public_tablegen_target(AddNVVMKernelPassIncGen)
add_mlir_doc(AddNVVMKernel AddNVVMKernel AddNVVMKernel/ -gen-pass-doc)
