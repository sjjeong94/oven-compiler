add_mlir_library(OvenToLLVM
  OvenToLLVM.cpp

  ${PROJECT_SOURCE_DIR}/lib/Conversion/OvenToLLVM/
  ADDITIONAL_HEADER_DIRS

  DEPENDS
  OvenToLLVMPassIncGen

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

set(LLVM_TARGET_DEFINITIONS OvenToLLVM.td)
mlir_tablegen(OvenToLLVM.h.inc -gen-pass-decls -name OvenToLLVM)
add_dependencies(mlir-headers MLIROvenOpsIncGen)
add_public_tablegen_target(OvenToLLVMPassIncGen)
add_mlir_doc(OvenToLLVM OvenToLLVM OvenToLLVM/ -gen-pass-doc)
