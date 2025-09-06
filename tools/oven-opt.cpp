#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "lib/Conversion/OvenToLLVM/OvenToLLVM.h"
#include "lib/Dialect/Oven/IR/OvenDialect.h"

void ovenToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::oven::createOvenToLLVM());
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::oven::OvenDialect>();
  registry.insert<mlir::NVVM::NVVMDialect>();
  mlir::registerAllDialects(registry);

  mlir::PassPipelineRegistration<>(
      "oven-to-llvm", "Run passes to lower the oven dialect to LLVM",
      ovenToLLVMPipelineBuilder);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Oven Opt", registry));
}
