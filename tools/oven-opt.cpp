#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "lib/Conversion/AddNVVMKernel/AddNVVMKernel.h"
#include "lib/Conversion/OvenToLLVM/OvenToLLVM.h"
#include "lib/Dialect/Oven/IR/OvenDialect.h"
#include "lib/Dialect/Oven/Transform/Passes.h"

void ovenToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  manager.addPass(mlir::oven::createOptimizeOven());
  manager.addPass(mlir::oven::createOvenToLLVM());
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createConvertSCFToCFPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertMathToLLVMPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::oven::createAddNVVMKernel());
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::oven::OvenDialect>();
  mlir::registerAllDialects(registry);

  mlir::PassPipelineRegistration<>(
      "oven-to-llvm", "Run passes to lower the oven dialect to LLVM",
      ovenToLLVMPipelineBuilder);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Oven Opt", registry));
}
