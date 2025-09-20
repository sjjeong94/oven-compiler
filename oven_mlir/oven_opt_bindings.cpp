#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "lib/Conversion/AddNVVMKernel/AddNVVMKernel.h"
#include "lib/Conversion/OvenToLLVM/OvenToLLVM.h"
#include "lib/Dialect/Oven/IR/OvenDialect.h"
#include "lib/Dialect/Oven/Transform/Passes.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"

namespace nb = nanobind;

// Helper function to run the oven-to-llvm pipeline
void runOvenToLLVMPipeline(mlir::OpPassManager &manager) {
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

class OvenOptimizer {
private:
  std::unique_ptr<mlir::MLIRContext> context;
  
public:
  OvenOptimizer() {
    context = std::make_unique<mlir::MLIRContext>();
    
    // Register dialects
    mlir::DialectRegistry registry;
    registry.insert<mlir::oven::OvenDialect>();
    mlir::registerAllDialects(registry);
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
  
  std::string optimize_mlir(const std::string& mlir_code) {
    // Parse the MLIR code
    auto sourceBuffer = llvm::MemoryBuffer::getMemBuffer(mlir_code);
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(sourceBuffer), llvm::SMLoc());
    
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, context.get());
    if (!module) {
      return "Error: Failed to parse MLIR code";
    }
    
    // Create and run the pass manager
    mlir::PassManager pm(context.get());
    runOvenToLLVMPipeline(pm);
    
    if (mlir::failed(pm.run(module.get()))) {
      return "Error: Failed to run optimization passes";
    }
    
    // Convert to string
    std::string result;
    llvm::raw_string_ostream stream(result);
    module->print(stream);
    return result;
  }
  
  std::string optimize_file(const std::string& filename) {
    // Read file
    auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
    if (std::error_code error = fileOrErr.getError()) {
      return "Error: Could not open file " + filename + ": " + error.message();
    }
    
    // Parse the MLIR file
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, context.get());
    if (!module) {
      return "Error: Failed to parse MLIR file " + filename;
    }
    
    // Create and run the pass manager
    mlir::PassManager pm(context.get());
    runOvenToLLVMPipeline(pm);
    
    if (mlir::failed(pm.run(module.get()))) {
      return "Error: Failed to run optimization passes on " + filename;
    }
    
    // Convert to string
    std::string result;
    llvm::raw_string_ostream stream(result);
    module->print(stream);
    return result;
  }
  
  std::string to_llvm_ir(const std::string& mlir_code) {
    // First optimize the MLIR
    std::string optimized = optimize_mlir(mlir_code);
    if (optimized.find("Error:") == 0) {
      return optimized;
    }
    
    // Parse the optimized MLIR
    auto sourceBuffer = llvm::MemoryBuffer::getMemBuffer(optimized);
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(sourceBuffer), llvm::SMLoc());
    
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, context.get());
    if (!module) {
      return "Error: Failed to parse optimized MLIR";
    }
    
    // Convert to LLVM IR
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module.get(), llvmContext);
    if (!llvmModule) {
      return "Error: Failed to translate to LLVM IR";
    }
    
    std::string result;
    llvm::raw_string_ostream stream(result);
    llvmModule->print(stream, nullptr);
    return result;
  }
};

NB_MODULE(oven_opt_py, m) {
    m.doc() = "Python bindings for oven-opt MLIR compiler";
    
    nb::class_<OvenOptimizer>(m, "OvenOptimizer")
        .def(nb::init<>(), "Create a new OvenOptimizer instance")
        .def("optimize_mlir", &OvenOptimizer::optimize_mlir, 
             "Optimize MLIR code string and return the result",
             nb::arg("mlir_code"))
        .def("optimize_file", &OvenOptimizer::optimize_file,
             "Optimize MLIR file and return the result", 
             nb::arg("filename"))
        .def("to_llvm_ir", &OvenOptimizer::to_llvm_ir,
             "Convert MLIR code to LLVM IR",
             nb::arg("mlir_code"));
             
    // Convenience functions
    m.def("optimize_string", [](const std::string& code) {
        OvenOptimizer opt;
        return opt.optimize_mlir(code);
    }, "Optimize MLIR code string", nb::arg("code"));
    
    m.def("optimize_file", [](const std::string& filename) {
        OvenOptimizer opt;
        return opt.optimize_file(filename);
    }, "Optimize MLIR file", nb::arg("filename"));
    
    m.def("to_llvm_ir", [](const std::string& code) {
        OvenOptimizer opt;
        return opt.to_llvm_ir(code);
    }, "Convert MLIR code to LLVM IR", nb::arg("code"));
}