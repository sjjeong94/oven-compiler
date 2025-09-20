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
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"

#include "lib/Conversion/AddNVVMKernel/AddNVVMKernel.h"
#include "lib/Conversion/OvenToLLVM/OvenToLLVM.h"
#include "lib/Dialect/Oven/IR/OvenDialect.h"
#include "lib/Dialect/Oven/Transform/Passes.h"

namespace nb = nanobind;

// Helper function to initialize LLVM targets for PTX generation
void initializeLLVMTargets() {
  static bool initialized = false;
  if (!initialized) {
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();
    initialized = true;
  }
}

// Helper function to compile LLVM IR to PTX
std::string compileToPTX(llvm::Module* llvmModule) {
  initializeLLVMTargets();
  
  std::string targetTriple = "nvptx64-nvidia-cuda";
  std::string error;
  
  const llvm::Target* target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!target) {
    return "Error: Failed to lookup target for PTX: " + error;
  }
  
  llvm::TargetOptions opt;
  auto relocationModel = llvm::Reloc::PIC_;
  std::unique_ptr<llvm::TargetMachine> targetMachine(
    target->createTargetMachine(targetTriple, "sm_50", "", opt, relocationModel));
  
  if (!targetMachine) {
    return "Error: Failed to create target machine for PTX";
  }
  
  llvmModule->setDataLayout(targetMachine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);
  
  llvm::SmallVector<char, 0> ptxBuffer;
  llvm::raw_svector_ostream dest(ptxBuffer);
  
  llvm::legacy::PassManager pass;
  if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, llvm::CodeGenFileType::AssemblyFile)) {
    return "Error: Target machine can't emit PTX";
  }
  
  pass.run(*llvmModule);
  
  return std::string(ptxBuffer.data(), ptxBuffer.size());
}

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
    // Register dialects and translation interfaces
    mlir::DialectRegistry registry;
    registry.insert<mlir::oven::OvenDialect>();
    mlir::registerAllDialects(registry);
    
    // Register LLVM translation interfaces
    mlir::registerAllToLLVMIRTranslations(registry);
    
    context = std::make_unique<mlir::MLIRContext>(registry);
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
  
  std::string to_ptx(const std::string& mlir_code) {
    // First convert to LLVM IR
    std::string llvm_ir = to_llvm_ir(mlir_code);
    if (llvm_ir.find("Error:") == 0) {
      return llvm_ir;
    }
    
    // Parse LLVM IR
    llvm::LLVMContext llvmContext;
    llvm::SMDiagnostic error;
    
    auto sourceBuffer = llvm::MemoryBuffer::getMemBuffer(llvm_ir);
    auto llvmModule = llvm::parseIR(sourceBuffer->getMemBufferRef(), error, llvmContext);
    if (!llvmModule) {
      return "Error: Failed to parse LLVM IR for PTX conversion: " + error.getMessage().str();
    }
    
    // Compile to PTX
    return compileToPTX(llvmModule.get());
  }
  
  std::string optimize_and_convert(const std::string& mlir_code, const std::string& format) {
    if (format == "mlir") {
      return optimize_mlir(mlir_code);
    } else if (format == "llvm" || format == "llvm-ir") {
      return to_llvm_ir(mlir_code);
    } else if (format == "ptx") {
      return to_ptx(mlir_code);
    } else {
      return "Error: Unsupported format '" + format + "'. Supported formats: mlir, llvm, ptx";
    }
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
             nb::arg("mlir_code"))
        .def("to_ptx", &OvenOptimizer::to_ptx,
             "Convert MLIR code to PTX assembly",
             nb::arg("mlir_code"))
        .def("optimize_and_convert", &OvenOptimizer::optimize_and_convert,
             "Optimize MLIR and convert to specified format (mlir, llvm, ptx)",
             nb::arg("mlir_code"), nb::arg("format"));
             
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
    
    m.def("to_ptx", [](const std::string& code) {
        OvenOptimizer opt;
        return opt.to_ptx(code);
    }, "Convert MLIR code to PTX assembly", nb::arg("code"));
    
    m.def("optimize_and_convert", [](const std::string& code, const std::string& format) {
        OvenOptimizer opt;
        return opt.optimize_and_convert(code, format);
    }, "Optimize MLIR and convert to specified format", nb::arg("code"), nb::arg("format"));
}