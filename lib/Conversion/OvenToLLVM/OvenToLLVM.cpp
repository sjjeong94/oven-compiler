#include "lib/Conversion/OvenToLLVM/OvenToLLVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"

#include "lib/Dialect/Oven/IR/OvenDialect.h"
#include "lib/Dialect/Oven/IR/OvenOps.h"

namespace mlir {
namespace oven {

#define GEN_PASS_DEF_OVENTOLLVM
#include "lib/Conversion/OvenToLLVM/OvenToLLVM.h.inc"

struct ConvertLoad : public OpConversionPattern<oven::LoadOp> {
  ConvertLoad(mlir::MLIRContext *context)
      : OpConversionPattern<oven::LoadOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(oven::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::GEPOp gep = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), op.getPtr().getType(), rewriter.getF32Type(), op.getPtr(),
        op.getOffset());
    LLVM::LoadOp llop = rewriter.create<LLVM::LoadOp>(
        op.getLoc(), op.getResult().getType(), gep.getResult());
    rewriter.replaceOp(op, llop);
    return success();
  }
};

struct ConvertStore : public OpConversionPattern<oven::StoreOp> {
  ConvertStore(mlir::MLIRContext *context)
      : OpConversionPattern<oven::StoreOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(oven::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::GEPOp gep = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), op.getPtr().getType(), rewriter.getF32Type(), op.getPtr(),
        op.getOffset());
    LLVM::StoreOp llop = rewriter.create<LLVM::StoreOp>(
        op.getLoc(), op.getValue(), gep.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertSmem : public OpConversionPattern<oven::SmemOp> {
  ConvertSmem(mlir::MLIRContext *context)
      : OpConversionPattern<oven::SmemOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  mutable int smemCount = 0;

  LogicalResult
  matchAndRewrite(oven::SmemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string symbolName = "smem" + std::to_string(smemCount++);
    auto symbol = SymbolRefAttr::get(rewriter.getContext(), symbolName);
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto i8Type = mlir::IntegerType::get(rewriter.getContext(), 8);
    auto arrayType = mlir::LLVM::LLVMArrayType::get(i8Type, 0);
    OpBuilder builder(moduleOp.getBody(), moduleOp.getBody()->begin());
    builder.create<LLVM::GlobalOp>(moduleOp.getLoc(), arrayType, false,
                                   LLVM::Linkage::External, symbolName,
                                   Attribute(), 16, 3);
    auto addressofOp =
        rewriter.create<LLVM::AddressOfOp>(op.getLoc(), ptrType, symbol);
    rewriter.replaceOp(op, addressofOp);
    return success();
  }
};

struct ConvertVload : public OpConversionPattern<oven::VloadOp> {
  ConvertVload(mlir::MLIRContext *context)
      : OpConversionPattern<oven::VloadOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(oven::VloadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::GEPOp gep = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), op.getPtr().getType(), rewriter.getF32Type(), op.getPtr(),
        op.getOffset());
    LLVM::LoadOp llop = rewriter.create<LLVM::LoadOp>(
        op.getLoc(), op.getResult().getType(), gep.getResult());
    rewriter.replaceOp(op, llop);
    return success();
  }
};

struct ConvertVstore : public OpConversionPattern<oven::VstoreOp> {
  ConvertVstore(mlir::MLIRContext *context)
      : OpConversionPattern<oven::VstoreOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(oven::VstoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::GEPOp gep = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), op.getPtr().getType(), rewriter.getF32Type(), op.getPtr(),
        op.getOffset());
    LLVM::StoreOp llop = rewriter.create<LLVM::StoreOp>(
        op.getLoc(), op.getValue(), gep.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertMathCos : public OpConversionPattern<math::CosOp> {
  ConvertMathCos(mlir::MLIRContext *context)
      : OpConversionPattern<math::CosOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::CosOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    
    // Check if __nv_cosf function is already declared
    auto cosfFunc = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("__nv_cosf");
    if (!cosfFunc) {
      // Create function declaration for __nv_cosf
      OpBuilder builder(moduleOp.getBody(), moduleOp.getBody()->begin());
      auto f32Type = rewriter.getF32Type();
      auto funcType = LLVM::LLVMFunctionType::get(f32Type, {f32Type});
      cosfFunc = builder.create<LLVM::LLVMFuncOp>(
          moduleOp.getLoc(), "__nv_cosf", funcType);
    }
    
    // Create function call
    auto callOp = rewriter.create<LLVM::CallOp>(
        op.getLoc(), cosfFunc, op.getOperand());
    
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

struct ConvertMathSin : public OpConversionPattern<math::SinOp> {
  ConvertMathSin(mlir::MLIRContext *context)
      : OpConversionPattern<math::SinOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::SinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    
    // Check if __nv_sinf function is already declared
    auto sinfFunc = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("__nv_sinf");
    if (!sinfFunc) {
      // Create function declaration for __nv_sinf
      OpBuilder builder(moduleOp.getBody(), moduleOp.getBody()->begin());
      auto f32Type = rewriter.getF32Type();
      auto funcType = LLVM::LLVMFunctionType::get(f32Type, {f32Type});
      sinfFunc = builder.create<LLVM::LLVMFuncOp>(
          moduleOp.getLoc(), "__nv_sinf", funcType);
    }
    
    // Create function call
    auto callOp = rewriter.create<LLVM::CallOp>(
        op.getLoc(), sinfFunc, op.getOperand());
    
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};


struct ConvertMathTan : public OpConversionPattern<math::TanOp> {
  ConvertMathTan(mlir::MLIRContext *context)
      : OpConversionPattern<math::TanOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::TanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    
    // Check if __nv_tanf function is already declared
    auto tanfFunc = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("__nv_tanf");
    if (!tanfFunc) {
      // Create function declaration for __nv_tanf
      OpBuilder builder(moduleOp.getBody(), moduleOp.getBody()->begin());
      auto f32Type = rewriter.getF32Type();
      auto funcType = LLVM::LLVMFunctionType::get(f32Type, {f32Type});
      tanfFunc = builder.create<LLVM::LLVMFuncOp>(
          moduleOp.getLoc(), "__nv_tanf", funcType);
    }
    
    // Create function call
    auto callOp = rewriter.create<LLVM::CallOp>(
        op.getLoc(), tanfFunc, op.getOperand());
    
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

struct ConvertMathLog : public OpConversionPattern<math::LogOp> {
  ConvertMathLog(mlir::MLIRContext *context)
      : OpConversionPattern<math::LogOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::LogOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    
    // Check if __nv_logf function is already declared
    auto logfFunc = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("__nv_logf");
    if (!logfFunc) {
      // Create function declaration for __nv_logf
      OpBuilder builder(moduleOp.getBody(), moduleOp.getBody()->begin());
      auto f32Type = rewriter.getF32Type();
      auto funcType = LLVM::LLVMFunctionType::get(f32Type, {f32Type});
      logfFunc = builder.create<LLVM::LLVMFuncOp>(
          moduleOp.getLoc(), "__nv_logf", funcType);
    }
    
    // Create function call
    auto callOp = rewriter.create<LLVM::CallOp>(
        op.getLoc(), logfFunc, op.getOperand());
    
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

struct ConvertMathLog2 : public OpConversionPattern<math::Log2Op> {
  ConvertMathLog2(mlir::MLIRContext *context)
      : OpConversionPattern<math::Log2Op>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::Log2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    
    // Check if __nv_log2f function is already declared
    auto log2fFunc = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("__nv_log2f");
    if (!log2fFunc) {
      // Create function declaration for __nv_log2f
      OpBuilder builder(moduleOp.getBody(), moduleOp.getBody()->begin());
      auto f32Type = rewriter.getF32Type();
      auto funcType = LLVM::LLVMFunctionType::get(f32Type, {f32Type});
      log2fFunc = builder.create<LLVM::LLVMFuncOp>(
          moduleOp.getLoc(), "__nv_log2f", funcType);
    }
    
    // Create function call
    auto callOp = rewriter.create<LLVM::CallOp>(
        op.getLoc(), log2fFunc, op.getOperand());
    
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

struct OvenToLLVM : impl::OvenToLLVMBase<OvenToLLVM> {
  using OvenToLLVMBase::OvenToLLVMBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<oven::OvenDialect>();
    target.addIllegalOp<math::CosOp>();
    target.addIllegalOp<math::SinOp>();
    target.addIllegalOp<math::TanOp>();
    target.addIllegalOp<math::LogOp>();
    target.addIllegalOp<math::Log2Op>();
    
    RewritePatternSet patterns(context);
    patterns.add<ConvertLoad, ConvertStore, ConvertSmem>(context);
    patterns.add<ConvertVload, ConvertVstore>(context);
    patterns.add<ConvertMathCos, ConvertMathSin>(context);
    patterns.add<ConvertMathTan, ConvertMathLog, ConvertMathLog2>(context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace oven
} // namespace mlir
