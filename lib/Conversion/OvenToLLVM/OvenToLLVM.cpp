#include "lib/Conversion/OvenToLLVM/OvenToLLVM.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

struct OvenToLLVM : impl::OvenToLLVMBase<OvenToLLVM> {
  using OvenToLLVMBase::OvenToLLVMBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<oven::OvenDialect>();

    RewritePatternSet patterns(context);
    patterns.add<ConvertLoad, ConvertStore>(context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace oven
} // namespace mlir
