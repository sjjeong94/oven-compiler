#include "lib/Conversion/OvenToLLVM/OvenToLLVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

    RewritePatternSet patterns(context);
    patterns.add<ConvertLoad, ConvertStore, ConvertSmem>(context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace oven
} // namespace mlir
