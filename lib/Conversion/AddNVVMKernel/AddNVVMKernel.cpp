#include "lib/Conversion/AddNVVMKernel/AddNVVMKernel.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace oven {

#define GEN_PASS_DEF_ADDNVVMKERNEL
#include "lib/Conversion/AddNVVMKernel/AddNVVMKernel.h.inc"

struct AddNVVMKernelAttrPattern
    : public mlir::OpRewritePattern<mlir::LLVM::LLVMFuncOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::LLVM::LLVMFuncOp funcOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (funcOp->hasAttr("nvvm.kernel"))
      return mlir::failure();

    auto attr = rewriter.getBoolAttr(true);
    funcOp->setAttr("nvvm.kernel", attr);
    return mlir::success();
  }
};

struct AddNVVMKernel : impl::AddNVVMKernelBase<AddNVVMKernel> {
  using AddNVVMKernelBase::AddNVVMKernelBase;

  void runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<AddNVVMKernelAttrPattern>(&getContext());
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace oven
} // namespace mlir
