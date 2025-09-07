#include "lib/Dialect/Oven/Transform/OptimizeOven.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "lib/Dialect/Oven/IR/OvenDialect.h"
#include "lib/Dialect/Oven/IR/OvenOps.h"

namespace mlir {
namespace oven {

#define GEN_PASS_DEF_OPTIMIZEOVEN
#include "lib/Dialect/Oven/Transform/Passes.h.inc"

struct DecomposeExpPattern : public OpRewritePattern<math::ExpOp> {
  DecomposeExpPattern(mlir::MLIRContext *context)
      : OpRewritePattern<math::ExpOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(math::ExpOp op,
                                PatternRewriter &rewriter) const override {
    arith::ConstantOp constant = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF32FloatAttr(1.442695));
    arith::MulFOp mul = rewriter.create<arith::MulFOp>(
        op.getLoc(), op.getType(), op.getOperand(), constant.getResult());

    math::Exp2Op exp2 = rewriter.create<math::Exp2Op>(op.getLoc(), op.getType(),
                                               mul.getResult());
    rewriter.replaceOp(op, exp2);
    return success();
  }
};

struct OptimizeOven : impl::OptimizeOvenBase<OptimizeOven> {
  using OptimizeOvenBase::OptimizeOvenBase;

  void runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeExpPattern>(&getContext());
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace oven
} // namespace mlir
