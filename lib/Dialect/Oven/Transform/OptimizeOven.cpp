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
    // exp(x) = 2^(x * log2(e)) where log2(e) ~= 1.442695
    Value constant = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF32FloatAttr(1.442695));
    Value mul = rewriter.create<arith::MulFOp>(op.getLoc(), op.getType(),
                                               op.getOperand(), constant);
    Value exp2 = rewriter.create<math::Exp2Op>(op.getLoc(), op.getType(), mul);
    rewriter.replaceOp(op, exp2);
    return success();
  }
};

struct DecomposeSigmoidPattern : public OpRewritePattern<oven::SigmoidOp> {
  DecomposeSigmoidPattern(mlir::MLIRContext *context)
      : OpRewritePattern<oven::SigmoidOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(oven::SigmoidOp op,
                                PatternRewriter &rewriter) const override {
    // sigmoid(x) = 1 / (1 + exp(-x))
    Value negX = rewriter.create<arith::NegFOp>(op.getLoc(), op.getOperand());
    Value expNegX = rewriter.create<math::ExpOp>(op.getLoc(), negX);
    Value one = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF32FloatAttr(1.0));
    Value denom = rewriter.create<arith::AddFOp>(op.getLoc(), one, expNegX);
    Value result = rewriter.create<arith::DivFOp>(op.getLoc(), one, denom);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct OptimizeOven : impl::OptimizeOvenBase<OptimizeOven> {
  using OptimizeOvenBase::OptimizeOvenBase;

  void runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeExpPattern>(&getContext());
    patterns.add<DecomposeSigmoidPattern>(&getContext());
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace oven
} // namespace mlir
