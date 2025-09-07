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
    llvm::outs() << "decomposing exp...\n";
    llvm::outs() << op << "\n";
    return failure();
  }
};

struct OptimizeOven : impl::OptimizeOvenBase<OptimizeOven> {
  using OptimizeOvenBase::OptimizeOvenBase;

  void runOnOperation() {
    llvm::outs() << "optimizing oven...\n";
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeExpPattern>(&getContext());
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace oven
} // namespace mlir
