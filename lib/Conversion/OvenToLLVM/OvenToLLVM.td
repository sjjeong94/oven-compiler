#ifndef OVENTOLLVM_TD_
#define OVENTOLLVM_TD_

include "mlir/Pass/PassBase.td"

def OvenToLLVM : Pass<"oven-to-llvm"> {
  let summary = "Lower `oven` to LLVM IR.";

  let description = [{
    This pass lowers the `oven` dialect to LLVM IR.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::oven::OvenDialect",
  ];
}

#endif  // OVENTOLLVM_TD_
