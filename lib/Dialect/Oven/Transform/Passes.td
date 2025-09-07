#ifndef OVENPASSES_TD_
#define OVENPASSES_TD_

include "mlir/Pass/PassBase.td"

def OptimizeOven : Pass<"optimize-oven"> {
  let summary = "Optimize the `oven` dialect.";

  let description = [{
    Optimize the `oven` dialect.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::oven::OvenDialect",
  ];
}

#endif  // OVENPASSES_TD_
