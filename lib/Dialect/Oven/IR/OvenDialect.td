#ifndef OVENDIALECT_TD_
#define OVENDIALECT_TD_

include "mlir/IR/DialectBase.td"

def Oven_Dialect : Dialect {
  let name = "oven";
  let summary = "A dialect for Operations";
  let description = [{
    Oven is a awesome dialect.
  }];

  let cppNamespace = "::mlir::oven";
}

#endif  // OVENDIALECT_TD_
