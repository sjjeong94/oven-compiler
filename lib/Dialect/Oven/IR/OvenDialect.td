#ifndef LIB_OVENDIALECT_TD_
#define LIB_OVENDIALECT_TD_

include "mlir/IR/DialectBase.td"

def Oven_Dialect : Dialect {
  let name = "oven";
  let summary = "A dialect for Operations";
  let description = [{
    Oven is a awesome dialect.
  }];

  let cppNamespace = "::mlir::oven";
}

#endif  // LIB_OVENDIALECT_TD_
