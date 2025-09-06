#ifndef LIB_OVENOPS_TD_
#define LIB_OVENOPS_TD_

include "mlir/IR/OpBase.td"

include "OvenDialect.td"

class Oven_BinOp<string mnemonic> : Op<Oven_Dialect, mnemonic> {
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)";
}

def Oven_AddOp : Oven_BinOp<"add"> {
  let summary = "Addition operation";
}

#endif  // LIB_OVENOPS_TD_
