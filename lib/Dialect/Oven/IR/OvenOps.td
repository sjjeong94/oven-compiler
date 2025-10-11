#ifndef OVENOPS_TD_
#define OVENOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "OvenDialect.td"

class Oven_UnaryOp<string mnemonic, list<Trait> traits = []> : Op<Oven_Dialect, mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins AnyType:$lhs);
  let results = (outs AnyType:$output);
  let assemblyFormat = "$lhs attr-dict `:` type($output)";
}

def Oven_SigmoidOp : Oven_UnaryOp<"sigmoid"> {
  let summary = "Sigmoid";
}

def Oven_LoadOp : Op<Oven_Dialect, "load"> {
  let summary = "Load from memory";
  let arguments = (ins AnyType:$ptr, I32:$offset);
  let results = (outs F32:$value);
  let assemblyFormat = "$ptr `,` $offset attr-dict `:` `(` type($ptr) `,` type($offset) `)` `->` type($value)";
}

def Oven_StoreOp : Op<Oven_Dialect, "store"> {
  let summary = "Store to memory";
  let arguments = (ins F32:$value, AnyType:$ptr, I32:$offset);
  let results = (outs);
  let assemblyFormat = "$value `,` $ptr `,` $offset attr-dict `:` `(` type($value) `,` type($ptr) `,` type($offset) `)`";
}

def Oven_VloadOp : Op<Oven_Dialect, "vload"> {
  let summary = "Vectorized load";
  let arguments = (ins AnyType:$ptr, I32:$offset);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$ptr `,` $offset attr-dict `:` `(` type($ptr) `,` type($offset) `)` `->` type($value)";
}

def Oven_VstoreOp : Op<Oven_Dialect, "vstore"> {
  let summary = "Vectorized store";
  let arguments = (ins AnyType:$value, AnyType:$ptr, I32:$offset);
  let results = (outs);
  let assemblyFormat = "$value `,` $ptr `,` $offset attr-dict `:` `(` type($value) `,` type($ptr) `,` type($offset) `)`";
}

def Oven_SmemOp : Op<Oven_Dialect, "smem"> {
  let summary = "Declare shared memory";
  let arguments = (ins);
  let results = (outs AnyType:$ptr);
  let assemblyFormat = "attr-dict `:` type($ptr)";
}

#endif  // OVENOPS_TD_
