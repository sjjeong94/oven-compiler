#ifndef LIB_OVENOPS_TD_
#define LIB_OVENOPS_TD_

include "mlir/IR/OpBase.td"

include "OvenDialect.td"

class Oven_UnaryOp<string mnemonic> : Op<Oven_Dialect, mnemonic> {
  let arguments = (ins F32:$lhs);
  let results = (outs F32:$output);
  let assemblyFormat = "$lhs attr-dict `:` type($lhs) `->` type($output)";
}

class Oven_BinOp<string mnemonic> : Op<Oven_Dialect, mnemonic> {
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)";
}

def Oven_AddOp : Oven_BinOp<"add"> {
  let summary = "Addition operation";
}

def Oven_SigmoidOp : Oven_UnaryOp<"sigmoid"> {
  let summary = "Sigmoid";
}


class Oven_IdxOp<string name, string desc> : Op<Oven_Dialect, name> {
  let summary = desc;
  let arguments = (ins);
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Oven_BlockIdxXOp    : Oven_IdxOp<"block_idx.x",    "Get blockIdx.x">;
def Oven_BlockIdxYOp    : Oven_IdxOp<"block_idx.y",    "Get blockIdx.y">;
def Oven_BlockIdxZOp    : Oven_IdxOp<"block_idx.z",    "Get blockIdx.z">;
def Oven_ThreadIdxXOp   : Oven_IdxOp<"thread_idx.x",   "Get threadIdx.x">;
def Oven_ThreadIdxYOp   : Oven_IdxOp<"thread_idx.y",   "Get threadIdx.y">;
def Oven_ThreadIdxZOp   : Oven_IdxOp<"thread_idx.z",   "Get threadIdx.z">;

def Oven_LoadOp : Op<Oven_Dialect, "load"> {
  let summary = "Load from memory";
  let arguments = (ins RankedTensorOf<[F32]>:$ptr, I32:$offset);
  let results = (outs F32:$value);
  let assemblyFormat = "$ptr `,` $offset attr-dict `:` `(` type($ptr) `,` type($offset) `)` `->` type($value)";
}

def Oven_StoreOp : Op<Oven_Dialect, "store"> {
  let summary = "Store to memory";
  let arguments = (ins F32:$value, RankedTensorOf<[F32]>:$ptr, I32:$offset);
  let results = (outs);
  let assemblyFormat = "$value `,` $ptr `,` $offset attr-dict `:` `(` type($value) `,` type($ptr) `,` type($offset) `)`";
}

#endif  // LIB_OVENOPS_TD_
