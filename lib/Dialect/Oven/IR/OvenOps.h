#ifndef OVENOPS_H_
#define OVENOPS_H_

#include "mlir/IR/BuiltinOps.h"   // from @llvm-project
#include "mlir/IR/BuiltinTypes.h" // from @llvm-project
#include "mlir/IR/Dialect.h"      // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "lib/Dialect/Oven/IR/OvenDialect.h"

#define GET_OP_CLASSES
#include "lib/Dialect/Oven/IR/OvenOps.h.inc"

#endif // OVENOPS_H_
