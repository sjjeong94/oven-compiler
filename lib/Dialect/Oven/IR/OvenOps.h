#ifndef LIB_OVENOPS_H_
#define LIB_OVENOPS_H_

#include "mlir/IR/BuiltinOps.h"   // from @llvm-project
#include "mlir/IR/BuiltinTypes.h" // from @llvm-project
#include "mlir/IR/Dialect.h"      // from @llvm-project

#include "lib/Dialect/Oven/IR/OvenDialect.h"

#define GET_OP_CLASSES
#include "lib/Dialect/Oven/IR/OvenOps.h.inc"

#endif // LIB_OVENOPS_H_
