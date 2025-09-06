#include "lib/Dialect/Oven/IR/OvenDialect.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "lib/Dialect/Oven/IR/OvenDialect.cpp.inc"
#include "lib/Dialect/Oven/IR/OvenOps.h"

#define GET_OP_CLASSES
#include "lib/Dialect/Oven/IR/OvenOps.cpp.inc"

namespace mlir {
namespace oven {

void OvenDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Oven/IR/OvenOps.cpp.inc"
      >();
}

} // namespace oven
} // namespace mlir
