#ifndef OPTIMIZEOVEN_H_
#define OPTIMIZEOVEN_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace oven {

#define GEN_PASS_DECL_OPTIMIZEOVEN
#include "lib/Dialect/Oven/Transform/Passes.h.inc"

} // namespace oven
} // namespace mlir

#endif // OPTIMIZEOVEN_H_
