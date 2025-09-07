#ifndef OVENPASSES_H_
#define OVENPASSES_H_

#include "lib/Dialect/Oven/Transform/OptimizeOven.h"

namespace mlir {
namespace oven {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Oven/Transform/Passes.h.inc"

} // namespace oven
} // namespace mlir

#endif // OVENPASSES_H_
