#ifndef OVENTOLLVM_H_
#define OVENTOLLVM_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace oven {

#define GEN_PASS_DECL
#include "lib/Conversion/OvenToLLVM/OvenToLLVM.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Conversion/OvenToLLVM/OvenToLLVM.h.inc"

} // namespace oven
} // namespace mlir

#endif // OVENTOLLVM_H_