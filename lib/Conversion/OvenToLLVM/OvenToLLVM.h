#ifndef OVENTOLLVM_H_
#define OVENTOLLVM_H_

#include "mlir/Pass/Pass.h" // from @llvm-project

// Extra includes needed for dependent dialects
#include "mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h" // from @llvm-project

namespace mlir {
namespace oven {

#define GEN_PASS_DECL
#include "lib/Conversion/OvenToLLVM/OvenToLLVM.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Conversion/OvenToLLVM/OvenToLLVM.h.inc"

} // namespace oven
} // namespace mlir

#endif // OVENTOLLVM_H_