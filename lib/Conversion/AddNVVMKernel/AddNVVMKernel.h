#ifndef ADDNVVMKERNEL_H_
#define ADDNVVMKERNEL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace oven {

#define GEN_PASS_DECL
#include "lib/Conversion/AddNVVMKernel/AddNVVMKernel.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Conversion/AddNVVMKernel/AddNVVMKernel.h.inc"

} // namespace oven
} // namespace mlir

#endif // ADDNVVMKERNEL_H_