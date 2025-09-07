#ifndef ADDNVVMKERNEL_TD_
#define ADDNVVMKERNEL_TD_

include "mlir/Pass/PassBase.td"

def AddNVVMKernel : Pass<"add-nvvm-kernel"> {
  let summary = "Add NVVM kernel attribute to functions.";

  let description = [{
    This pass adds the NVVM kernel attribute to functions.
  }];
  let dependentDialects = [
  ];
}

#endif  // ADDNVVMKERNEL_TD_
