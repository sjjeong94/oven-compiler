// RUN: oven-opt %s --oven-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @function
func.func @function(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  // CHECK: %[[CONST_ONE:.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<4xf32>) : vector<4xf32>
  // CHECK: %[[CONST_LOG2E:.*]] = llvm.mlir.constant(dense<1.44269502> : vector<4xf32>) : vector<4xf32>
  // CHECK: %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: %[[NTID:.*]] = nvvm.read.ptx.sreg.ntid.x : i32
  %0 = nvvm.read.ptx.sreg.ntid.x : i32
  // CHECK: %[[CTAID:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
  %1 = nvvm.read.ptx.sreg.ctaid.x : i32
  // CHECK: %[[TID:.*]] = nvvm.read.ptx.sreg.tid.x : i32
  %2 = nvvm.read.ptx.sreg.tid.x : i32
  // CHECK: %[[MUL1:.*]] = llvm.mul %[[NTID]], %[[CTAID]] : i32
  %3 = arith.muli %0, %1 : i32
  // CHECK: %[[ADD:.*]] = llvm.add %[[TID]], %[[MUL1]] : i32
  %4 = arith.addi %2, %3 : i32
  %c4_i32 = arith.constant 4 : i32
  // CHECK: %[[MUL2:.*]] = llvm.mul %[[ADD]], %[[C4]] : i32
  %5 = arith.muli %4, %c4_i32 : i32
  // CHECK: %[[GEP_LOAD:.*]] = llvm.getelementptr %arg0[%[[MUL2]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
  // CHECK: %[[VLOAD:.*]] = llvm.load %[[GEP_LOAD]] : !llvm.ptr -> vector<4xf32>
  %6 = oven.vload %arg0, %5, 4 : (!llvm.ptr, i32) -> vector<4xf32>
  // CHECK: %[[NEG:.*]] = llvm.fneg %[[VLOAD]] : vector<4xf32>
  // CHECK: %[[MUL_LOG2E:.*]] = llvm.fmul %[[NEG]], %[[CONST_LOG2E]] : vector<4xf32>
  // CHECK: %[[EXP:.*]] = llvm.intr.exp2(%[[MUL_LOG2E]]) : (vector<4xf32>) -> vector<4xf32>
  // CHECK: %[[ADD_ONE:.*]] = llvm.fadd %[[EXP]], %[[CONST_ONE]] : vector<4xf32>
  // CHECK: %[[SIGMOID:.*]] = llvm.fdiv %[[CONST_ONE]], %[[ADD_ONE]] : vector<4xf32>
  %7 = oven.sigmoid %6 : vector<4xf32> -> vector<4xf32>
  // CHECK: %[[GEP_STORE:.*]] = llvm.getelementptr %arg1[%[[MUL2]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
  // CHECK: llvm.store %[[SIGMOID]], %[[GEP_STORE]] : vector<4xf32>, !llvm.ptr
  oven.vstore %7, %arg1, %5, 4 : (vector<4xf32>, !llvm.ptr, i32)
  // CHECK: llvm.return
  return
}