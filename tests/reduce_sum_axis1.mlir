// RUN: oven-opt %s --oven-to-llvm | FileCheck %s

func.func @reduceSumAxis1(%input: !llvm.ptr, %output: !llvm.ptr, %rows: i32, %cols: i32) {
  // CHECK-LABEL: llvm.func @reduceSumAxis1
  
  // Get shared memory for reduction
  // CHECK: llvm.mlir.addressof @smem0
  %sharedData = oven.smem : !llvm.ptr<3>
  
  // Get thread and block indices
  // CHECK: nvvm.read.ptx.sreg.ctaid.x : i32
  %row = nvvm.read.ptx.sreg.ctaid.x : i32
  // CHECK: nvvm.read.ptx.sreg.tid.x : i32
  %tid = nvvm.read.ptx.sreg.tid.x : i32
  // CHECK: nvvm.read.ptx.sreg.ntid.x : i32
  %blockDim = nvvm.read.ptx.sreg.ntid.x : i32
  
  // Constants
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %zero_f32 = arith.constant 0.0 : f32
  
  // Calculate input index: row * cols + tid
  // CHECK: llvm.mul
  %row_offset = arith.muli %row, %cols : i32
  // CHECK: llvm.add
  %input_idx = arith.addi %row_offset, %tid : i32
  
  // Load element with bounds check using scf.if
  // CHECK: llvm.icmp "slt"
  %tid_in_bounds = arith.cmpi slt, %tid, %cols : i32
  
  // CHECK: llvm.cond_br
  %loaded_value = scf.if %tid_in_bounds -> f32 {
    // CHECK: llvm.getelementptr
    // CHECK: llvm.load
    %val = oven.load %input, %input_idx : (!llvm.ptr, i32) -> f32
    scf.yield %val : f32
  } else {
    scf.yield %zero_f32 : f32
  }
  
  // Store to shared memory
  // CHECK: llvm.getelementptr
  // CHECK: llvm.store
  oven.store %loaded_value, %sharedData, %tid : (f32, !llvm.ptr<3>, i32)
  
  // Synchronize threads
  // CHECK: nvvm.barrier0
  nvvm.barrier0
  
  // Perform reduction in shared memory
  // Start with s = blockDim / 2
  %initial_s = arith.divui %blockDim, %c2 : i32
  
  // Loop for reduction: for (s = blockDim/2; s > 0; s >>= 1)
  %final_s = scf.while (%s = %initial_s) : (i32) -> i32 {
    // Loop condition: s > 0
    // CHECK: llvm.icmp "sgt"
    %s_gt_zero = arith.cmpi sgt, %s, %c0 : i32
    scf.condition(%s_gt_zero) %s : i32
  } do {
  ^bb0(%s: i32):
    // Check if tid < s
    // CHECK: llvm.icmp "slt"
    %tid_lt_s = arith.cmpi slt, %tid, %s : i32
    
    // If tid < s, perform reduction
    // CHECK: llvm.cond_br
    scf.if %tid_lt_s {
      // Calculate tid + s
      // CHECK: llvm.add
      %tid_plus_s = arith.addi %tid, %s : i32
      
      // Load sharedData[tid]
      // CHECK: llvm.getelementptr
      // CHECK: llvm.load
      %left_val = oven.load %sharedData, %tid : (!llvm.ptr<3>, i32) -> f32
      
      // Load sharedData[tid + s]
      // CHECK: llvm.getelementptr
      // CHECK: llvm.load
      %right_val = oven.load %sharedData, %tid_plus_s : (!llvm.ptr<3>, i32) -> f32
      
      // Add values
      // CHECK: llvm.fadd
      %sum = arith.addf %left_val, %right_val : f32
      
      // Store back to sharedData[tid]
      // CHECK: llvm.getelementptr
      // CHECK: llvm.store
      oven.store %sum, %sharedData, %tid : (f32, !llvm.ptr<3>, i32)
    }
    
    // Synchronize threads
    // CHECK: nvvm.barrier0
    nvvm.barrier0
    
    // Update s = s >> 1 (equivalent to s = s / 2)
    %next_s = arith.divui %s, %c2 : i32
    scf.yield %next_s : i32
  }
  
  // Write result if tid == 0
  // CHECK: llvm.icmp "eq"
  %is_thread_zero = arith.cmpi eq, %tid, %c0 : i32
  
  // CHECK: llvm.cond_br
  scf.if %is_thread_zero {
    // Load final result from sharedData[0]
    // CHECK: llvm.load
    %result = oven.load %sharedData, %c0 : (!llvm.ptr<3>, i32) -> f32
    
    // Store to output[row]
    // CHECK: llvm.getelementptr
    // CHECK: llvm.store
    oven.store %result, %output, %row : (f32, !llvm.ptr, i32)
  }
  
  return
}