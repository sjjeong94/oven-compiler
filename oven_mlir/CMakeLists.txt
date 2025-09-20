cmake_minimum_required(VERSION 3.20.0)

# Create the nanobind module for oven-opt
nanobind_add_module(
  oven_opt_py
  NB_STATIC
  oven_opt_bindings.cpp
)

# Get all MLIR libraries
get_property(mlir_libs GLOBAL PROPERTY MLIR_ALL_LIBS)

# Link against MLIR libraries and oven libraries
target_link_libraries(oven_opt_py PRIVATE
  # Link all MLIR libraries to avoid missing symbols
  ${mlir_libs}
  
  # Oven specific libraries
  ${PROJECT_BINARY_DIR}/lib/Dialect/Oven/IR/libMLIROven.a
  ${PROJECT_BINARY_DIR}/lib/Dialect/Oven/Transform/libOvenPasses.a
  ${PROJECT_BINARY_DIR}/lib/Conversion/OvenToLLVM/libOvenToLLVM.a
  ${PROJECT_BINARY_DIR}/lib/Conversion/AddNVVMKernel/libAddNVVMKernel.a
)

# Add include directories
target_include_directories(oven_opt_py PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/lib
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

# Set installation directory
install(TARGETS oven_opt_py LIBRARY DESTINATION .)