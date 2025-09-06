get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set (LIBS
    ${dialect_libs}
    MLIROven
    MLIROptLib
    OvenToLLVM
)

add_llvm_executable(oven-opt oven-opt.cpp)

llvm_update_compile_flags(oven-opt)
target_link_libraries(oven-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(oven-opt)
