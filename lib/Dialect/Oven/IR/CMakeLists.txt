
# Dialect TableGen
set(LLVM_TARGET_DEFINITIONS OvenDialect.td)
mlir_tablegen(OvenDialect.h.inc -gen-dialect-decls -dialect=oven)
mlir_tablegen(OvenDialect.cpp.inc -gen-dialect-defs -dialect=oven)

# Ops TableGen
set(LLVM_TARGET_DEFINITIONS OvenOps.td)
mlir_tablegen(OvenOps.h.inc -gen-op-decls)
mlir_tablegen(OvenOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(MLIROvenOpsIncGen)
add_dependencies(mlir-headers MLIROvenOpsIncGen)

add_mlir_doc(OvenDialect OvenDialect Oven/ -gen-dialect-doc)

add_mlir_dialect_library(MLIROven
    OvenDialect.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/Dialect/Oven/IR

    LINK_LIBS PUBLIC
)
