add_mlir_library(OvenPasses
    OptimizeOven.cpp

    ${PROJECT_SOURCE_DIR}/lib/Dialect/Oven/Transform/
    ADDITIONAL_HEADER_DIRS

    DEPENDS
    MLIROven
    MLIROvenPasses

    LINK_LIBS PUBLIC
)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIROvenPasses)
add_mlir_doc(Passes OvenPasses ./ -gen-pass-doc)