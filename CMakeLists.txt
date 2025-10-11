cmake_minimum_required(VERSION 3.20.0)

project(mlir-practice LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_DEPS ON)

# Set LLVM and MLIR directories (LLVM is in parent directory)
set(LLVM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llvm-project/build/lib/cmake/llvm")
set(MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llvm-project/build/lib/cmake/mlir")
find_package(MLIR REQUIRED CONFIG)

# Find Python and nanobind for Python bindings
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Try to find nanobind via pip
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE nanobind_RESULT
)

if(nanobind_RESULT EQUAL 0)
    set(CMAKE_PREFIX_PATH "${nanobind_DIR}" ${CMAKE_PREFIX_PATH})
    find_package(nanobind CONFIG REQUIRED)
    set(NANOBIND_FOUND TRUE)
else()
    message(WARNING "nanobind not found via pip, Python bindings will be disabled")
    set(NANOBIND_FOUND FALSE)
endif()

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using Python: ${Python_EXECUTABLE}")

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

include(AddLLVM)
include(TableGen)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/llvm-project/llvm/include)
include_directories(${PROJECT_SOURCE_DIR}/llvm-project/mlir/include)

add_subdirectory(lib)
add_subdirectory(tools)

# Only add Python bindings if nanobind is available
if(NANOBIND_FOUND)
    message(STATUS "Adding Python bindings")
    add_subdirectory(oven_compiler)
else()
    message(STATUS "Skipping Python bindings (nanobind not found)")
endif()
