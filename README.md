# oven-mlir

**Python-to-PTX GPU Kernel Compiler**

`oven-mlir` is a comprehensive compilation pipeline that transforms Python source code into optimized PTX assembly for NVIDIA GPUs. It combines the power of MLIR (Multi-Level Intermediate Representation) with an intuitive Python interface.

## Features

- 🐍 **Python → PTX Direct Compilation**: Compile Python functions directly to PTX assembly
- 🔧 **MLIR Optimization Pipeline**: Advanced optimization passes for GPU kernels  
- 🚀 **Command Line Interface**: Easy-to-use CLI for batch compilation
- 📁 **Intermediate File Support**: Save MLIR and LLVM IR for debugging
- 🎯 **GPU-Optimized Output**: Generate efficient PTX code for CUDA kernels

## Installation

```bash
pip install oven-mlir
```

### Dependencies

For Python-to-PTX compilation, you'll also need:

```bash
pip install oven-compiler
```

## Quick Start

### Python API

```python
import oven_mlir

# Compile Python function to PTX
python_code = """
def add_vectors(a, b):
    return a + b
"""

ptx_code = oven_mlir.compile_python_string_to_ptx(python_code)
print(ptx_code)
```

### Command Line Interface

```bash
# Compile Python file to PTX
oven-mlir --python my_kernel.py -o kernel.ptx

# Compile Python string to PTX
oven-mlir --python-string "def square(x): return x*x" -o square.ptx

# Save intermediate files for debugging
oven-mlir --python kernel.py --intermediate --intermediate-dir ./debug --verbose

# Compile MLIR to PTX (traditional usage)
oven-mlir input.mlir --format ptx -o output.ptx
```

## Compilation Pipeline

```
Python Source → MLIR → LLVM IR → PTX Assembly
```

1. **Python → MLIR**: Uses `oven-compiler` to convert Python AST to MLIR
2. **MLIR Optimization**: Applies GPU-specific optimization passes
3. **LLVM Translation**: Converts optimized MLIR to LLVM IR
4. **PTX Generation**: Produces NVIDIA PTX assembly code

## Examples

### Basic Usage

```python
import oven_mlir

# Simple arithmetic function
def multiply(x, y):
    return x * y

# Compile to PTX
compiler = oven_mlir.PythonToPTXCompiler()
ptx = compiler.compile_python_to_ptx("def multiply(x, y): return x * y")

# Generated PTX will contain optimized GPU kernel
```

### Advanced Usage with Intermediate Files

```python
import oven_mlir

python_code = """
def vector_add(a, b, c):
    c = a + b
    return c
"""

compiler = oven_mlir.PythonToPTXCompiler()
result = compiler.compile_with_intermediate_files(
    python_code, 
    output_dir="./debug"
)

print(f"MLIR: {result['mlir_code']}")
print(f"PTX: {result['ptx_code']}")
```

## API Reference

### Core Classes

- `PythonToPTXCompiler`: Main compiler class for Python → PTX compilation
- `OvenOptimizer`: MLIR optimization and compilation utilities
- `OvenCompiler`: Low-level MLIR compilation interface

### Functions

- `compile_python_string_to_ptx(code)`: Compile Python string to PTX
- `compile_python_file_to_ptx(file)`: Compile Python file to PTX  
- `optimize_string(mlir)`: Optimize MLIR code
- `to_ptx(mlir)`: Convert MLIR to PTX assembly

## Requirements

- Python 3.8+
- NVIDIA GPU (for PTX execution)
- CUDA Toolkit (recommended)

## Contributing

We welcome contributions! Please see our [GitHub repository](https://github.com/sjjeong94/oven) for more information.

## License

MIT License - see LICENSE file for details.

## Related Projects

- [oven-compiler](https://pypi.org/project/oven-compiler/): Python-to-MLIR frontend compiler
- [MLIR](https://mlir.llvm.org/): Multi-Level Intermediate Representation framework