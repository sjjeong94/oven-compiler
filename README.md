# Oven Compiler

> **A Python-to-PTX GPU Kernel Compiler**

Transform Python functions into optimized NVIDIA PTX assembly using MLIR infrastructure.

## Overview

Oven Compiler provides a complete compilation pipeline from Python source code to GPU-ready PTX assembly. Built on MLIR (Multi-Level Intermediate Representation), it offers powerful optimization passes specifically designed for GPU kernels.

**Pipeline**: Python → MLIR → LLVM IR → PTX Assembly

## Installation

```bash
pip install oven-compiler
```

## Quick Start

### Command Line

```bash
# Compile Python to PTX
oven-compiler --python kernel.py -o output.ptx

# From Python string
oven-compiler --python-string "def add(a, b): return a + b" -o add.ptx

# With debug information
oven-compiler --python kernel.py --intermediate --verbose
```

### Python API

```python
import oven_compiler

# Basic compilation
python_code = "def multiply(x, y): return x * y"
ptx_code = oven_compiler.compile_python_string_to_ptx(python_code)

# Advanced usage with intermediate files
compiler = oven_compiler.PythonToPTXCompiler()
result = compiler.compile_with_intermediate_files(
    python_code, 
    output_dir="./debug"
)
```

## Features

- **Direct Python → PTX**: Compile Python functions to GPU assembly
- **MLIR Optimization**: Advanced GPU-specific optimization passes
- **Debugging Support**: Save intermediate MLIR and LLVM IR files
- **CLI & API**: Both command-line and programmatic interfaces
- **GPU Optimized**: Efficient PTX generation for CUDA kernels

## API Reference

### Core Classes
- `PythonToPTXCompiler` - Main compiler for Python → PTX
- `OvenOptimizer` - MLIR optimization utilities
- `OvenCompiler` - Low-level MLIR interface

### Key Functions
- `compile_python_string_to_ptx(code)` - Compile Python string
- `compile_python_file_to_ptx(file)` - Compile Python file
- `optimize_string(mlir)` - Optimize MLIR code
- `to_ptx(mlir)` - Convert MLIR to PTX

## Requirements

- Python 3.8+
- NVIDIA GPU (for execution)
- CUDA Toolkit (recommended)

## License

MIT License
