# Oven Compiler

Python-to-PTX GPU compiler using MLIR.

## Installation

```bash
pip install oven-compiler
```

## Quick Start

### Command Line

```bash
# Compile Python to PTX
oven-compiler --python kernel.py -o output.ptx
```

### Python API

```python
import oven_compiler

python_code = "def add(a, b): return a + b"
ptx_code = oven_compiler.compile_python_string_to_ptx(python_code)
```

## Requirements

- Python ≥ 3.12
- CUDA-capable GPU (for execution)

## License

MIT License
