# Development Scripts

This directory contains utility scripts for oven-mlir development with GPU compute capability support.

## Scripts Overview

### 🚀 `build_complete.sh` *(NEW)*
Complete build script for oven-mlir with GPU compute capability support.

```bash
./scripts/build_complete.sh [options]
```

**Options:**
- `-c, --clean` - Clean all build directories before building
- `-n, --native-only` - Build only native modules (skip wheel)
- `-w, --wheel-only` - Build only wheel (skip native build)
- `-t, --skip-tests` - Skip running tests
- `-v, --verbose` - Enable verbose output
- `--skip-deps` - Skip dependency checking
- `--platform TAG` - Force platform tag (e.g., manylinux2014_x86_64)

**Examples:**
```bash
./scripts/build_complete.sh                    # Complete build (native + wheel + tests)
./scripts/build_complete.sh -c -v              # Clean build with verbose output
./scripts/build_complete.sh -n                 # Build only native modules
./scripts/build_complete.sh -w                 # Build only wheel
```

**Build Process:**
1. Check dependencies (LLVM, MLIR, CMake)
2. Build native C++ modules with GPU support
3. Build Python wheel with manylinux compatibility
4. Run comprehensive tests including GPU functionality
5. Validate PyPI compatibility

### ⚡ `quick_build.sh` *(UPDATED)*
Quick build script for development with GPU support validation.

```bash
./scripts/quick_build.sh
```

**What it does:**
- Builds native modules if missing
- Creates Python wheel with GPU compute capability support
- Validates PyPI compatibility
- Tests GPU functionality (compute capability, target checking)

### 🔧 `setup_venv.sh`
Sets up a complete development environment with virtual environment and all dependencies.

```bash
./scripts/setup_venv.sh
```

**What it does:**
- Creates Python virtual environment
- Installs build dependencies (nanobind, cmake, ninja, etc.)
- Installs development dependencies (pytest, black, isort, flake8)
- Installs oven-mlir in development mode

### 📦 `build_wheel.sh`
Enhanced wheel build script with platform detection and upload options.

```bash
./scripts/build_wheel.sh [options]
```

**Options:**
- `-c, --clean` - Clean build directories before building
- `-t, --test-upload` - Upload to Test PyPI after building
- `-p, --pypi-upload` - Upload to PyPI after building
- `-s, --skip-tests` - Skip running tests before building
- `-v, --verbose` - Enable verbose output
- `--platform PLATFORM` - Force platform tag
- `--source` - Also build source distribution

### 🧪 `run_tests.sh`
Runs the test suite with various options.

```bash
./scripts/run_tests.sh [options]
```

**Options:**
- `--coverage` - Run tests with coverage report
- `--verbose` - Extra verbose output
- `--quiet` - Minimal output
- `--specific PATTERN` - Run only tests matching pattern
- `--help` - Show help

**Examples:**
```bash
./scripts/run_tests.sh                    # Run all tests
./scripts/run_tests.sh --coverage         # Run with coverage
./scripts/run_tests.sh --specific import  # Run only import tests
```

### 🧹 `clean.sh`
Cleans up build artifacts and temporary files.

```bash
./scripts/clean.sh [options]
```

**Options:**
- `--all` - Clean everything including virtual environment
- `--venv` - Clean virtual environment only
- `--build-only` - Clean only build artifacts
- `--help` - Show help

### 🛠️ `dev_tools.sh`
Development tools for code formatting, linting, and building.

```bash
./scripts/dev_tools.sh [command]
```

**Commands:**
- `format` - Format code with black and isort
- `lint` - Run flake8 linter
- `check` - Run both formatting and linting checks
- `fix` - Format code and fix auto-fixable issues
- `build` - Build the package
- `install` - Install in development mode
- `help` - Show help

### 🚀 `ci.sh`
Comprehensive CI/CD pipeline script that runs all checks.

```bash
./scripts/ci.sh
```

**What it checks:**
- Code formatting (black)
- Import sorting (isort)
- Linting (flake8)
- Package build
- Unit tests
- Package import
- CLI functionality

## Quick Start

1. **Set up development environment:**
   ```bash
   ./scripts/setup_venv.sh
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Build with GPU support:**
   ```bash
   ./scripts/build_complete.sh -c -v    # Complete clean build
   # OR
   ./scripts/quick_build.sh             # Quick development build
   ```

4. **Run tests:**
   ```bash
   ./scripts/run_tests.sh
   ```

5. **Format and check code:**
   ```bash
   ./scripts/dev_tools.sh fix
   ./scripts/dev_tools.sh check
   ```

6. **Run full CI pipeline:**
   ```bash
   ./scripts/ci.sh
   ```

## Development Workflow

For daily development:

```bash
# Initial setup (once)
./scripts/setup_venv.sh

# Quick builds during development
./scripts/quick_build.sh            # Build and test GPU features

# Before committing changes
./scripts/dev_tools.sh fix          # Fix formatting
./scripts/run_tests.sh              # Run tests
./scripts/ci.sh                     # Full CI check

# Complete builds for distribution
./scripts/build_complete.sh -c      # Clean build with all tests

# Clean up when needed
./scripts/clean.sh                  # Clean build artifacts
./scripts/clean.sh --all            # Full cleanup
```

## GPU Compute Capability Features

The build scripts now include comprehensive GPU compute capability support:

### ✅ **Automatic Detection**
- Detects GPU architecture at runtime
- Supports environment variable override (`OVEN_SM_ARCH`)
- Falls back to sm_50 if detection fails

### ✅ **CLI Integration**
```bash
oven-mlir input.mlir --format ptx --compute-capability sm_80
oven-mlir input.mlir --format ptx --sm sm_75
OVEN_SM_ARCH=sm_70 oven-mlir input.mlir --format ptx
```

### ✅ **Python API**
```python
import oven_mlir

# Get/set compute capability
print(oven_mlir.get_compute_capability())
oven_mlir.set_compute_capability('sm_80')

# Check target support
print(oven_mlir.check_targets())
print(oven_mlir.check_ptx_support())
```

### ✅ **Build Validation**
All build scripts now test:
- Native module GPU functionality
- CLI compute capability options
- Python API GPU functions
- PTX generation with correct targets
- PyPI manylinux compatibility

## Distribution

### **Test PyPI Upload**
```bash
./scripts/build_wheel.sh -c -t        # Build and upload to Test PyPI
```

### **Production PyPI Upload**
```bash
./scripts/build_wheel.sh -c -p        # Build and upload to PyPI
```