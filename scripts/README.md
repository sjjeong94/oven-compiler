# oven-compiler Scripts

This directory contains 3 core scripts essential for oven-compiler development.

## 📁 Scripts Overview

### 1. 🔧 install_mlir.sh
**Purpose**: Install MLIR/LLVM dependencies
```bash
./scripts/install_mlir.sh
```
- Clone and build LLVM project
- Install MLIR libraries
- Set up initial development environment

### 2. ⚡ quick_build.sh  
**Purpose**: Fast development build
```bash
./scripts/quick_build.sh
```
- Build native modules
- Install Python package in development mode
- Auto-detect GPU compute capability
- Support rapid development cycles

### 3. 🧪 run_tests.sh
**Purpose**: Run test suite
```bash
./scripts/run_tests.sh
```
- Execute unit tests
- Perform integration tests
- Validate GPU functionality
- Check code quality

## 🚀 Quick Start

Complete setup and development workflow:

```bash
# 1. Install MLIR
./scripts/install_mlir.sh

# 2. Build
./scripts/quick_build.sh

# 3. Test
./scripts/run_tests.sh
```

## 📝 Notes

- Run all scripts from the project root directory
- GPU compute capability is auto-detected (using nvidia-smi)
- Use `quick_build.sh` frequently during development