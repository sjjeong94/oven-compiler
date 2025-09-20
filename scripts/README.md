# Development Scripts

This directory contains utility scripts for oven-mlir development.

## Scripts Overview

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

3. **Run tests:**
   ```bash
   ./scripts/run_tests.sh
   ```

4. **Format and check code:**
   ```bash
   ./scripts/dev_tools.sh fix
   ./scripts/dev_tools.sh check
   ```

5. **Run full CI pipeline:**
   ```bash
   ./scripts/ci.sh
   ```

## Development Workflow

For daily development:

```bash
# Initial setup (once)
./scripts/setup_venv.sh

# Before committing changes
./scripts/dev_tools.sh fix          # Fix formatting
./scripts/run_tests.sh              # Run tests
./scripts/ci.sh                     # Full CI check

# Clean up when needed
./scripts/clean.sh                  # Clean build artifacts
./scripts/clean.sh --all            # Full cleanup
```