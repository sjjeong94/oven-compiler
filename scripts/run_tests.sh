#!/bin/bash
# Run tests for oven-mlir package
# Usage: ./scripts/run_tests.sh [options]

set -e

echo "🧪 Running oven-mlir tests..."

# Check if virtual environment exists, activate if available
if [ -d "venv" ]; then
    echo "📁 Using virtual environment..."
    source venv/bin/activate
else
    echo "📁 Using system Python environment..."
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Installing..."
    pip install pytest
fi

# Default options
PYTEST_ARGS="-v"
RUN_PYTHON_TESTS=true
RUN_MLIR_TESTS=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage)
            echo "📊 Running tests with coverage..."
            pip install pytest-cov
            PYTEST_ARGS="$PYTEST_ARGS --cov=oven_mlir --cov-report=html --cov-report=term"
            shift
            ;;
        --verbose)
            PYTEST_ARGS="$PYTEST_ARGS -vv"
            shift
            ;;
        --quiet)
            PYTEST_ARGS="-q"
            shift
            ;;
        --python-only)
            echo "🐍 Running Python tests only..."
            RUN_MLIR_TESTS=false
            shift
            ;;
        --mlir-only)
            echo "📄 Running MLIR tests only..."
            RUN_PYTHON_TESTS=false
            shift
            ;;
        --specific)
            if [[ -n $2 ]]; then
                PYTEST_ARGS="$PYTEST_ARGS -k $2"
                shift 2
            else
                echo "❌ --specific requires a test name pattern"
                exit 1
            fi
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --coverage    Run tests with coverage report"
            echo "  --verbose     Extra verbose output"
            echo "  --quiet       Minimal output"
            echo "  --python-only Run only Python tests"
            echo "  --mlir-only   Run only MLIR FileCheck tests"
            echo "  --specific PATTERN  Run only tests matching pattern"
            echo "  --help        Show this help"
            echo ""
            echo "This script runs both Python tests and MLIR FileCheck tests:"
            echo "  • Python tests: tests/python/ directory using pytest"
            echo "  • MLIR tests: tests/*.mlir files using oven-opt + FileCheck"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Running: python -m pytest $PYTEST_ARGS"
if [ "$RUN_PYTHON_TESTS" = true ]; then
    python -m pytest $PYTEST_ARGS
fi

if [ "$RUN_MLIR_TESTS" = true ]; then
    echo "🔍 Running MLIR FileCheck tests..."

# Check if oven-opt tool exists
if [ ! -f "./oven-opt" ] && [ ! -f "./tools/build/oven-opt" ]; then
    echo "⚠️ oven-opt tool not found. Building first..."
    ./scripts/quick_build.sh > /dev/null 2>&1
fi

# Set oven-opt path
if [ -f "./oven-opt" ]; then
    OVEN_OPT="./oven-opt"
elif [ -f "./tools/build/oven-opt" ]; then
    OVEN_OPT="./tools/build/oven-opt"
else
    echo "❌ Could not find oven-opt tool"
    exit 1
fi

# Check if FileCheck exists
FILECHECK=""
if [ -f "llvm-project/build/bin/FileCheck" ]; then
    FILECHECK="llvm-project/build/bin/FileCheck"
elif command -v FileCheck &> /dev/null; then
    FILECHECK="FileCheck"
else
    echo "⚠️ FileCheck not found. MLIR tests will be skipped."
    echo "✅ Python tests completed!"
    exit 0
fi

# Find MLIR test files and run FileCheck
MLIR_TEST_COUNT=0
MLIR_PASS_COUNT=0
MLIR_FAIL_COUNT=0

echo "📁 Searching for MLIR test files..."

for mlir_file in tests/*.mlir; do
    if [ -f "$mlir_file" ]; then
        # Check if file contains CHECK patterns
        if grep -q "CHECK" "$mlir_file"; then
            MLIR_TEST_COUNT=$((MLIR_TEST_COUNT + 1))
            echo "🧪 Testing: $mlir_file"
            
            # Run oven-opt and pipe to FileCheck
            if $OVEN_OPT "$mlir_file" --oven-to-llvm | $FILECHECK "$mlir_file" > /dev/null 2>&1; then
                echo "  ✅ PASS: $mlir_file"
                MLIR_PASS_COUNT=$((MLIR_PASS_COUNT + 1))
            else
                echo "  ❌ FAIL: $mlir_file"
                MLIR_FAIL_COUNT=$((MLIR_FAIL_COUNT + 1))
                
                # Show detailed error for failed tests
                echo "     Error details:"
                $OVEN_OPT "$mlir_file" --oven-to-llvm | $FILECHECK "$mlir_file" 2>&1 | head -5 | sed 's/^/     /'
            fi
        else
            echo "⚠️ Skipping $mlir_file (no CHECK patterns found)"
        fi
    fi
done

echo ""
echo "📊 MLIR FileCheck Results:"
echo "   Total tests: $MLIR_TEST_COUNT"
echo "   Passed: $MLIR_PASS_COUNT"
echo "   Failed: $MLIR_FAIL_COUNT"

if [ $MLIR_FAIL_COUNT -gt 0 ]; then
    echo "❌ Some MLIR tests failed!"
    exit 1
fi

fi  # End of RUN_MLIR_TESTS condition

echo "✅ Tests completed!"