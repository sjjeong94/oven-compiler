#!/bin/bash
# Run tests for oven-mlir package
# Usage: ./scripts/run_tests.sh [options]

set -e

echo "🧪 Running oven-mlir tests..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./scripts/setup_venv.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Installing..."
    pip install pytest
fi

# Default options
PYTEST_ARGS="-v"

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
            echo "  --specific PATTERN  Run only tests matching pattern"
            echo "  --help        Show this help"
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
python -m pytest $PYTEST_ARGS

echo "✅ Tests completed!"