#!/bin/bash
# Clean up build artifacts and temporary files
# Usage: ./scripts/clean.sh [options]

set -e

echo "🧹 Cleaning oven-mlir development environment..."

# Default behavior
CLEAN_VENV=false
CLEAN_BUILD=true
CLEAN_CACHE=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            CLEAN_VENV=true
            CLEAN_BUILD=true
            CLEAN_CACHE=true
            shift
            ;;
        --venv)
            CLEAN_VENV=true
            shift
            ;;
        --build-only)
            CLEAN_VENV=false
            CLEAN_BUILD=true
            CLEAN_CACHE=false
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --all         Clean everything including virtual environment"
            echo "  --venv        Clean virtual environment"
            echo "  --build-only  Clean only build artifacts"
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

# Clean build artifacts
if [ "$CLEAN_BUILD" = true ]; then
    echo "🗑️ Removing build artifacts..."
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    rm -rf oven_mlir.egg-info/
    find . -name "*.so" -type f -delete
    find . -name "*.dylib" -type f -delete
    find . -name "*.dll" -type f -delete
    echo "✅ Build artifacts cleaned"
fi

# Clean cache files
if [ "$CLEAN_CACHE" = true ]; then
    echo "🗑️ Removing cache files..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete
    find . -name "*.pyo" -delete
    rm -rf .coverage
    rm -rf htmlcov/
    echo "✅ Cache files cleaned"
fi

# Clean virtual environment
if [ "$CLEAN_VENV" = true ]; then
    echo "🗑️ Removing virtual environment..."
    rm -rf venv/
    echo "✅ Virtual environment removed"
    echo "ℹ️ Run ./scripts/setup_venv.sh to recreate it"
fi

echo "🎉 Cleanup complete!"