#!/bin/bash
# Development tools for oven-mlir
# Usage: ./scripts/dev_tools.sh [command] [options]

set -e

# Check if virtual environment exists and activate it
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./scripts/setup_venv.sh first"
    exit 1
fi

source venv/bin/activate

# Function to show help
show_help() {
    echo "oven-mlir Development Tools"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  format        Format code with black and isort"
    echo "  lint          Run flake8 linter"
    echo "  check         Run both formatting and linting checks"
    echo "  fix           Format code and fix auto-fixable issues"
    echo "  build         Build the package"
    echo "  install       Install in development mode"
    echo "  docs          Generate documentation (if available)"
    echo "  help          Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 format"
    echo "  $0 lint"
    echo "  $0 check"
    echo "  $0 build"
}

# Function to install dev dependencies if needed
ensure_dev_deps() {
    if ! python -c "import black, isort, flake8" 2>/dev/null; then
        echo "📦 Installing development dependencies..."
        pip install black isort flake8
    fi
}

# Parse command
case "${1:-help}" in
    format)
        ensure_dev_deps
        echo "🎨 Formatting code with black and isort..."
        black oven_mlir/ tests/ --line-length 88
        isort oven_mlir/ tests/ --profile black
        echo "✅ Code formatted!"
        ;;
    
    lint)
        ensure_dev_deps
        echo "🔍 Running flake8 linter..."
        flake8 oven_mlir/ tests/ --max-line-length=88 --ignore=E203,W503
        echo "✅ Linting passed!"
        ;;
    
    check)
        ensure_dev_deps
        echo "🔍 Checking code formatting..."
        black oven_mlir/ tests/ --check --line-length 88
        isort oven_mlir/ tests/ --check-only --profile black
        echo "🔍 Running linter..."
        flake8 oven_mlir/ tests/ --max-line-length=88 --ignore=E203,W503
        echo "✅ All checks passed!"
        ;;
    
    fix)
        ensure_dev_deps
        echo "🔧 Fixing code formatting and style..."
        black oven_mlir/ tests/ --line-length 88
        isort oven_mlir/ tests/ --profile black
        echo "✅ Code fixed!"
        ;;
    
    build)
        echo "🔨 Building package..."
        python setup.py build
        echo "✅ Build completed!"
        ;;
    
    install)
        echo "📦 Installing in development mode..."
        pip install -e .
        echo "✅ Installation completed!"
        ;;
    
    docs)
        echo "📚 Documentation generation not yet implemented"
        echo "ℹ️ You can add sphinx or other documentation tools here"
        ;;
    
    help|--help|-h)
        show_help
        ;;
    
    *)
        echo "❌ Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac