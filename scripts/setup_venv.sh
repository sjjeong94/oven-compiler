#!/bin/bash
# Setup virtual environment and install oven-mlir package
# Usage: ./scripts/setup_venv.sh

set -e  # Exit on any error

echo "🔧 Setting up virtual environment for oven-mlir..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "📦 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install build dependencies
echo "🛠️ Installing build dependencies..."
pip install nanobind cmake ninja setuptools wheel numpy

# Install development dependencies
echo "🧪 Installing development dependencies..."
pip install pytest black isort flake8

# Install the package in development mode
echo "📦 Installing oven-mlir in development mode..."
pip install -e .

echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  python -m pytest"
echo ""
echo "To use the CLI:"
echo "  oven-compile --help"