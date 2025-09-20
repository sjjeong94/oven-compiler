#!/bin/bash
#
# Quick build script for development
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🔧 Quick oven-mlir build"
echo "========================"

cd "$PROJECT_ROOT"

# Install build dependencies if needed
echo "📦 Installing build dependencies..."
pip install -q build twine wheel

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Build package
echo "🏗️  Building package..."
python -m build --wheel

# Validate package
echo "✅ Validating package..."
python -m twine check dist/*

echo
echo "🎉 Build completed successfully!"
echo "📁 Files created:"
ls -lah dist/

echo
echo "🚀 To upload to Test PyPI:"
echo "   python -m twine upload --repository testpypi dist/*"
echo
echo "🚀 To upload to PyPI:"  
echo "   python -m twine upload dist/*"