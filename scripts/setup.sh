#!/bin/bash

echo "Setting up Oven MLIR Compiler Environment"
echo "=========================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is required but not installed."
    echo "Please install python3-pip first:"
    echo "  sudo apt update && sudo apt install python3-pip"
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Check if CUDA is available (optional)
if command -v nvcc &> /dev/null; then
    echo "CUDA detected: $(nvcc --version | grep release)"
else
    echo "Warning: CUDA not detected. GPU tests may not work."
    echo "To install CUDA, please visit: https://developer.nvidia.com/cuda-downloads"
fi

# Verify PyCUDA installation (if CUDA is available)
if command -v nvcc &> /dev/null; then
    echo "Testing PyCUDA installation..."
    if python3 -c "import pycuda.driver as cuda; print('PyCUDA OK')" 2>/dev/null; then
        echo "PyCUDA installation successful!"
    else
        echo "Warning: PyCUDA installation may have issues."
        echo "You might need to set CUDA_HOME environment variable:"
        echo "  export CUDA_HOME=/usr/local/cuda"
    fi
fi

echo ""
echo "Setup complete!"
echo "You can now run tests with: ./scripts/run_test.sh"
echo "Or compile kernels with: ./scripts/compile.sh <file.mlir>"