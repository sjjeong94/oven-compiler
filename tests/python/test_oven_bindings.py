"""
Tests for Oven MLIR Python bindings
"""

import pytest
import tempfile
import os
from pathlib import Path


def test_import_oven():
    """Test that we can import the oven package"""
    import oven_mlir

    assert hasattr(oven_mlir, "__version__")


def test_oven_optimizer_creation():
    """Test creating an OvenOptimizer instance"""
    import oven_mlir

    try:
        optimizer = oven_mlir.OvenOptimizer()
        assert optimizer is not None
    except ImportError:
        pytest.skip("Native module not available")


def test_basic_mlir_optimization():
    """Test basic MLIR optimization"""
    import oven_mlir

    try:
        optimizer = oven_mlir.OvenOptimizer()
    except ImportError:
        pytest.skip("Native module not available")

    # Simple MLIR code
    mlir_code = """func.func @test_add(%arg0: f32, %arg1: f32) -> f32 {
  %result = arith.addf %arg0, %arg1 : f32
  return %result : f32
}"""

    result = optimizer.optimize_mlir(mlir_code)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "llvm.func" in result  # Should be converted to LLVM dialect


def test_file_optimization():
    """Test file-based optimization"""
    import oven_mlir

    try:
        optimizer = oven_mlir.OvenOptimizer()
    except ImportError:
        pytest.skip("Native module not available")

    # Find a test MLIR file
    test_files = ["tests/sigmoid.mlir", "tests/oven.mlir", "tests/matmul.mlir"]

    test_file = None
    for file_path in test_files:
        if os.path.exists(file_path):
            test_file = file_path
            break

    if test_file is None:
        pytest.skip("No test MLIR files found")

    result = optimizer.optimize_file(test_file)
    assert isinstance(result, str)
    assert len(result) > 0


def test_convenience_functions():
    """Test convenience functions"""
    import oven_mlir

    try:
        # Test the convenience functions exist
        assert hasattr(oven_mlir, "optimize_string")
        assert hasattr(oven_mlir, "optimize_file")

        # Try to use them
        mlir_code = """func.func @simple(%arg0: f32) -> f32 {
  return %arg0 : f32
}"""

        result = oven_mlir.optimize_string(mlir_code)
        assert isinstance(result, str)

    except ImportError:
        pytest.skip("Native module not available")


def test_oven_compiler():
    """Test the high-level OvenCompiler interface"""
    import oven_mlir

    try:
        compiler = oven_mlir.OvenCompiler()
        assert compiler is not None

        mlir_code = """func.func @test(%arg0: f32) -> f32 {
  %c1 = arith.constant 1.0 : f32
  %result = arith.addf %arg0, %c1 : f32
  return %result : f32
}"""

        # Test different formats
        result_mlir = compiler.compile_string(mlir_code, output_format="mlir")
        assert isinstance(result_mlir, str)
        assert len(result_mlir) > 0

    except (ImportError, AttributeError):
        pytest.skip("OvenCompiler not available")


def test_cli_import():
    """Test that CLI module can be imported"""
    import oven_mlir.cli

    assert hasattr(oven_mlir.cli, "main")


@pytest.mark.parametrize(
    "test_file", ["tests/sigmoid.mlir", "tests/oven.mlir", "tests/vectorize_op.mlir"]
)
def test_specific_oven_files(test_file):
    """Test optimization of specific Oven MLIR files"""
    import oven_mlir

    if not os.path.exists(test_file):
        pytest.skip(f"Test file {test_file} not found")

    try:
        optimizer = oven_mlir.OvenOptimizer()
        result = optimizer.optimize_file(test_file)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "llvm.func" in result

        # Check for Oven-specific optimizations
        if "sigmoid" in test_file:
            assert any(keyword in result for keyword in ["exp2", "fdiv", "fadd"])

    except ImportError:
        pytest.skip("Native module not available")
    except Exception as e:
        pytest.fail(f"Failed to optimize {test_file}: {e}")


def test_error_handling():
    """Test error handling for invalid MLIR"""
    import oven_mlir

    try:
        optimizer = oven_mlir.OvenOptimizer()
    except ImportError:
        pytest.skip("Native module not available")

    # Invalid MLIR should return an error message
    invalid_mlir = "this is not valid mlir code"
    result = optimizer.optimize_mlir(invalid_mlir)

    # Should return an error message, not crash
    assert isinstance(result, str)
    assert "Error" in result or "error" in result
