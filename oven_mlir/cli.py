#!/usr/bin/env python3
"""
Command-line interface for Oven MLIR compiler
Supports both MLIR optimization and Python → PTX compilation
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional


def compile_mlir(input_file: str, output_file: Optional[str], format: str) -> int:
    """Compile MLIR file (original functionality)."""
    try:
        import oven_mlir

        # Use the direct optimizer interface for better compatibility
        optimizer = oven_mlir.OvenOptimizer()

        # Read input file
        with open(input_file, "r") as f:
            mlir_code = f.read()

        # Use the new optimize_and_convert method that supports all formats
        result = optimizer.optimize_and_convert(mlir_code, format)

        # Write output
        if output_file:
            with open(output_file, "w") as f:
                f.write(result)
            print(f"Output written to {output_file}")
        else:
            print(result)

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def compile_python_to_ptx(
    input_file: Optional[str] = None,
    input_string: Optional[str] = None,
    output_file: Optional[str] = None,
    save_intermediate: bool = False,
    intermediate_dir: str = ".",
    verbose: bool = False,
) -> int:
    """Compile Python to PTX (new functionality)."""
    try:
        # Import here to provide better error messages
        try:
            import oven_mlir
        except ImportError:
            print("❌ Error: oven-mlir not properly installed", file=sys.stderr)
            return 1

        # Check for Python compilation dependencies
        try:
            compiler = oven_mlir.PythonToPTXCompiler()
        except Exception as e:
            print(f"❌ Error: Python compilation not available: {e}", file=sys.stderr)
            print(
                "💡 Hint: Install oven-compiler with 'pip install oven-compiler'",
                file=sys.stderr,
            )
            return 1

        if verbose:
            print("🚀 Starting Python → PTX compilation...")

        # Get Python source code
        if input_file:
            if not os.path.exists(input_file):
                print(
                    f"❌ Error: Input file '{input_file}' does not exist",
                    file=sys.stderr,
                )
                return 1

            if verbose:
                print(f"📁 Compiling file: {input_file}")

            with open(input_file, "r") as f:
                python_code = f.read()
        else:
            python_code = input_string
            if verbose:
                print("📝 Compiling from string input")

        # Compile with or without intermediate files
        if save_intermediate:
            result = compiler.compile_with_intermediate_files(
                python_code, output_dir=intermediate_dir
            )
            ptx_code = result["ptx_code"]

            if verbose:
                print(f"💾 Saved intermediate files to: {intermediate_dir}")
                print(f"   - MLIR: {result['mlir']}")
                print(f"   - LLVM IR: {result['llvm_ir']}")
                print(f"   - PTX: {result['ptx']}")
        else:
            ptx_code = compiler.compile_python_to_ptx(python_code)

        if verbose:
            print("✅ Compilation successful!")

        # Handle output
        if output_file:
            # Create output directory if needed
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                f.write(ptx_code)

            if verbose:
                print(f"💾 PTX saved to: {output_file}")
        else:
            # Print to stdout
            print(ptx_code)

        return 0

    except Exception as e:
        print(f"❌ Compilation failed: {e}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


def main():
    """Main CLI entry point with support for both MLIR and Python compilation."""
    parser = argparse.ArgumentParser(
        prog="oven-mlir",
        description="Oven MLIR Compiler - Optimize MLIR and compile Python to PTX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MLIR compilation (original functionality)
  oven-mlir input.mlir --format ptx -o output.ptx
  
  # Python to PTX compilation
  oven-mlir --python my_kernel.py -o kernel.ptx
  
  # Python string to PTX
  oven-mlir --python-string "def add(a,b): return a+b" -o add.ptx
  
  # Save intermediate files for debugging
  oven-mlir --python kernel.py --intermediate --intermediate-dir ./debug
  
  # Verbose mode
  oven-mlir --python kernel.py --verbose
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "input", nargs="?", help="Input MLIR file (for MLIR compilation mode)"
    )
    input_group.add_argument(
        "--python", "-p", type=str, help="Python source file to compile to PTX"
    )
    input_group.add_argument(
        "--python-string",
        "-s",
        type=str,
        help="Python source code as string to compile to PTX",
    )

    # Output options
    parser.add_argument(
        "-o", "--output", type=str, help="Output file (default: print to stdout)"
    )

    # MLIR-specific options
    parser.add_argument(
        "--format",
        choices=["mlir", "llvm", "ptx"],
        default="mlir",
        help="Output format for MLIR compilation (default: mlir)",
    )

    # Python-specific options
    parser.add_argument(
        "--intermediate",
        action="store_true",
        help="Save intermediate MLIR and LLVM IR files (Python mode only)",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=str,
        default=".",
        help="Directory for intermediate files (default: current directory)",
    )

    # General options
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument("--version", action="version", version="oven-mlir 1.0.0")

    # Parse arguments
    args = parser.parse_args()

    # Determine compilation mode
    if args.python or args.python_string:
        # Python → PTX mode
        exit_code = compile_python_to_ptx(
            input_file=args.python,
            input_string=args.python_string,
            output_file=args.output,
            save_intermediate=args.intermediate,
            intermediate_dir=args.intermediate_dir,
            verbose=args.verbose,
        )
    elif args.input:
        # MLIR compilation mode
        exit_code = compile_mlir(
            input_file=args.input, output_file=args.output, format=args.format
        )
    else:
        parser.error("Must provide either input MLIR file or --python/--python-string")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
