#!/usr/bin/env python3
"""
Command-line interface for Oven MLIR compiler
"""

import argparse
import sys
import os
from pathlib import Path


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Oven MLIR Compiler - Optimize and compile MLIR kernels for GPU"
    )

    parser.add_argument("input", help="Input MLIR file")

    parser.add_argument("-o", "--output", help="Output file (default: stdout)")

    parser.add_argument(
        "--format",
        choices=["mlir", "llvm", "ptx"],
        default="mlir",
        help="Output format (default: mlir)",
    )

    parser.add_argument("--version", action="version", version="oven-mlir 0.1.0")

    args = parser.parse_args()

    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)

    try:
        import oven_mlir

        # Use the direct optimizer interface for better compatibility
        optimizer = oven_mlir.OvenOptimizer()

        # Read input file
        with open(args.input, "r") as f:
            mlir_code = f.read()

        # Use the new optimize_and_convert method that supports all formats
        result = optimizer.optimize_and_convert(mlir_code, args.format)

        # Output result
        if args.output:
            with open(args.output, "w") as f:
                f.write(result)
            print(f"Output written to {args.output}")
        else:
            print(result)

    except ImportError as e:
        print(f"Error: Oven MLIR native module not available: {e}", file=sys.stderr)
        print(
            "Make sure the package was installed correctly with native extensions",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error during compilation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
