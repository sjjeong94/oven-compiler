#!/usr/bin/env python3
"""
Setup script for oven-mlir package with native module support.

This script ensures that wheels are built with the correct platform tags
when native modules are included.
"""

import os
import sys
import glob
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

# Try to import bdist_wheel, fall back if not available
try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:
    try:
        from setuptools.command.bdist_wheel import bdist_wheel
    except ImportError:
        bdist_wheel = None

import platform


class CustomBuildPy(build_py):
    """Custom build command that ensures native modules are included."""

    def run(self):
        super().run()

        # Copy native modules to build directory
        native_modules = (
            glob.glob("oven_mlir/*.so")
            + glob.glob("oven_mlir/*.pyd")
            + glob.glob("oven_mlir/*.dll")
            + glob.glob("oven_mlir/*.dylib")
        )

        if native_modules:
            print(f"Found native modules: {native_modules}")

            # Ensure build directory exists
            build_lib = self.build_lib
            target_dir = os.path.join(build_lib, "oven_mlir")
            os.makedirs(target_dir, exist_ok=True)

            # Copy native modules
            for module in native_modules:
                target = os.path.join(target_dir, os.path.basename(module))
                self.copy_file(module, target)
                print(f"Copied {module} -> {target}")


class CustomBdistWheel(bdist_wheel if bdist_wheel else object):
    """Custom wheel building that forces platform-specific tags."""

    def finalize_options(self):
        if bdist_wheel:
            super().finalize_options()

        # Check if we have native modules
        has_native = any(
            glob.glob(pattern)
            for pattern in [
                "oven_mlir/*.so",
                "oven_mlir/*.pyd",
                "oven_mlir/*.dll",
                "oven_mlir/*.dylib",
            ]
        )

        if has_native:
            # Force platform-specific wheel
            self.root_is_pure = False
            print("Native modules detected - building platform-specific wheel")
        else:
            print("No native modules found - building universal wheel")

    def get_tag(self):
        if not bdist_wheel:
            return "py3", "none", "any"

        # Get the default tag
        python, abi, plat = super().get_tag()

        # Check if we have native modules
        has_native = any(
            glob.glob(pattern)
            for pattern in [
                "oven_mlir/*.so",
                "oven_mlir/*.pyd",
                "oven_mlir/*.dll",
                "oven_mlir/*.dylib",
            ]
        )

        if has_native:
            # Use manylinux tags for PyPI compatibility
            if plat == "any" or plat.startswith("linux"):
                # Determine actual platform
                if sys.platform.startswith("linux"):
                    # Use manylinux2014 for better compatibility with modern systems
                    arch = platform.machine()
                    if arch == "x86_64":
                        plat = "manylinux2014_x86_64"
                    elif arch == "i686":
                        plat = "manylinux2014_i686"
                    elif arch == "aarch64":
                        plat = "manylinux2014_aarch64"
                    else:
                        plat = f"manylinux2014_{arch}"
                elif sys.platform == "darwin":
                    mac_ver = platform.mac_ver()[0].replace(".", "_")
                    plat = f"macosx_{mac_ver}_{platform.machine()}"
                elif sys.platform == "win32":
                    arch = platform.machine().lower()
                    if arch in ["amd64", "x86_64"]:
                        plat = "win_amd64"
                    elif arch in ["i386", "i686"]:
                        plat = "win32"
                    else:
                        plat = f"win_{arch}"

            print(f"Using platform tag: {plat}")

        return python, abi, plat


def has_native_modules():
    """Check if native modules are present."""
    patterns = [
        "oven_mlir/*.so",
        "oven_mlir/*.pyd",
        "oven_mlir/*.dll",
        "oven_mlir/*.dylib",
    ]

    for pattern in patterns:
        if glob.glob(pattern):
            return True
    return False


if __name__ == "__main__":
    # Check for native modules
    native_present = has_native_modules()
    print(f"Native modules present: {native_present}")

    if native_present:
        native_files = []
        for pattern in ["*.so", "*.pyd", "*.dll", "*.dylib"]:
            native_files.extend(glob.glob(f"oven_mlir/{pattern}"))
        print(f"Native files: {native_files}")

    # Prepare cmdclass
    cmdclass = {
        "build_py": CustomBuildPy,
    }

    if bdist_wheel:
        cmdclass["bdist_wheel"] = CustomBdistWheel

    setup(cmdclass=cmdclass)
