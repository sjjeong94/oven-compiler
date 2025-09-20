#!/usr/bin/env python3
"""
Setup script for Oven MLIR Python bindings
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """A Python extension that is built using CMake"""

    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build command that uses CMake to build the extension"""

    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DMLIR_DIR={ext.sourcedir}/llvm-project/build/lib/cmake/mlir",
            f"-DLLVM_DIR={ext.sourcedir}/llvm-project/build/lib/cmake/llvm",
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        # Platform-specific arguments
        if sys.platform.startswith("win"):
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]
            build_args += ["--", "-j4"]

        env = os.environ.copy()
        env["CXXFLAGS"] = (
            f"{env.get('CXXFLAGS', '')} -DVERSION_INFO=\"{self.distribution.get_version()}\""
        )

        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # Check if we need to find nanobind
        try:
            import nanobind

            cmake_args.append(f"-Dnanobind_DIR={nanobind.cmake_dir()}")
        except ImportError:
            print("Warning: nanobind not found, trying system installation")

        # Configure
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp, env=env
        )

        # Build only the Python extension target
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "oven_opt_py"] + build_args,
            cwd=build_temp,
        )


# Read the README file
def read_readme():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Oven MLIR Python Bindings"


setup(
    name="oven-mlir",
    version="0.1.0",
    description="Python bindings for Oven MLIR compiler",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Sinjin Jeong",
    author_email="sjjeong94@gmail.com",
    url="https://github.com/sjjeong94/oven",
    packages=["oven_mlir"],
    ext_modules=[CMakeExtension("oven_mlir.oven_opt_py")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
    install_requires=["numpy>=1.20.0"],
    extras_require={
        "dev": ["pytest>=6.0", "pytest-cov", "black", "isort", "flake8"],
        "docs": ["sphinx", "sphinx-rtd-theme", "myst-parser"],
    },
    entry_points={
        "console_scripts": [
            "oven-compile=oven_mlir.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Compilers",
    ],
    zip_safe=False,
)
