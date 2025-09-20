#!/usr/bin/env python3
"""
Package configuration checker for oven-mlir
Validates pyproject.toml and build setup
"""

import sys
import os
from pathlib import Path
import subprocess
import importlib.util


def check_color(text, color_code):
    """Add color to terminal output"""
    return f"\033[{color_code}m{text}\033[0m"


def success(text):
    return check_color(f"✓ {text}", "92")


def warning(text):
    return check_color(f"⚠ {text}", "93")


def error(text):
    return check_color(f"✗ {text}", "91")


def info(text):
    return check_color(f"ℹ {text}", "94")


def check_file_exists(file_path, description):
    """Check if a file exists"""
    if file_path.exists():
        print(success(f"{description} exists: {file_path}"))
        return True
    else:
        print(error(f"{description} missing: {file_path}"))
        return False


def check_python_imports():
    """Check if package can be imported"""
    print("\n" + "=" * 50)
    print("🐍 Python Import Checks")
    print("=" * 50)

    try:
        import oven_mlir

        print(success("oven_mlir package imports successfully"))

        # Check version
        if hasattr(oven_mlir, "__version__"):
            print(info(f"Package version: {oven_mlir.__version__}"))

        # Check main components
        try:
            from oven_mlir import cli

            print(success("CLI module imports successfully"))
        except ImportError as e:
            print(warning(f"CLI import issue: {e}"))

        try:
            from oven_mlir import python_to_ptx

            print(success("python_to_ptx module imports successfully"))
        except ImportError as e:
            print(warning(f"python_to_ptx import issue: {e}"))

        return True
    except ImportError as e:
        print(error(f"Cannot import oven_mlir: {e}"))
        return False


def check_dependencies():
    """Check build dependencies"""
    print("\n" + "=" * 50)
    print("📦 Build Dependencies")
    print("=" * 50)

    dependencies = [
        ("build", "python -m build"),
        ("twine", "python -m twine"),
        ("wheel", "pip show wheel"),
        ("setuptools", "pip show setuptools"),
    ]

    all_good = True
    for dep, check_cmd in dependencies:
        try:
            result = subprocess.run(check_cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print(success(f"{dep} is available"))
            else:
                print(error(f"{dep} not found"))
                all_good = False
        except FileNotFoundError:
            print(error(f"{dep} check failed - command not found"))
            all_good = False

    return all_good


def check_files():
    """Check required files"""
    print("\n" + "=" * 50)
    print("📄 Required Files")
    print("=" * 50)

    project_root = Path(__file__).parent.parent

    required_files = [
        (project_root / "pyproject.toml", "Build configuration"),
        (project_root / "README.md", "Package description"),
        (project_root / "LICENSE", "License file"),
        (project_root / "oven_mlir" / "__init__.py", "Package __init__.py"),
        (project_root / "oven_mlir" / "cli.py", "CLI module"),
        (project_root / "oven_mlir" / "python_to_ptx.py", "Python-to-PTX compiler"),
    ]

    all_exist = True
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            all_exist = False

    return all_exist


def check_pyproject_toml():
    """Check pyproject.toml configuration"""
    print("\n" + "=" * 50)
    print("⚙️  pyproject.toml Configuration")
    print("=" * 50)

    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        print(error("pyproject.toml not found"))
        return False

    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            print(warning("Cannot parse TOML - install tomli/tomllib"))
            return False

    try:
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        # Check project metadata
        project = config.get("project", {})

        required_fields = ["name", "version", "description", "authors"]
        for field in required_fields:
            if field in project:
                print(success(f"Project {field}: {project[field]}"))
            else:
                print(error(f"Missing project.{field}"))

        # Check dependencies
        deps = project.get("dependencies", [])
        print(info(f"Dependencies: {len(deps)} packages"))

        # Check optional dependencies
        optional_deps = project.get("optional-dependencies", {})
        print(info(f"Optional dependency groups: {list(optional_deps.keys())}"))

        # Check scripts
        scripts = project.get("scripts", {})
        if scripts:
            print(success(f"Console scripts: {list(scripts.keys())}"))
        else:
            print(warning("No console scripts defined"))

        return True

    except Exception as e:
        print(error(f"Error parsing pyproject.toml: {e}"))
        return False


def check_build_artifacts():
    """Check for existing build artifacts"""
    print("\n" + "=" * 50)
    print("🏗️  Build Artifacts")
    print("=" * 50)

    project_root = Path(__file__).parent.parent

    # Check for compiled extensions
    so_files = list(project_root.glob("oven_mlir/*.so"))
    if so_files:
        print(success(f"Found {len(so_files)} compiled extensions"))
        for so_file in so_files:
            print(info(f"  • {so_file.name}"))
    else:
        print(warning("No compiled extensions found"))

    # Check dist directory
    dist_dir = project_root / "dist"
    if dist_dir.exists():
        wheels = list(dist_dir.glob("*.whl"))
        tarballs = list(dist_dir.glob("*.tar.gz"))

        if wheels:
            print(success(f"Found {len(wheels)} wheel files"))
        if tarballs:
            print(success(f"Found {len(tarballs)} source distributions"))

        if not wheels and not tarballs:
            print(info("No built packages in dist/"))
    else:
        print(info("No dist/ directory found"))


def run_build_test():
    """Test the build process"""
    print("\n" + "=" * 50)
    print("🧪 Build Test")
    print("=" * 50)

    try:
        print(info("Testing package build..."))
        result = subprocess.run(
            ["python", "-m", "build", "--wheel", "--no-isolation"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            print(success("Build test passed"))
            return True
        else:
            print(error("Build test failed"))
            print(f"Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(error("Build test timed out"))
        return False
    except Exception as e:
        print(error(f"Build test error: {e}"))
        return False


def main():
    """Main checker function"""
    print("🔍 oven-mlir Package Configuration Checker")
    print("=" * 60)

    checks = [
        ("Files", check_files),
        ("pyproject.toml", check_pyproject_toml),
        ("Dependencies", check_dependencies),
        ("Python Imports", check_python_imports),
        ("Build Artifacts", check_build_artifacts),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(error(f"Check {name} failed with exception: {e}"))
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = success("PASS") if result else error("FAIL")
        print(f"{name:20} {status}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print(success("\n🎉 All checks passed! Package is ready for distribution."))

        print("\nNext steps:")
        print("  1. Build package: ./scripts/build_wheel.sh")
        print("  2. Test upload: ./scripts/build_wheel.sh -t")
        print("  3. Upload to PyPI: ./scripts/build_wheel.sh -p")

        return 0
    else:
        print(
            error(
                f"\n❌ {total - passed} checks failed. Please fix issues before building."
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
