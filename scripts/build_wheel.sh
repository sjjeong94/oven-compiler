#!/bin/bash
#
# Enhanced Wheel Build Script for oven-mlir
# Builds and validates Python wheels for PyPI distribution with platform detection
#

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$PROJECT_ROOT/dist"
BUILD_DIR="$PROJECT_ROOT/build"

# Build options
CLEAN_BUILD=false
UPLOAD_TEST=false
UPLOAD_PYPI=false
SKIP_TESTS=false
VERBOSE=false
CHECK_DEPS=true
FORCE_PLATFORM=""
BUILD_SOURCE=false

# Detect platform
PLATFORM=$(python -c "import platform; print(platform.system().lower())")
ARCH=$(python -c "import platform; print(platform.machine())")
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${PURPLE}[DEBUG]${NC} $1"
    fi
}

print_header() {
    echo -e "${CYAN}================================${NC}"
    echo -e "${CYAN} $1${NC}"
    echo -e "${CYAN}================================${NC}"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build oven-mlir wheel packages for PyPI distribution with enhanced platform support.

OPTIONS:
    -c, --clean              Clean build directories before building
    -t, --test-upload        Upload to Test PyPI after building
    -p, --pypi-upload        Upload to PyPI after building (requires confirmation)
    -s, --skip-tests         Skip running tests before building
    -v, --verbose            Enable verbose output
    --no-deps-check          Skip dependency checking
    --platform PLATFORM      Force platform tag (e.g., linux_x86_64, win_amd64)
    --source                 Also build source distribution
    -h, --help               Show this help message

EXAMPLES:
    $0                       # Basic build
    $0 -c -v                 # Clean build with verbose output
    $0 -c -t                 # Clean build and upload to Test PyPI
    $0 -c -p                 # Clean build and upload to PyPI
    $0 --platform win_amd64  # Force Windows platform tag
    $0 --source              # Build both wheel and source distribution

PLATFORM DETECTION:
    Current Platform: $PLATFORM ($ARCH)
    Python Version: $PYTHON_VERSION
    
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -t|--test-upload)
            UPLOAD_TEST=true
            shift
            ;;
        -p|--pypi-upload)
            UPLOAD_PYPI=true
            shift
            ;;
        -s|--skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --no-deps-check)
            CHECK_DEPS=false
            shift
            ;;
        --platform)
            FORCE_PLATFORM="$2"
            shift 2
            ;;
        --source)
            BUILD_SOURCE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check and install dependencies
check_dependencies() {
    if [[ "$CHECK_DEPS" == "false" ]]; then
        print_status "Skipping dependency check"
        return 0
    fi
    
    print_header "Checking Build Dependencies"
    
    local missing_deps=()
    
    # Check Python build tools
    if ! python -c "import build" >/dev/null 2>&1; then
        missing_deps+=("build")
    fi
    
    if ! python -c "import twine" >/dev/null 2>&1; then
        missing_deps+=("twine")
    fi
    
    if ! python -c "import wheel" >/dev/null 2>&1; then
        missing_deps+=("wheel")
    fi
    
    # Check project dependencies
    if ! python -c "import numpy" >/dev/null 2>&1; then
        missing_deps+=("numpy")
    fi
    
    if ! python -c "import oven_compiler" >/dev/null 2>&1; then
        missing_deps+=("oven-compiler")
    fi
    
    # Install missing dependencies
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_warning "Missing dependencies: ${missing_deps[*]}"
        print_status "Installing missing dependencies..."
        pip install "${missing_deps[@]}"
        print_success "Dependencies installed"
    else
        print_success "All dependencies are available"
    fi
}

# Function to detect native modules
detect_native_modules() {
    print_status "Detecting native modules..."
    
    local native_files=()
    while IFS= read -r -d '' file; do
        native_files+=("$file")
    done < <(find "$PROJECT_ROOT/oven_mlir" -name "*.so" -o -name "*.pyd" -o -name "*.dll" -o -name "*.dylib" -print0 2>/dev/null)
    
    if [[ ${#native_files[@]} -gt 0 ]]; then
        print_success "Found ${#native_files[@]} native module(s):"
        for file in "${native_files[@]}"; do
            local size=$(du -h "$file" | cut -f1)
            print_debug "  $(basename "$file") ($size)"
        done
        return 0
    else
        print_warning "No native modules found - will build universal wheel"
        return 1
    fi
}

# Function to get platform tag
get_platform_tag() {
    if [[ -n "$FORCE_PLATFORM" ]]; then
        echo "$FORCE_PLATFORM"
        return 0
    fi
    
    case "$PLATFORM" in
        linux)
            # Use manylinux tags for PyPI compatibility
            case "$ARCH" in
                x86_64)
                    echo "manylinux2014_x86_64"
                    ;;
                aarch64)
                    echo "manylinux2014_aarch64"
                    ;;
                i686)
                    echo "manylinux2014_i686"
                    ;;
                *)
                    echo "linux_${ARCH}"
                    ;;
            esac
            ;;
        darwin)
            local macos_version=$(python -c "import platform; print('_'.join(platform.mac_ver()[0].split('.')[:2]))")
            echo "macosx_${macos_version}_${ARCH}"
            ;;
        windows)
            case "$ARCH" in
                x86_64|AMD64)
                    echo "win_amd64"
                    ;;
                i*86)
                    echo "win32"
                    ;;
                *)
                    echo "win_${ARCH}"
                    ;;
            esac
            ;;
        *)
            echo "any"
            ;;
    esac
}

# Function to clean build directories
clean_build() {
    print_header "Cleaning Build Environment"
    
    if [[ -d "$DIST_DIR" ]]; then
        rm -rf "$DIST_DIR"
        print_status "Removed $DIST_DIR"
    fi
    
    if [[ -d "$BUILD_DIR" ]]; then
        rm -rf "$BUILD_DIR"
        print_status "Removed $BUILD_DIR"
    fi
    
    if [[ -d "$PROJECT_ROOT/oven_mlir.egg-info" ]]; then
        rm -rf "$PROJECT_ROOT/oven_mlir.egg-info"
        print_status "Removed egg-info directory"
    fi
    
    # Clean Python cache
    find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true
    
    print_success "Build directories cleaned"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    # Check if pytest is available
    if ! python -c "import pytest" >/dev/null 2>&1; then
        print_warning "pytest not found, installing..."
        pip install pytest
    fi
    
    # Run tests if they exist
    if [[ -d "tests" ]]; then
        if [[ "$VERBOSE" == "true" ]]; then
            python -m pytest tests/ -v
        else
            python -m pytest tests/ -q
        fi
        print_success "Tests passed"
    else
        print_warning "No tests directory found, skipping tests"
    fi
}

# Function to build wheel
build_wheel() {
    print_header "Building Wheel Package"
    
    cd "$PROJECT_ROOT"
    
    # Create dist directory
    mkdir -p "$DIST_DIR"
    
    # Detect native modules
    local has_native=false
    if detect_native_modules; then
        has_native=true
        local platform_tag=$(get_platform_tag)
        print_status "Building platform-specific wheel for: $platform_tag"
    else
        print_status "Building universal wheel"
    fi
    
    # Build arguments
    local build_args=("--wheel")
    if [[ "$VERBOSE" == "true" ]]; then
        build_args+=("--verbose")
    fi
    
    # Build source distribution if requested
    if [[ "$BUILD_SOURCE" == "true" ]]; then
        print_status "Building source distribution..."
        python -m build --sdist "${build_args[@]}"
        print_success "Source distribution built"
    fi
    
    # Build the wheel
    print_status "Building wheel..."
    python -m build "${build_args[@]}"
    
    # Check if build was successful
    local wheel_count=$(find "$DIST_DIR" -name "*.whl" | wc -l)
    if [[ $wheel_count -eq 0 ]]; then
        print_error "Wheel build failed - no .whl file found"
        exit 1
    fi
    
    print_success "Wheel built successfully"
    
    # Show built files with details
    print_status "Built files:"
    for file in "$DIST_DIR"/*; do
        if [[ -f "$file" ]]; then
            local size=$(du -h "$file" | cut -f1)
            local filename=$(basename "$file")
            printf "  %-50s %s\n" "$filename" "$size"
            
            # Analyze wheel contents if verbose
            if [[ "$VERBOSE" == "true" && "$file" == *.whl ]]; then
                print_debug "Wheel contents:"
                python -m zipfile -l "$file" | head -20
                if [[ $(python -m zipfile -l "$file" | wc -l) -gt 20 ]]; then
                    print_debug "... (truncated, total $(python -m zipfile -l "$file" | wc -l) files)"
                fi
            fi
        fi
    done
}

# Function to validate wheel
validate_wheel() {
    print_header "Validating Wheel Package"
    
    cd "$PROJECT_ROOT"
    
    # Check package with twine
    print_status "Running twine check..."
    python -m twine check "$DIST_DIR"/*
    
    if [[ $? -eq 0 ]]; then
        print_success "Twine validation passed"
    else
        print_error "Twine validation failed"
        exit 1
    fi
    
    # Check wheel metadata
    for wheel in "$DIST_DIR"/*.whl; do
        if [[ -f "$wheel" ]]; then
            print_status "Checking wheel metadata: $(basename "$wheel")"
            
            # Extract and show WHEEL metadata for platform tag verification
            local wheel_metadata=$(python -c "
import zipfile
import sys
try:
    with zipfile.ZipFile('$wheel', 'r') as zf:
        wheel_files = [f for f in zf.namelist() if f.endswith('.dist-info/WHEEL')]
        if wheel_files:
            content = zf.read(wheel_files[0]).decode('utf-8')
            print(content)
except Exception as e:
    print(f'Error reading wheel: {e}', file=sys.stderr)
")
            
            if [[ -n "$wheel_metadata" ]]; then
                echo "$wheel_metadata" | grep -E "^(Wheel-Version|Generator|Root-Is-Purelib|Tag):" | while read line; do
                    if [[ "$line" == Tag:* ]]; then
                        local tag="${line#Tag: }"
                        if [[ "$tag" == *"manylinux"* ]]; then
                            print_success "  $line (PyPI compatible)"
                        elif [[ "$tag" == *"linux"* ]]; then
                            print_warning "  $line (may not be PyPI compatible - consider manylinux)"
                        else
                            print_debug "  $line"
                        fi
                    else
                        print_debug "  $line"
                    fi
                done
            fi
            
            # Extract and show key metadata
            local metadata=$(python -c "
import zipfile
try:
    with zipfile.ZipFile('$wheel', 'r') as zf:
        metadata_files = [f for f in zf.namelist() if f.endswith('.dist-info/METADATA')]
        if metadata_files:
            content = zf.read(metadata_files[0]).decode('utf-8')
            print(content)
except:
    pass
")
            
            if [[ -n "$metadata" ]]; then
                echo "$metadata" | grep -E "^(Name|Version|Platform|Requires-Dist):" | while read line; do
                    print_debug "  $line"
                done
            fi
            
            # Check for native modules in wheel
            if python -m zipfile -l "$wheel" | grep -E "\.(so|pyd|dll|dylib)$" >/dev/null; then
                print_success "  Contains native modules - platform-specific wheel"
                
                # List native modules
                if [[ "$VERBOSE" == "true" ]]; then
                    print_debug "  Native modules found:"
                    python -m zipfile -l "$wheel" | grep -E "\.(so|pyd|dll|dylib)$" | while read line; do
                        print_debug "    $(echo "$line" | awk '{print $NF}')"
                    done
                fi
            else
                print_warning "  No native modules found - universal wheel"
            fi
        fi
    done
}

# Function to test wheel installation
test_wheel_install() {
    print_status "Testing wheel installation..."
    
    # Create temporary virtual environment for testing
    local temp_venv=$(mktemp -d)
    python -m venv "$temp_venv"
    source "$temp_venv/bin/activate"
    
    # Install the wheel
    pip install "$DIST_DIR"/*.whl
    
    # Test import
    python -c "import oven_mlir; print('✓ oven_mlir import successful')"
    python -c "import oven_mlir.cli; print('✓ CLI module import successful')"
    
    # Test CLI
    oven-mlir --version
    
    # Cleanup
    deactivate
    rm -rf "$temp_venv"
    
    print_success "Wheel installation test passed"
}

# Function to upload to Test PyPI
upload_test_pypi() {
    print_status "Uploading to Test PyPI..."
    
    cd "$PROJECT_ROOT"
    
    python -m twine upload --repository testpypi "$DIST_DIR"/*
    
    if [[ $? -eq 0 ]]; then
        print_success "Successfully uploaded to Test PyPI"
        print_status "You can install with: pip install -i https://test.pypi.org/simple/ oven-mlir"
    else
        print_error "Upload to Test PyPI failed"
        exit 1
    fi
}

# Function to upload to PyPI
upload_pypi() {
    print_warning "You are about to upload to the REAL PyPI!"
    print_warning "This action cannot be undone."
    echo
    read -p "Are you sure you want to continue? (type 'yes' to confirm): " confirm
    
    if [[ "$confirm" != "yes" ]]; then
        print_status "Upload cancelled"
        exit 0
    fi
    
    print_status "Uploading to PyPI..."
    
    cd "$PROJECT_ROOT"
    
    python -m twine upload "$DIST_DIR"/*
    
    if [[ $? -eq 0 ]]; then
        print_success "Successfully uploaded to PyPI"
        print_status "Package available at: https://pypi.org/project/oven-mlir/"
    else
        print_error "Upload to PyPI failed"
        exit 1
    fi
}

# Main execution
main() {
    print_header "oven-mlir Wheel Builder"
    
    print_status "Build configuration:"
    print_debug "  Platform: $PLATFORM ($ARCH)"
    print_debug "  Python: $PYTHON_VERSION"
    print_debug "  Project root: $PROJECT_ROOT"
    print_debug "  Clean build: $CLEAN_BUILD"
    print_debug "  Skip tests: $SKIP_TESTS"
    print_debug "  Build source: $BUILD_SOURCE"
    print_debug "  Upload test: $UPLOAD_TEST"
    print_debug "  Upload PyPI: $UPLOAD_PYPI"
    print_debug "  Force platform: ${FORCE_PLATFORM:-auto}"
    echo
    
    # Check dependencies
    check_dependencies
    
    # Clean if requested
    if [[ "$CLEAN_BUILD" == "true" ]]; then
        clean_build
    fi
    
    # Run tests unless skipped
    if [[ "$SKIP_TESTS" == "false" ]]; then
        run_tests
    fi
    
    # Build wheel
    build_wheel
    
    # Validate wheel
    validate_wheel
    
    # Test wheel installation
    test_wheel_install
    
    # Upload if requested
    if [[ "$UPLOAD_TEST" == "true" ]]; then
        upload_test_pypi
    elif [[ "$UPLOAD_PYPI" == "true" ]]; then
        upload_pypi
    fi
    
    print_success "Build process completed successfully!"
    
    # Show final status
    echo
    print_header "Build Summary"
    
    local wheel_count=$(find "$DIST_DIR" -name "*.whl" | wc -l)
    local source_count=$(find "$DIST_DIR" -name "*.tar.gz" | wc -l)
    
    echo "  📦 Files built: $((wheel_count + source_count))"
    
    if [[ $wheel_count -gt 0 ]]; then
        local wheel_size=$(du -sh "$DIST_DIR"/*.whl 2>/dev/null | cut -f1 | head -1)
        echo "  🛞 Wheel size: ${wheel_size:-N/A}"
        
        # Show platform info for wheels
        for wheel in "$DIST_DIR"/*.whl; do
            if [[ -f "$wheel" ]]; then
                local wheel_name=$(basename "$wheel")
                echo "  🎯 Platform: ${wheel_name#*-}"
                break
            fi
        done
    fi
    
    if [[ $source_count -gt 0 ]]; then
        local source_size=$(du -sh "$DIST_DIR"/*.tar.gz 2>/dev/null | cut -f1 | head -1)
        echo "  📋 Source size: ${source_size:-N/A}"
    fi
    
    echo "  📁 Location: $DIST_DIR"
    
    # Show next steps
    echo
    print_status "Next steps:"
    if [[ "$UPLOAD_TEST" == "false" && "$UPLOAD_PYPI" == "false" ]]; then
        echo "  • Test install: pip install $DIST_DIR/*.whl"
        echo "  • Upload to Test PyPI: $0 --test-upload"
        echo "  • Upload to PyPI: $0 --pypi-upload"
    fi
}

# Run main function
main "$@"