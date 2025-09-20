#!/bin/bash
# Continuous Integration script for oven-mlir
# This script runs all checks and tests for CI/CD pipelines
# Usage: ./scripts/ci.sh

set -e

echo "🚀 Running CI/CD pipeline for oven-mlir..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "SUCCESS")
            echo -e "${GREEN}✅ $message${NC}"
            ;;
        "FAILURE")
            echo -e "${RED}❌ $message${NC}"
            ;;
        "INFO")
            echo -e "${YELLOW}ℹ️ $message${NC}"
            ;;
    esac
}

# Function to run a command and check its status
run_check() {
    local name=$1
    local command=$2
    
    print_status "INFO" "Running $name..."
    
    if eval "$command"; then
        print_status "SUCCESS" "$name passed"
        return 0
    else
        print_status "FAILURE" "$name failed"
        return 1
    fi
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_status "INFO" "Setting up virtual environment..."
    ./scripts/setup_venv.sh
fi

# Activate virtual environment
source venv/bin/activate

# Initialize counters
total_checks=0
failed_checks=0

# Code formatting check
total_checks=$((total_checks + 1))
if ! run_check "Code formatting (black)" "black oven_mlir/ tests/ --check --line-length 88"; then
    failed_checks=$((failed_checks + 1))
fi

# Import sorting check
total_checks=$((total_checks + 1))
if ! run_check "Import sorting (isort)" "isort oven_mlir/ tests/ --check-only --profile black"; then
    failed_checks=$((failed_checks + 1))
fi

# Linting check
total_checks=$((total_checks + 1))
if ! run_check "Linting (flake8)" "flake8 oven_mlir/ tests/ --max-line-length=88 --ignore=E203,W503"; then
    failed_checks=$((failed_checks + 1))
fi

# Build check
total_checks=$((total_checks + 1))
if ! run_check "Package build" "python setup.py build"; then
    failed_checks=$((failed_checks + 1))
fi

# Test execution
total_checks=$((total_checks + 1))
if ! run_check "Unit tests" "python -m pytest"; then
    failed_checks=$((failed_checks + 1))
fi

# Import check
total_checks=$((total_checks + 1))
if ! run_check "Package import" "python -c 'import oven_mlir; print(oven_mlir.__version__)'"; then
    failed_checks=$((failed_checks + 1))
fi

# CLI check
total_checks=$((total_checks + 1))
if ! run_check "CLI functionality" "oven-compile --help > /dev/null"; then
    failed_checks=$((failed_checks + 1))
fi

# Summary
echo ""
echo "📊 CI/CD Summary:"
echo "   Total checks: $total_checks"
echo "   Passed: $((total_checks - failed_checks))"
echo "   Failed: $failed_checks"
echo ""

if [ $failed_checks -eq 0 ]; then
    print_status "SUCCESS" "All CI/CD checks passed! 🎉"
    exit 0
else
    print_status "FAILURE" "$failed_checks check(s) failed"
    exit 1
fi