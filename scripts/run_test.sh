#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

echo "Running Oven MLIR Tests..."
echo "=========================="

# Function to run a single test
run_test() {
    local test_file=$1
    local test_name=$(basename "$test_file" .mlir)
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Testing $test_name... "
    
    # Check if the file has CHECK labels (indicating it should use FileCheck)
    if grep -q "// CHECK" "$test_file"; then
        # Use FileCheck for verification
        if ./build-ninja/tools/oven-opt "$test_file" --oven-to-llvm | ./llvm-project/build/bin/FileCheck "$test_file" > /dev/null 2>&1; then
            echo -e "${GREEN}PASS${NC}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
            return 0
        else
            echo -e "${RED}FAIL${NC}"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            echo "  FileCheck verification failed:"
            ./build-ninja/tools/oven-opt "$test_file" --oven-to-llvm | ./llvm-project/build/bin/FileCheck "$test_file" 2>&1 | sed 's/^/    /'
            return 1
        fi
    else
        # Fallback to simple compilation check
        if ./build-ninja/tools/oven-opt "$test_file" --oven-to-llvm > /dev/null 2>&1; then
            echo -e "${GREEN}PASS${NC}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
            return 0
        else
            echo -e "${RED}FAIL${NC}"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            echo "  Error details:"
            ./build-ninja/tools/oven-opt "$test_file" --oven-to-llvm 2>&1 | sed 's/^/    /'
            return 1
        fi
    fi
}

# Test basic functionality with oven.mlir
echo "=== Basic MLIR Compilation Tests ==="
run_test "./tests/oven.mlir"
run_test "./tests/sigmoid.mlir"

# Test more complex functionality
echo "=== Advanced MLIR Tests ==="
run_test "./tests/smem.mlir"
run_test "./tests/matmul.mlir"

# Test complex loops and shared memory
echo "=== Complex Kernel Tests ==="
if [ -f "./tests/matmul_block.mlir" ]; then
    run_test "./tests/matmul_block.mlir"
fi

# Test full compilation pipeline
echo "=== Full Compilation Pipeline Tests ==="
echo -n "Testing full compilation pipeline... "
if ./scripts/compile.sh ./tests/sigmoid.mlir > /dev/null 2>&1; then
    echo -e "${GREEN}PASS${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
else
    echo -e "${RED}FAIL${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo "  Full pipeline error details:"
    ./scripts/compile.sh ./tests/sigmoid.mlir 2>&1 | sed 's/^/    /'
fi

# Summary
echo ""
echo "=========================="
echo "Test Summary:"
echo "  Total tests: $TOTAL_TESTS"
echo -e "  Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "  Failed: ${RED}$FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
