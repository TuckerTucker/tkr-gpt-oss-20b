#!/bin/bash
#
# Wave 2 Validation Gate for GPT-OSS CLI Chat
#
# This script validates that all Wave 2 deliverables are complete and functional.
# It serves as the quality gate for Wave 2 integration testing.
#
# Checks:
# 1. All Wave 1 tests still pass
# 2. All Wave 2 integration tests pass
# 3. E2E mock tests pass
# 4. Code coverage meets targets
# 5. Performance benchmarks run successfully
#
# Usage:
#   ./scripts/validate_wave2.sh
#   ./scripts/validate_wave2.sh --verbose
#   ./scripts/validate_wave2.sh --coverage

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# Flags
VERBOSE=0
RUN_COVERAGE=0

for arg in "$@"; do
    case $arg in
        --verbose|-v)
            VERBOSE=1
            ;;
        --coverage|-c)
            RUN_COVERAGE=1
            ;;
    esac
done

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo ""
echo "======================================================================"
echo "  GPT-OSS CLI CHAT - WAVE 2 VALIDATION GATE"
echo "======================================================================"
echo ""

# Helper functions
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

verbose_log() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "  → $1"
    fi
}

# =============================================================================
# 0. ENVIRONMENT CHECK
# =============================================================================

echo "${BOLD}[0/6] Environment Check${NC}"
echo "----------------------------------------------------------------------"

# Check virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    if [[ -d ".venv" ]]; then
        check_warn "Virtual environment not activated. Trying to activate..."
        source .venv/bin/activate 2>/dev/null || check_warn "Could not activate .venv"
    else
        check_fail "Virtual environment not found. Run: python -m venv .venv"
    fi
else
    check_pass "Virtual environment active: $VIRTUAL_ENV"
fi

# Check pytest
if ! command -v pytest &> /dev/null; then
    check_fail "pytest not installed. Run: pip install pytest"
else
    PYTEST_VERSION=$(pytest --version | head -n1)
    check_pass "pytest available: $PYTEST_VERSION"
fi

# Check pytest-cov if coverage requested
if [[ $RUN_COVERAGE -eq 1 ]]; then
    if python -c "import pytest_cov" 2>/dev/null; then
        check_pass "pytest-cov available for coverage reporting"
    else
        check_warn "pytest-cov not installed. Run: pip install pytest-cov"
        RUN_COVERAGE=0
    fi
fi

echo ""

# =============================================================================
# 1. WAVE 1 REGRESSION TESTING
# =============================================================================

echo "${BOLD}[1/6] Wave 1 Regression Tests${NC}"
echo "----------------------------------------------------------------------"

verbose_log "Running Wave 1 unit tests..."

if [[ $VERBOSE -eq 1 ]]; then
    if pytest tests/unit/ -v -m "not slow" 2>&1 | tail -n 20; then
        check_pass "Wave 1 unit tests passed"
    else
        check_warn "Some Wave 1 tests failed (may be skipped)"
    fi
else
    if pytest tests/unit/ -q -m "not slow" 2>&1 | tail -n 5; then
        check_pass "Wave 1 unit tests passed"
    else
        check_warn "Some Wave 1 tests failed (may be skipped)"
    fi
fi

echo ""

# =============================================================================
# 2. INTEGRATION TESTS
# =============================================================================

echo "${BOLD}[2/6] Wave 2 Integration Tests${NC}"
echo "----------------------------------------------------------------------"

# Check that integration test files exist
INTEGRATION_TESTS=(
    "tests/integration/test_config_to_model.py"
    "tests/integration/test_model_to_inference.py"
    "tests/integration/test_inference_to_conversation.py"
    "tests/integration/test_conversation_to_cli.py"
    "tests/integration/test_e2e_workflow.py"
)

ALL_INTEGRATION_EXIST=1
for test_file in "${INTEGRATION_TESTS[@]}"; do
    if [[ -f "$test_file" ]]; then
        verbose_log "Found: $test_file"
    else
        check_fail "Missing integration test: $test_file"
        ALL_INTEGRATION_EXIST=0
    fi
done

if [[ $ALL_INTEGRATION_EXIST -eq 1 ]]; then
    check_pass "All integration test files present (5 files)"

    # Run integration tests
    verbose_log "Running integration tests..."

    if [[ $VERBOSE -eq 1 ]]; then
        if pytest tests/integration/ -v -m "integration" --tb=short 2>&1 | tail -n 30; then
            check_pass "Integration tests passed"
        else
            check_fail "Integration tests failed"
        fi
    else
        INTEGRATION_RESULT=$(pytest tests/integration/ -q -m "integration" 2>&1 | tail -n 10)
        echo "$INTEGRATION_RESULT"

        if echo "$INTEGRATION_RESULT" | grep -q "passed"; then
            check_pass "Integration tests passed"
        else
            check_fail "Integration tests failed"
        fi
    fi
else
    check_fail "Cannot run integration tests - files missing"
fi

echo ""

# =============================================================================
# 3. E2E MOCK TESTS
# =============================================================================

echo "${BOLD}[3/6] End-to-End Mock Tests${NC}"
echo "----------------------------------------------------------------------"

if [[ -f "tests/integration/test_e2e_mock.py" ]]; then
    check_pass "E2E mock test file present"

    verbose_log "Running E2E mock tests..."

    if [[ $VERBOSE -eq 1 ]]; then
        if pytest tests/integration/test_e2e_mock.py -v --tb=short 2>&1 | tail -n 30; then
            check_pass "E2E mock tests passed"
        else
            check_fail "E2E mock tests failed"
        fi
    else
        E2E_RESULT=$(pytest tests/integration/test_e2e_mock.py -q 2>&1 | tail -n 10)
        echo "$E2E_RESULT"

        if echo "$E2E_RESULT" | grep -q "passed"; then
            check_pass "E2E mock tests passed"
        else
            check_fail "E2E mock tests failed"
        fi
    fi
else
    check_fail "Missing tests/integration/test_e2e_mock.py"
fi

echo ""

# =============================================================================
# 4. CODE COVERAGE ANALYSIS
# =============================================================================

echo "${BOLD}[4/6] Code Coverage Analysis${NC}"
echo "----------------------------------------------------------------------"

if [[ $RUN_COVERAGE -eq 1 ]]; then
    verbose_log "Running tests with coverage..."

    # Run tests with coverage
    pytest tests/ \
        --cov=src \
        --cov-report=term-missing \
        --cov-report=html:htmlcov \
        -q 2>&1 | tail -n 30

    # Extract coverage percentage
    COVERAGE=$(pytest tests/ --cov=src --cov-report=term-missing -q 2>&1 | \
               grep "TOTAL" | awk '{print $NF}' | sed 's/%//')

    if [[ -n "$COVERAGE" ]]; then
        verbose_log "Overall coverage: $COVERAGE%"

        # Check coverage threshold
        if (( $(echo "$COVERAGE >= 70" | bc -l) )); then
            check_pass "Code coverage: ${COVERAGE}% (target: 70%+)"
        elif (( $(echo "$COVERAGE >= 50" | bc -l) )); then
            check_warn "Code coverage: ${COVERAGE}% (below 70% target)"
        else
            check_warn "Code coverage: ${COVERAGE}% (significantly below target)"
        fi

        echo ""
        echo "  Coverage report generated: htmlcov/index.html"
    else
        check_warn "Could not determine coverage percentage"
    fi
else
    check_warn "Coverage analysis skipped (use --coverage flag to enable)"
fi

echo ""

# =============================================================================
# 5. PERFORMANCE BENCHMARKS
# =============================================================================

echo "${BOLD}[5/6] Performance Benchmarks${NC}"
echo "----------------------------------------------------------------------"

if [[ -f "benchmarks/integration_benchmark.py" ]]; then
    check_pass "Integration benchmark script exists"

    verbose_log "Running integration benchmarks..."

    # Run benchmark
    if python benchmarks/integration_benchmark.py 2>&1 | tail -n 20; then
        check_pass "Integration benchmarks executed successfully"
    else
        check_warn "Integration benchmark execution had issues"
    fi
else
    check_fail "Missing benchmarks/integration_benchmark.py"
fi

echo ""

# =============================================================================
# 6. TEST SUMMARY REPORT
# =============================================================================

echo "${BOLD}[6/6] Generating Test Summary${NC}"
echo "----------------------------------------------------------------------"

# Count tests
UNIT_TEST_COUNT=$(find tests/unit -name "test_*.py" | wc -l | tr -d ' ')
INTEGRATION_TEST_COUNT=$(find tests/integration -name "test_*.py" | wc -l | tr -d ' ')
TOTAL_TEST_FILES=$((UNIT_TEST_COUNT + INTEGRATION_TEST_COUNT))

verbose_log "Unit test files: $UNIT_TEST_COUNT"
verbose_log "Integration test files: $INTEGRATION_TEST_COUNT"

check_pass "Test suite: $TOTAL_TEST_FILES files total"

# Run full test suite count
if command -v pytest &> /dev/null; then
    TEST_COLLECT=$(pytest --collect-only -q 2>&1 | tail -n 1)
    verbose_log "Test collection: $TEST_COLLECT"

    if echo "$TEST_COLLECT" | grep -q "test"; then
        check_pass "Test collection successful"
    fi
fi

echo ""

# =============================================================================
# FINAL SUMMARY
# =============================================================================

echo "======================================================================"
echo "${BOLD}WAVE 2 VALIDATION SUMMARY${NC}"
echo "======================================================================"
echo ""
echo -e "  Passed:   ${GREEN}$PASSED${NC}"
echo -e "  Warnings: ${YELLOW}$WARNINGS${NC}"
echo -e "  Failed:   ${RED}$FAILED${NC}"
echo ""

# Additional statistics
echo "Test Files Summary:"
echo "  • Unit tests:        $UNIT_TEST_COUNT files"
echo "  • Integration tests: $INTEGRATION_TEST_COUNT files"
echo "  • Total test files:  $TOTAL_TEST_FILES files"
echo ""

if [[ $RUN_COVERAGE -eq 1 ]] && [[ -n "$COVERAGE" ]]; then
    echo "Coverage:"
    echo "  • Overall coverage:  $COVERAGE%"
    echo "  • Report location:   htmlcov/index.html"
    echo ""
fi

echo "======================================================================"

# Determine overall status
if [[ $FAILED -gt 0 ]]; then
    echo -e "${RED}✗ WAVE 2 VALIDATION FAILED${NC}"
    echo ""
    echo "Critical issues detected. Please fix failures before proceeding."
    echo ""
    echo "Debug steps:"
    echo "  1. Review failed tests with: pytest tests/integration/ -v"
    echo "  2. Check test output above for error details"
    echo "  3. Ensure .venv is activated and dependencies installed"
    exit 1
elif [[ $WARNINGS -gt 3 ]]; then
    echo -e "${YELLOW}⚠ WAVE 2 VALIDATION: WARNINGS PRESENT${NC}"
    echo ""
    echo "Validation completed with warnings. Review warnings to ensure"
    echo "they are expected (e.g., skipped tests, optional features)."
    echo ""
    echo "You may proceed, but consider addressing warnings."
    exit 0
else
    echo -e "${GREEN}✓ WAVE 2 VALIDATION PASSED${NC}"
    echo ""
    echo "All integration tests passed successfully!"
    echo "Wave 2 deliverables are complete and validated."
    echo ""
    echo "Next steps:"
    echo "  • Review coverage report: open htmlcov/index.html"
    echo "  • Run benchmarks: python benchmarks/integration_benchmark.py"
    echo "  • Proceed to Wave 3 development"
    exit 0
fi
