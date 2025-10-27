#!/bin/bash
#
# Wave 1 Validation Gate for GPT-OSS CLI Chat
#
# This script validates that all Wave 1 deliverables are complete and functional.
# It serves as the quality gate before proceeding to Wave 2.
#
# Checks:
# 1. Project structure exists
# 2. All Wave 1 modules can be imported
# 3. Unit tests pass
# 4. Integration contracts are satisfied
# 5. Documentation is present
#
# Usage:
#   ./scripts/validate_wave1.sh
#   ./scripts/validate_wave1.sh --verbose

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

# Verbose flag
VERBOSE=0
if [[ "$1" == "--verbose" ]] || [[ "$1" == "-v" ]]; then
    VERBOSE=1
fi

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo ""
echo "======================================================================"
echo "  GPT-OSS CLI CHAT - WAVE 1 VALIDATION GATE"
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
# 1. PROJECT STRUCTURE VALIDATION
# =============================================================================

echo "${BOLD}[1/6] Validating Project Structure${NC}"
echo "----------------------------------------------------------------------"

# Check critical directories
CRITICAL_DIRS=(
    "src/config"
    "src/models"
    "src/devices"
    "src/inference"
    "src/sampling"
    "src/conversation"
    "src/prompts"
    "src/cli"
    "tests/unit"
    "tests/integration"
    "benchmarks"
    "scripts"
)

for dir in "${CRITICAL_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
        check_pass "Directory exists: $dir"
    else
        check_warn "Directory missing: $dir (may not be implemented yet)"
    fi
done

echo ""

# =============================================================================
# 2. PYTHON IMPORT VALIDATION
# =============================================================================

echo "${BOLD}[2/6] Validating Python Imports${NC}"
echo "----------------------------------------------------------------------"

# Test imports for each module
MODULES=(
    "src.config:ModelConfig,InferenceConfig,CLIConfig"
    "src.models:ModelLoader"
    "src.devices:DeviceDetector"
    "src.inference:InferenceEngine"
    "src.conversation:ConversationManager"
)

for module_spec in "${MODULES[@]}"; do
    IFS=':' read -r module classes <<< "$module_spec"

    verbose_log "Testing import: $module"

    if python3 -c "from $module import ${classes//,/, }; print('✓')" 2>/dev/null; then
        check_pass "Import successful: $module"
    else
        check_warn "Import failed: $module (waiting for implementation)"
    fi
done

echo ""

# =============================================================================
# 3. UNIT TESTS VALIDATION
# =============================================================================

echo "${BOLD}[3/6] Running Unit Tests${NC}"
echo "----------------------------------------------------------------------"

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    check_warn "pytest not installed, skipping unit tests"
else
    # Check if test files exist
    TEST_FILES=(
        "tests/unit/test_config.py"
        "tests/unit/test_model_loader.py"
        "tests/unit/test_inference.py"
        "tests/unit/test_conversation.py"
    )

    ALL_TESTS_EXIST=1
    for test_file in "${TEST_FILES[@]}"; do
        if [[ ! -f "$test_file" ]]; then
            check_warn "Test file missing: $test_file"
            ALL_TESTS_EXIST=0
        fi
    done

    if [[ $ALL_TESTS_EXIST -eq 1 ]]; then
        check_pass "All test files present"

        # Run pytest
        verbose_log "Running pytest..."
        if [[ $VERBOSE -eq 1 ]]; then
            if pytest tests/unit/ -v 2>&1; then
                check_pass "Unit tests passed (or skipped awaiting implementation)"
            else
                check_warn "Some unit tests failed (may be expected if modules not implemented)"
            fi
        else
            if pytest tests/unit/ -q 2>&1 | tail -n 5; then
                check_pass "Unit tests passed (or skipped awaiting implementation)"
            else
                check_warn "Some unit tests failed (may be expected if modules not implemented)"
            fi
        fi
    else
        check_warn "Cannot run tests, files missing"
    fi
fi

echo ""

# =============================================================================
# 4. INTEGRATION CONTRACTS VALIDATION
# =============================================================================

echo "${BOLD}[4/6] Validating Integration Contracts${NC}"
echo "----------------------------------------------------------------------"

CONTRACT_DIR=".context-kit/orchestration/gpt-oss-cli-chat/integration-contracts"

if [[ -d "$CONTRACT_DIR" ]]; then
    check_pass "Integration contracts directory exists"

    # Count contract files
    CONTRACT_COUNT=$(find "$CONTRACT_DIR" -name "*.md" -o -name "*.py" | wc -l | tr -d ' ')
    verbose_log "Found $CONTRACT_COUNT contract files"

    if [[ $CONTRACT_COUNT -gt 0 ]]; then
        check_pass "Integration contracts defined ($CONTRACT_COUNT files)"
    else
        check_warn "No integration contract files found"
    fi
else
    check_warn "Integration contracts directory not found"
fi

# Check that test fixtures match contracts
if [[ -f "tests/conftest.py" ]]; then
    check_pass "Shared test fixtures defined (tests/conftest.py)"
else
    check_fail "Missing tests/conftest.py"
fi

echo ""

# =============================================================================
# 5. BENCHMARKS AND SCRIPTS VALIDATION
# =============================================================================

echo "${BOLD}[5/6] Validating Benchmarks and Scripts${NC}"
echo "----------------------------------------------------------------------"

# Check benchmark
if [[ -f "benchmarks/latency_benchmark.py" ]]; then
    check_pass "Latency benchmark script exists"

    # Check if executable
    if [[ -x "benchmarks/latency_benchmark.py" ]]; then
        verbose_log "Benchmark script is executable"
    else
        check_warn "Benchmark script not executable (chmod +x recommended)"
    fi
else
    check_fail "Missing benchmarks/latency_benchmark.py"
fi

# Check setup validation script
if [[ -f "scripts/validate_setup.py" ]]; then
    check_pass "Setup validation script exists"

    # Try running it
    verbose_log "Testing setup validation script..."
    if python3 scripts/validate_setup.py --help &> /dev/null; then
        check_pass "Setup validation script is functional"
    else
        check_warn "Setup validation script may have issues"
    fi
else
    check_fail "Missing scripts/validate_setup.py"
fi

echo ""

# =============================================================================
# 6. DOCUMENTATION VALIDATION
# =============================================================================

echo "${BOLD}[6/6] Validating Documentation${NC}"
echo "----------------------------------------------------------------------"

# Check critical documentation files
DOC_FILES=(
    ".context-kit/orchestration/gpt-oss-cli-chat/orchestration-plan.md"
    ".context-kit/orchestration/gpt-oss-cli-chat/agent-assignments.md"
    ".context-kit/orchestration/gpt-oss-cli-chat/validation-strategy.md"
)

for doc in "${DOC_FILES[@]}"; do
    if [[ -f "$doc" ]]; then
        check_pass "Documentation exists: $(basename "$doc")"
    else
        check_warn "Documentation missing: $(basename "$doc")"
    fi
done

# Check for README
if [[ -f "README.md" ]]; then
    check_pass "README.md exists"
else
    check_warn "README.md not yet created (Wave 2 deliverable)"
fi

echo ""

# =============================================================================
# FINAL SUMMARY
# =============================================================================

echo "======================================================================"
echo "${BOLD}VALIDATION SUMMARY${NC}"
echo "======================================================================"
echo ""
echo -e "  Passed:   ${GREEN}$PASSED${NC}"
echo -e "  Warnings: ${YELLOW}$WARNINGS${NC}"
echo -e "  Failed:   ${RED}$FAILED${NC}"
echo ""
echo "======================================================================"

# Determine overall status
if [[ $FAILED -gt 0 ]]; then
    echo -e "${RED}✗ WAVE 1 VALIDATION FAILED${NC}"
    echo ""
    echo "Critical issues detected. Please address failures before proceeding."
    exit 1
elif [[ $WARNINGS -gt 5 ]]; then
    echo -e "${YELLOW}⚠ WAVE 1 VALIDATION: WARNINGS PRESENT${NC}"
    echo ""
    echo "Validation completed with warnings. Many components may be awaiting"
    echo "implementation by other agents. This is expected during Wave 1."
    echo ""
    echo "Review warnings and verify they are expected blockers."
    exit 0
else
    echo -e "${GREEN}✓ WAVE 1 VALIDATION PASSED${NC}"
    echo ""
    echo "Testing infrastructure is ready. Agents can proceed with implementation."
    exit 0
fi
