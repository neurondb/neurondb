#!/bin/bash
# Master test runner for NeuronDesktop

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}NeuronDesktop Test Suite${NC}"
echo "================================"
echo ""

# Setup test environment
echo -e "${YELLOW}Setting up test environment...${NC}"
"$SCRIPT_DIR/setup_test_env.sh" || {
    echo -e "${RED}Failed to setup test environment${NC}"
    exit 1
}
echo ""

# Change to API directory
cd "$PROJECT_ROOT/api"

# Run tests with coverage
echo -e "${YELLOW}Running backend unit tests...${NC}"
go test -v -coverprofile=coverage.out ./internal/handlers/... ./internal/db/... || {
    echo -e "${RED}Unit tests failed${NC}"
    exit 1
}
echo ""

echo -e "${YELLOW}Running integration tests...${NC}"
go test -v -tags=integration -coverprofile=coverage_integration.out ../tests/integration/... || {
    echo -e "${YELLOW}Warning: Some integration tests may fail if external services are not available${NC}"
}
echo ""

echo -e "${YELLOW}Running end-to-end tests...${NC}"
go test -v -coverprofile=coverage_e2e.out ../../tests/e2e/... || {
    echo -e "${YELLOW}Warning: Some E2E tests may fail if external services are not available${NC}"
}
echo ""

# Generate combined coverage report
echo -e "${YELLOW}Generating coverage report...${NC}"
if command -v gocovmerge &> /dev/null; then
    gocovmerge coverage.out coverage_integration.out coverage_e2e.out > coverage_combined.out 2>/dev/null || {
        echo -e "${YELLOW}Warning: Could not merge coverage reports${NC}"
    }
    
    if [ -f coverage_combined.out ]; then
        go tool cover -html=coverage_combined.out -o coverage.html
        echo -e "${GREEN}Coverage report generated: api/coverage.html${NC}"
    fi
else
    echo -e "${YELLOW}Warning: gocovmerge not found. Install with: go install github.com/wadey/gocovmerge@latest${NC}"
    if [ -f coverage.out ]; then
        go tool cover -html=coverage.out -o coverage.html
        echo -e "${GREEN}Coverage report generated: api/coverage.html${NC}"
    fi
fi
echo ""

# Show coverage summary
if [ -f coverage.out ]; then
    echo -e "${YELLOW}Coverage Summary:${NC}"
    go tool cover -func=coverage.out | tail -1
    echo ""
fi

echo -e "${GREEN}All tests completed!${NC}"

# Cleanup (optional)
if [ "${CLEANUP_AFTER_TESTS:-false}" = "true" ]; then
    echo -e "${YELLOW}Cleaning up...${NC}"
    "$SCRIPT_DIR/cleanup_test_env.sh"
fi








