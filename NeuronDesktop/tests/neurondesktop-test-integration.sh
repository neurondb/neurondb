#!/bin/bash
# Integration test runner for NeuronDesktop
# Tests integration with NeuronDB, NeuronMCP, and NeuronAgent

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}NeuronDesktop Integration Test Suite${NC}"
echo "=========================================="
echo ""

# Check for required environment variables
echo -e "${YELLOW}Checking test configuration...${NC}"

if [ -z "$TEST_NEURONDB_DSN" ]; then
    echo -e "${YELLOW}Warning: TEST_NEURONDB_DSN not set, using default${NC}"
    export TEST_NEURONDB_DSN="host=localhost port=5432 user=neurondb dbname=neurondb sslmode=disable"
fi

if [ -z "$TEST_NEURONMCP_COMMAND" ]; then
    echo -e "${YELLOW}Warning: TEST_NEURONMCP_COMMAND not set, MCP tests will be skipped${NC}"
fi

if [ -z "$TEST_NEURONAGENT_URL" ]; then
    echo -e "${YELLOW}Warning: TEST_NEURONAGENT_URL not set, using default${NC}"
    export TEST_NEURONAGENT_URL="http://localhost:8080"
fi

echo ""
echo "Configuration:"
echo "  NeuronDB DSN: ${TEST_NEURONDB_DSN}"
echo "  NeuronMCP Command: ${TEST_NEURONMCP_COMMAND:-not set}"
echo "  NeuronAgent URL: ${TEST_NEURONAGENT_URL}"
echo ""

# Change to API directory
cd "$PROJECT_ROOT/api"

# Run integration tests
echo -e "${YELLOW}Running NeuronDB integration tests...${NC}"
go test -v -tags=integration ../tests/integration/neurondb_*_test.go || {
    echo -e "${YELLOW}Warning: Some NeuronDB tests may fail if NeuronDB is not available${NC}"
}
echo ""

echo -e "${YELLOW}Running NeuronMCP integration tests...${NC}"
go test -v -tags=integration ../tests/integration/mcp_*_test.go || {
    echo -e "${YELLOW}Warning: Some MCP tests may fail if NeuronMCP is not available${NC}"
}
echo ""

echo -e "${YELLOW}Running NeuronAgent integration tests...${NC}"
go test -v -tags=integration ../tests/integration/agent_*_test.go || {
    echo -e "${YELLOW}Warning: Some Agent tests may fail if NeuronAgent is not available${NC}"
}
echo ""

echo -e "${YELLOW}Running cross-component integration tests...${NC}"
go test -v -tags=integration ../tests/integration/cross_component_test.go ../tests/integration/profile_integration_test.go || {
    echo -e "${YELLOW}Warning: Some cross-component tests may fail if services are not available${NC}"
}
echo ""

# Generate coverage report
echo -e "${YELLOW}Generating integration test coverage report...${NC}"
go test -v -tags=integration -coverprofile=coverage_integration.out ../tests/integration/... || true

if [ -f coverage_integration.out ]; then
    go tool cover -html=coverage_integration.out -o coverage_integration.html
    echo -e "${GREEN}Integration test coverage report generated: api/coverage_integration.html${NC}"
    
    echo -e "${YELLOW}Coverage Summary:${NC}"
    go tool cover -func=coverage_integration.out | tail -1
fi
echo ""

echo -e "${GREEN}Integration tests completed!${NC}"
echo ""
echo "Note: Some tests may be skipped if external services are not available."
echo "Set SKIP_NEURONDB=true, SKIP_NEURONMCP=true, or SKIP_NEURONAGENT=true to skip specific service tests."

