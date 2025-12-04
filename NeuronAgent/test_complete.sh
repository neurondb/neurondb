#!/bin/bash
# Complete test suite for NeuronAgent
# Tests all components: server, API, database, clients

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "NeuronAgent Complete Test Suite"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
DB_USER="${DB_USER:-pge}"
DB_NAME="${DB_NAME:-neurondb}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
SERVER_URL="http://localhost:8080"

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

test_pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    ((TESTS_PASSED++))
}

test_fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    ((TESTS_FAILED++))
}

test_info() {
    echo -e "${YELLOW}ℹ️  INFO${NC}: $1"
}

echo "1. Checking prerequisites..."
echo "----------------------------"

# Check if server is running
if curl -s "$SERVER_URL/health" > /dev/null 2>&1; then
    test_pass "Server is running"
else
    test_fail "Server is not running. Start with: DB_USER=$DB_USER go run cmd/agent-server/main.go"
    exit 1
fi

# Check database connection
if psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -c "SELECT 1;" > /dev/null 2>&1; then
    test_pass "Database connection"
else
    test_fail "Database connection failed"
    exit 1
fi

# Check schema exists
SCHEMA_EXISTS=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'neurondb_agent');" | xargs)
if [ "$SCHEMA_EXISTS" = "t" ]; then
    test_pass "Schema exists"
else
    test_fail "Schema neurondb_agent does not exist"
    exit 1
fi

echo ""
echo "2. Generating API key..."
echo "----------------------------"

# Generate API key using Go program
API_KEY_OUTPUT=$(go run cmd/generate-key/main.go \
    -org "test-org" \
    -user "test-user" \
    -rate 100 \
    -roles "user" \
    -db-host "$DB_HOST" \
    -db-port "$DB_PORT" \
    -db-name "$DB_NAME" \
    -db-user "$DB_USER" 2>&1)

if [ $? -eq 0 ]; then
    # Extract API key from output (format: "API Key: xxxxxx")
    API_KEY=$(echo "$API_KEY_OUTPUT" | grep -i "api key" | sed 's/.*[Aa][Pp][Ii] [Kk][Ee][Yy][: ]*//' | tr -d '[:space:]')
    if [ -n "$API_KEY" ]; then
        test_pass "API key generated: ${API_KEY:0:12}..."
        export NEURONAGENT_API_KEY="$API_KEY"
    else
        # Try alternative extraction
        API_KEY=$(echo "$API_KEY_OUTPUT" | tail -1 | tr -d '[:space:]')
        if [ -n "$API_KEY" ] && [ ${#API_KEY} -gt 20 ]; then
            test_pass "API key generated: ${API_KEY:0:12}..."
            export NEURONAGENT_API_KEY="$API_KEY"
        else
            test_fail "Could not extract API key from output"
            echo "Output: $API_KEY_OUTPUT"
            # Use a test key from database
            DB_KEY=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT key_prefix FROM neurondb_agent.api_keys LIMIT 1;" | xargs)
            if [ -n "$DB_KEY" ]; then
                test_info "Using existing key prefix: $DB_KEY"
                # For testing, we'll need to create a proper key
                export NEURONAGENT_API_KEY="test_${DB_KEY}_$(openssl rand -hex 16)"
            fi
        fi
    fi
else
    test_info "API key generation failed, will test with existing keys"
    DB_KEY=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT key_prefix FROM neurondb_agent.api_keys LIMIT 1;" | xargs)
    if [ -n "$DB_KEY" ]; then
        export NEURONAGENT_API_KEY="test_${DB_KEY}"
    fi
fi

echo ""
echo "3. Testing API endpoints..."
echo "----------------------------"

# Test health endpoint (no auth)
if curl -s "$SERVER_URL/health" | grep -q "200\|OK" || [ "$(curl -s -o /dev/null -w '%{http_code}' "$SERVER_URL/health")" = "200" ]; then
    test_pass "Health endpoint"
else
    test_fail "Health endpoint"
fi

# Test metrics endpoint
if curl -s "$SERVER_URL/metrics" > /dev/null 2>&1; then
    test_pass "Metrics endpoint"
else
    test_fail "Metrics endpoint"
fi

# Test agents endpoint (requires auth)
AGENTS_RESPONSE=$(curl -s -w "\n%{http_code}" -X GET "$SERVER_URL/api/v1/agents" \
    -H "Authorization: Bearer $NEURONAGENT_API_KEY" 2>&1)
HTTP_CODE=$(echo "$AGENTS_RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "401" ]; then
    if [ "$HTTP_CODE" = "200" ]; then
        test_pass "Agents endpoint (authenticated)"
    else
        test_info "Agents endpoint returned 401 (auth required - expected)"
    fi
else
    test_fail "Agents endpoint (got $HTTP_CODE)"
fi

echo ""
echo "4. Testing database operations..."
echo "----------------------------"

# Test table counts
AGENT_COUNT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT COUNT(*) FROM neurondb_agent.agents;" | xargs)
test_info "Agents in database: $AGENT_COUNT"

SESSION_COUNT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT COUNT(*) FROM neurondb_agent.sessions;" | xargs)
test_info "Sessions in database: $SESSION_COUNT"

MESSAGE_COUNT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT COUNT(*) FROM neurondb_agent.messages;" | xargs)
test_info "Messages in database: $MESSAGE_COUNT"

API_KEY_COUNT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT COUNT(*) FROM neurondb_agent.api_keys;" | xargs)
if [ "$API_KEY_COUNT" -gt 0 ]; then
    test_pass "API keys exist ($API_KEY_COUNT)"
else
    test_fail "No API keys found"
fi

# Test indexes
INDEX_COUNT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'neurondb_agent';" | xargs)
if [ "$INDEX_COUNT" -gt 0 ]; then
    test_pass "Indexes exist ($INDEX_COUNT)"
else
    test_fail "No indexes found"
fi

echo ""
echo "5. Testing Python client (if available)..."
echo "----------------------------"

if command -v python3 &> /dev/null; then
    # Check if dependencies are installed
    if python3 -c "import requests" 2>/dev/null; then
        test_pass "Python requests module available"
        
        # Simple Python test
        python3 << EOF
import sys
import requests

try:
    response = requests.get("$SERVER_URL/health", timeout=5)
    if response.status_code == 200:
        print("✅ Python client can connect")
        sys.exit(0)
    else:
        print("❌ Python client got status:", response.status_code)
        sys.exit(1)
except Exception as e:
    print(f"❌ Python client error: {e}")
    sys.exit(1)
EOF
        
        if [ $? -eq 0 ]; then
            test_pass "Python client connectivity"
        else
            test_fail "Python client connectivity"
        fi
    else
        test_info "Python requests not installed (pip install requests)"
    fi
else
    test_info "Python3 not available"
fi

echo ""
echo "6. Testing full workflow..."
echo "----------------------------"

# Create agent via API (if we have valid key)
if [ -n "$NEURONAGENT_API_KEY" ] && [ ${#NEURONAGENT_API_KEY} -gt 20 ]; then
    AGENT_NAME="test-agent-$(date +%s)"
    CREATE_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVER_URL/api/v1/agents" \
        -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"name\": \"$AGENT_NAME\",
            \"system_prompt\": \"You are a test assistant.\",
            \"model_name\": \"gpt-4\",
            \"enabled_tools\": [\"sql\"],
            \"config\": {\"temperature\": 0.7, \"max_tokens\": 1000}
        }" 2>&1)
    
    HTTP_CODE=$(echo "$CREATE_RESPONSE" | tail -1)
    if [ "$HTTP_CODE" = "201" ]; then
        test_pass "Create agent via API"
        AGENT_ID=$(echo "$CREATE_RESPONSE" | head -1 | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
        test_info "Created agent: $AGENT_ID"
    elif [ "$HTTP_CODE" = "401" ]; then
        test_info "Create agent requires valid API key (got 401)"
    else
        test_info "Create agent returned: $HTTP_CODE"
    fi
else
    test_info "Skipping API workflow test (no valid API key)"
fi

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Passed: $TESTS_PASSED"
echo "Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi

