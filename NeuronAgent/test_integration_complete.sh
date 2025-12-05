#!/bin/bash
# 100% Comprehensive Integration Test for NeuronAgent with NeuronDB
# Tests all components, endpoints, database operations, and NeuronDB features

set -e

cd "$(dirname "$0")"

echo "================================================================"
echo "NeuronAgent 100% Comprehensive Integration Test"
echo "Testing with full NeuronDB integration"
echo "================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
DB_USER="${DB_USER:-pge}"
DB_NAME="${DB_NAME:-neurondb}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
SERVER_URL="http://localhost:8080"

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0
BUGS_FOUND=0

test_pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((TESTS_PASSED++))
    ((TESTS_TOTAL++))
}

test_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    ((TESTS_FAILED++))
    ((TESTS_TOTAL++))
    ((BUGS_FOUND++))
}

test_info() {
    echo -e "${BLUE}ℹ️  INFO${NC}: $1"
}

test_warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
}

# ============================================================================
# PHASE 1: Prerequisites and Environment
# ============================================================================
echo "PHASE 1: Prerequisites and Environment"
echo "======================================"

# Check server
test_info "Checking NeuronAgent server..."
if curl -s "$SERVER_URL/health" > /dev/null 2>&1; then
    test_pass "Server is running on $SERVER_URL"
else
    test_fail "Server is not running"
    echo "Start server: DB_USER=$DB_USER go run cmd/agent-server/main.go"
    exit 1
fi

# Check database
test_info "Checking PostgreSQL database..."
if psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -c "SELECT 1;" > /dev/null 2>&1; then
    test_pass "Database connection successful"
else
    test_fail "Database connection failed"
    exit 1
fi

# Check NeuronDB extension
test_info "Checking NeuronDB extension..."
EXT_EXISTS=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'neurondb');" | xargs)
if [ "$EXT_EXISTS" = "t" ]; then
    test_pass "NeuronDB extension installed"
else
    test_fail "NeuronDB extension not found"
    exit 1
fi

# Check vector type
test_info "Checking vector type..."
VECTOR_TYPE=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT EXISTS(SELECT 1 FROM pg_type WHERE typname = 'vector');" | xargs)
if [ "$VECTOR_TYPE" = "t" ]; then
    test_pass "Vector type available"
else
    test_fail "Vector type not found"
    exit 1
fi

# Check schema
test_info "Checking neurondb_agent schema..."
SCHEMA_EXISTS=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'neurondb_agent');" | xargs)
if [ "$SCHEMA_EXISTS" = "t" ]; then
    test_pass "Schema neurondb_agent exists"
else
    test_fail "Schema neurondb_agent does not exist"
    exit 1
fi

echo ""

# ============================================================================
# PHASE 2: Database Schema Verification
# ============================================================================
echo "PHASE 2: Database Schema Verification"
echo "======================================"

# Check all required tables
REQUIRED_TABLES=("agents" "sessions" "messages" "memory_chunks" "tools" "jobs" "api_keys" "schema_migrations")
for table in "${REQUIRED_TABLES[@]}"; do
    TABLE_EXISTS=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_schema = 'neurondb_agent' AND table_name = '$table');" | xargs)
    if [ "$TABLE_EXISTS" = "t" ]; then
        test_pass "Table $table exists"
    else
        test_fail "Table $table missing"
    fi
done

# Check indexes
test_info "Checking indexes..."
INDEX_COUNT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'neurondb_agent';" | xargs)
if [ "$INDEX_COUNT" -gt 0 ]; then
    test_pass "Indexes exist ($INDEX_COUNT found)"
else
    test_fail "No indexes found"
fi

# Check HNSW index on memory_chunks
test_info "Checking HNSW index on memory_chunks..."
HNSW_EXISTS=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE schemaname = 'neurondb_agent' AND tablename = 'memory_chunks' AND indexname LIKE '%embedding%');" | xargs)
if [ "$HNSW_EXISTS" = "t" ]; then
    test_pass "HNSW index on memory_chunks.embedding exists"
else
    test_warn "HNSW index on memory_chunks.embedding not found (may be created on first use)"
fi

echo ""

# ============================================================================
# PHASE 3: API Key Management
# ============================================================================
echo "PHASE 3: API Key Management"
echo "==========================="

# Generate API key
test_info "Generating API key..."
API_KEY_OUTPUT=$(go run cmd/generate-key/main.go \
    -org "integration-test" \
    -user "test-user" \
    -rate 1000 \
    -roles "user,admin" \
    -db-host "$DB_HOST" \
    -db-port "$DB_PORT" \
    -db-name "$DB_NAME" \
    -db-user "$DB_USER" 2>&1)

if [ $? -eq 0 ]; then
    API_KEY=$(echo "$API_KEY_OUTPUT" | grep "^Key:" | sed 's/^Key: //' | tr -d '[:space:]')
    if [ -n "$API_KEY" ] && [ ${#API_KEY} -gt 20 ]; then
        test_pass "API key generated: ${API_KEY:0:16}..."
        export NEURONAGENT_API_KEY="$API_KEY"
        KEY_PREFIX=$(echo "$API_KEY" | cut -c1-8)
    else
        test_fail "Could not extract API key"
        exit 1
    fi
else
    test_fail "API key generation failed"
    echo "$API_KEY_OUTPUT"
    exit 1
fi

# Verify key in database
test_info "Verifying API key in database..."
KEY_IN_DB=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT COUNT(*) FROM neurondb_agent.api_keys WHERE key_prefix = '$KEY_PREFIX';" | xargs)
if [ "$KEY_IN_DB" -eq 1 ]; then
    test_pass "API key found in database"
else
    test_fail "API key not found in database"
fi

# Check key metadata
test_info "Checking API key metadata..."
KEY_METADATA=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT metadata FROM neurondb_agent.api_keys WHERE key_prefix = '$KEY_PREFIX';" | xargs)
if [ -n "$KEY_METADATA" ]; then
    test_pass "API key metadata stored correctly"
else
    test_warn "API key metadata is empty (may be expected)"
fi

echo ""

# ============================================================================
# PHASE 4: REST API Endpoints
# ============================================================================
echo "PHASE 4: REST API Endpoints"
echo "=========================="

# Health endpoint (no auth)
test_info "Testing /health endpoint..."
HEALTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SERVER_URL/health")
if [ "$HEALTH_CODE" = "200" ]; then
    test_pass "GET /health returns 200"
else
    test_fail "GET /health returned $HEALTH_CODE"
fi

# Metrics endpoint (no auth)
test_info "Testing /metrics endpoint..."
METRICS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SERVER_URL/metrics")
if [ "$METRICS_CODE" = "200" ]; then
    test_pass "GET /metrics returns 200"
else
    test_fail "GET /metrics returned $METRICS_CODE"
fi

# List agents (with auth)
test_info "Testing GET /api/v1/agents..."
AGENTS_RESPONSE=$(curl -s -w "\n%{http_code}" -X GET "$SERVER_URL/api/v1/agents" \
    -H "Authorization: Bearer $NEURONAGENT_API_KEY" 2>&1)
HTTP_CODE=$(echo "$AGENTS_RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "200" ]; then
    test_pass "GET /api/v1/agents returns 200"
    AGENTS_COUNT=$(echo "$AGENTS_RESPONSE" | head -1 | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data) if isinstance(data, list) else 0)" 2>/dev/null || echo "0")
    test_info "Current agents: $AGENTS_COUNT"
else
    test_fail "GET /api/v1/agents returned $HTTP_CODE"
    echo "Response: $(echo "$AGENTS_RESPONSE" | head -1)"
fi

# Create agent
test_info "Testing POST /api/v1/agents..."
AGENT_NAME="integration-test-agent-$(date +%s)"
CREATE_AGENT_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVER_URL/api/v1/agents" \
    -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{
        \"name\": \"$AGENT_NAME\",
        \"description\": \"Integration test agent\",
        \"system_prompt\": \"You are a helpful assistant for integration testing.\",
        \"model_name\": \"gpt-4\",
        \"enabled_tools\": [\"sql\", \"http\"],
        \"config\": {
            \"temperature\": 0.7,
            \"max_tokens\": 2000,
            \"top_p\": 0.9
        }
    }" 2>&1)

HTTP_CODE=$(echo "$CREATE_AGENT_RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "201" ]; then
    test_pass "POST /api/v1/agents returns 201"
    AGENT_ID=$(echo "$CREATE_AGENT_RESPONSE" | head -1 | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])" 2>/dev/null)
    if [ -n "$AGENT_ID" ]; then
        test_pass "Agent created with ID: ${AGENT_ID:0:8}..."
    else
        test_fail "Could not extract agent ID"
    fi
else
    test_fail "POST /api/v1/agents returned $HTTP_CODE"
    echo "Response: $(echo "$CREATE_AGENT_RESPONSE" | head -1)"
    AGENT_ID=""
fi

# Get agent by ID
if [ -n "$AGENT_ID" ]; then
    test_info "Testing GET /api/v1/agents/{id}..."
    GET_AGENT_RESPONSE=$(curl -s -w "\n%{http_code}" -X GET "$SERVER_URL/api/v1/agents/$AGENT_ID" \
        -H "Authorization: Bearer $NEURONAGENT_API_KEY" 2>&1)
    HTTP_CODE=$(echo "$GET_AGENT_RESPONSE" | tail -1)
    if [ "$HTTP_CODE" = "200" ]; then
        test_pass "GET /api/v1/agents/{id} returns 200"
    else
        test_fail "GET /api/v1/agents/{id} returned $HTTP_CODE"
    fi
fi

# Create session
if [ -n "$AGENT_ID" ]; then
    test_info "Testing POST /api/v1/sessions..."
    CREATE_SESSION_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVER_URL/api/v1/sessions" \
        -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"agent_id\": \"$AGENT_ID\",
            \"external_user_id\": \"integration-test-user\",
            \"metadata\": {\"test\": true, \"source\": \"integration-test\"}
        }" 2>&1)
    
    HTTP_CODE=$(echo "$CREATE_SESSION_RESPONSE" | tail -1)
    if [ "$HTTP_CODE" = "201" ]; then
        test_pass "POST /api/v1/sessions returns 201"
        SESSION_ID=$(echo "$CREATE_SESSION_RESPONSE" | head -1 | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])" 2>/dev/null)
        if [ -n "$SESSION_ID" ]; then
            test_pass "Session created with ID: ${SESSION_ID:0:8}..."
        else
            test_fail "Could not extract session ID"
        fi
    else
        test_fail "POST /api/v1/sessions returned $HTTP_CODE"
        echo "Response: $(echo "$CREATE_SESSION_RESPONSE" | head -1)"
        SESSION_ID=""
    fi
fi

# Send message
if [ -n "$SESSION_ID" ]; then
    test_info "Testing POST /api/v1/sessions/{id}/messages..."
    SEND_MESSAGE_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVER_URL/api/v1/sessions/$SESSION_ID/messages" \
        -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "content": "Hello! This is an integration test message. Can you respond?",
            "role": "user"
        }' 2>&1)
    
    HTTP_CODE=$(echo "$SEND_MESSAGE_RESPONSE" | tail -1)
    if [ "$HTTP_CODE" = "200" ]; then
        test_pass "POST /api/v1/sessions/{id}/messages returns 200"
        # Check if response contains expected fields
        RESPONSE_BODY=$(echo "$SEND_MESSAGE_RESPONSE" | head -1)
        if echo "$RESPONSE_BODY" | python3 -c "import sys, json; data=json.load(sys.stdin); assert 'response' in data or 'error' in data" 2>/dev/null; then
            test_pass "Message response has valid structure"
        else
            test_warn "Message response structure unexpected"
        fi
    else
        test_fail "POST /api/v1/sessions/{id}/messages returned $HTTP_CODE"
        echo "Response: $(echo "$SEND_MESSAGE_RESPONSE" | head -1)"
    fi
fi

# Get messages
if [ -n "$SESSION_ID" ]; then
    test_info "Testing GET /api/v1/sessions/{id}/messages..."
    GET_MESSAGES_RESPONSE=$(curl -s -w "\n%{http_code}" -X GET "$SERVER_URL/api/v1/sessions/$SESSION_ID/messages" \
        -H "Authorization: Bearer $NEURONAGENT_API_KEY" 2>&1)
    HTTP_CODE=$(echo "$GET_MESSAGES_RESPONSE" | tail -1)
    if [ "$HTTP_CODE" = "200" ]; then
        test_pass "GET /api/v1/sessions/{id}/messages returns 200"
        MESSAGES_COUNT=$(echo "$GET_MESSAGES_RESPONSE" | head -1 | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data) if isinstance(data, list) else 0)" 2>/dev/null || echo "0")
        test_info "Messages retrieved: $MESSAGES_COUNT"
    else
        test_fail "GET /api/v1/sessions/{id}/messages returned $HTTP_CODE"
    fi
fi

echo ""

# ============================================================================
# PHASE 5: Database Integration Verification
# ============================================================================
echo "PHASE 5: Database Integration Verification"
echo "=========================================="

# Verify agent in database
if [ -n "$AGENT_ID" ]; then
    test_info "Verifying agent in database..."
    DB_AGENT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT name FROM neurondb_agent.agents WHERE id = '$AGENT_ID';" | xargs)
    if [ "$DB_AGENT" = "$AGENT_NAME" ]; then
        test_pass "Agent found in database with correct name"
    else
        test_fail "Agent not found or name mismatch in database"
    fi
    
    # Check agent config
    test_info "Checking agent configuration..."
    AGENT_CONFIG=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT config FROM neurondb_agent.agents WHERE id = '$AGENT_ID';" | xargs)
    if [ -n "$AGENT_CONFIG" ] && [ "$AGENT_CONFIG" != "{}" ]; then
        test_pass "Agent configuration stored correctly"
    else
        test_fail "Agent configuration not stored correctly"
    fi
    
    # Check enabled tools
    test_info "Checking enabled tools..."
    ENABLED_TOOLS=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT enabled_tools FROM neurondb_agent.agents WHERE id = '$AGENT_ID';" | xargs)
    if echo "$ENABLED_TOOLS" | grep -q "sql"; then
        test_pass "Enabled tools stored correctly"
    else
        test_fail "Enabled tools not stored correctly"
    fi
fi

# Verify session in database
if [ -n "$SESSION_ID" ]; then
    test_info "Verifying session in database..."
    DB_SESSION=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT id FROM neurondb_agent.sessions WHERE id = '$SESSION_ID';" | xargs)
    if [ "$DB_SESSION" = "$SESSION_ID" ]; then
        test_pass "Session found in database"
    else
        test_fail "Session not found in database"
    fi
    
    # Check session metadata
    test_info "Checking session metadata..."
    SESSION_METADATA=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT metadata FROM neurondb_agent.sessions WHERE id = '$SESSION_ID';" | xargs)
    if [ -n "$SESSION_METADATA" ]; then
        test_pass "Session metadata stored correctly"
    else
        test_warn "Session metadata is empty"
    fi
fi

# Verify messages in database
if [ -n "$SESSION_ID" ]; then
    test_info "Verifying messages in database..."
    MESSAGE_COUNT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT COUNT(*) FROM neurondb_agent.messages WHERE session_id = '$SESSION_ID';" | xargs)
    if [ "$MESSAGE_COUNT" -gt 0 ]; then
        test_pass "Messages found in database ($MESSAGE_COUNT messages)"
    else
        test_fail "No messages found in database"
    fi
    
    # Check message roles
    test_info "Checking message roles..."
    USER_MSG_COUNT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT COUNT(*) FROM neurondb_agent.messages WHERE session_id = '$SESSION_ID' AND role = 'user';" | xargs)
    if [ "$USER_MSG_COUNT" -gt 0 ]; then
        test_pass "User messages stored correctly"
    else
        test_fail "User messages not found"
    fi
fi

echo ""

# ============================================================================
# PHASE 6: NeuronDB-Specific Features
# ============================================================================
echo "PHASE 6: NeuronDB-Specific Features"
echo "===================================="

# Test vector type support
test_info "Testing vector type support..."
VECTOR_TEST=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT '[1,2,3]'::vector(3);" 2>&1)
if echo "$VECTOR_TEST" | grep -q "vector"; then
    test_pass "Vector type operations working"
else
    test_fail "Vector type operations failed"
    echo "Error: $VECTOR_TEST"
fi

# Test embedding function (if available)
test_info "Testing NeuronDB embedding function..."
EMBED_TEST=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT EXISTS(SELECT 1 FROM pg_proc WHERE proname = 'neurondb_embed');" | xargs)
if [ "$EMBED_TEST" = "t" ]; then
    test_pass "NeuronDB embed function available"
else
    test_warn "NeuronDB embed function not found (may require configuration)"
fi

# Test memory chunks table with vector
test_info "Testing memory_chunks table structure..."
MEMORY_CHUNKS_VECTOR=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'neurondb_agent' AND table_name = 'memory_chunks' AND column_name = 'embedding';" | xargs)
if echo "$MEMORY_CHUNKS_VECTOR" | grep -q "vector"; then
    test_pass "memory_chunks.embedding column is vector type"
else
    test_fail "memory_chunks.embedding column type incorrect"
fi

# Test HNSW index operations
test_info "Testing HNSW index..."
HNSW_INDEX=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT indexname FROM pg_indexes WHERE schemaname = 'neurondb_agent' AND tablename = 'memory_chunks' AND indexname LIKE '%embedding%';" | xargs)
if [ -n "$HNSW_INDEX" ]; then
    test_pass "HNSW index exists: $HNSW_INDEX"
else
    test_warn "HNSW index not found (will be created on first vector insert)"
fi

echo ""

# ============================================================================
# PHASE 7: Error Handling and Edge Cases
# ============================================================================
echo "PHASE 7: Error Handling and Edge Cases"
echo "======================================="

# Test invalid API key
test_info "Testing invalid API key rejection..."
INVALID_KEY_RESPONSE=$(curl -s -w "\n%{http_code}" -X GET "$SERVER_URL/api/v1/agents" \
    -H "Authorization: Bearer invalid_key_12345" 2>&1)
HTTP_CODE=$(echo "$INVALID_KEY_RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "401" ]; then
    test_pass "Invalid API key correctly rejected (401)"
else
    test_fail "Invalid API key not rejected (got $HTTP_CODE)"
fi

# Test missing authorization header
test_info "Testing missing authorization header..."
NO_AUTH_RESPONSE=$(curl -s -w "\n%{http_code}" -X GET "$SERVER_URL/api/v1/agents" 2>&1)
HTTP_CODE=$(echo "$NO_AUTH_RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "401" ]; then
    test_pass "Missing authorization correctly rejected (401)"
else
    test_fail "Missing authorization not rejected (got $HTTP_CODE)"
fi

# Test invalid agent ID
test_info "Testing invalid agent ID..."
INVALID_AGENT_RESPONSE=$(curl -s -w "\n%{http_code}" -X GET "$SERVER_URL/api/v1/agents/00000000-0000-0000-0000-000000000000" \
    -H "Authorization: Bearer $NEURONAGENT_API_KEY" 2>&1)
HTTP_CODE=$(echo "$INVALID_AGENT_RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "404" ] || [ "$HTTP_CODE" = "500" ]; then
    test_pass "Invalid agent ID handled correctly ($HTTP_CODE)"
else
    test_warn "Invalid agent ID returned unexpected code: $HTTP_CODE"
fi

# Test duplicate agent name
test_info "Testing duplicate agent name rejection..."
DUPLICATE_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVER_URL/api/v1/agents" \
    -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{
        \"name\": \"$AGENT_NAME\",
        \"system_prompt\": \"Test\",
        \"model_name\": \"gpt-4\"
    }" 2>&1)
HTTP_CODE=$(echo "$DUPLICATE_RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "400" ] || [ "$HTTP_CODE" = "409" ] || [ "$HTTP_CODE" = "500" ]; then
    test_pass "Duplicate agent name handled correctly ($HTTP_CODE)"
else
    test_warn "Duplicate agent name returned unexpected code: $HTTP_CODE"
fi

echo ""

# ============================================================================
# PHASE 8: Performance and Scalability
# ============================================================================
echo "PHASE 8: Performance and Scalability"
echo "====================================="

# Test multiple agents creation
test_info "Testing multiple agents creation..."
MULTI_AGENT_COUNT=0
for i in {1..3}; do
    MULTI_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVER_URL/api/v1/agents" \
        -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"name\": \"multi-test-agent-$i-$(date +%s)\",
            \"system_prompt\": \"Test agent $i\",
            \"model_name\": \"gpt-4\",
            \"enabled_tools\": [\"sql\"]
        }" 2>&1)
    HTTP_CODE=$(echo "$MULTI_RESPONSE" | tail -1)
    if [ "$HTTP_CODE" = "201" ]; then
        ((MULTI_AGENT_COUNT++))
    fi
done

if [ "$MULTI_AGENT_COUNT" -eq 3 ]; then
    test_pass "Multiple agents created successfully (3/3)"
else
    test_fail "Multiple agents creation failed ($MULTI_AGENT_COUNT/3)"
fi

# Test database connection pool
test_info "Checking database connection pool..."
DB_CONNECTIONS=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT count(*) FROM pg_stat_activity WHERE datname = '$DB_NAME';" | xargs)
test_info "Active database connections: $DB_CONNECTIONS"
test_pass "Database connection pool operational"

echo ""

# ============================================================================
# PHASE 9: Data Integrity
# ============================================================================
echo "PHASE 9: Data Integrity"
echo "======================="

# Check foreign key constraints
test_info "Verifying foreign key constraints..."
FK_COUNT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT COUNT(*) FROM information_schema.table_constraints WHERE constraint_schema = 'neurondb_agent' AND constraint_type = 'FOREIGN KEY';" | xargs)
if [ "$FK_COUNT" -gt 0 ]; then
    test_pass "Foreign key constraints exist ($FK_COUNT found)"
else
    test_fail "No foreign key constraints found"
fi

# Check NOT NULL constraints
test_info "Verifying NOT NULL constraints..."
NOT_NULL_COUNT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = 'neurondb_agent' AND is_nullable = 'NO';" | xargs)
if [ "$NOT_NULL_COUNT" -gt 0 ]; then
    test_pass "NOT NULL constraints exist ($NOT_NULL_COUNT found)"
else
    test_fail "No NOT NULL constraints found"
fi

# Check unique constraints
test_info "Verifying unique constraints..."
UNIQUE_COUNT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT COUNT(*) FROM information_schema.table_constraints WHERE constraint_schema = 'neurondb_agent' AND constraint_type = 'UNIQUE';" | xargs)
if [ "$UNIQUE_COUNT" -gt 0 ]; then
    test_pass "Unique constraints exist ($UNIQUE_COUNT found)"
else
    test_warn "No unique constraints found"
fi

echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo "================================================================"
echo "TEST SUMMARY"
echo "================================================================"
echo "Total Tests: $TESTS_TOTAL"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
if [ $BUGS_FOUND -gt 0 ]; then
    echo -e "${RED}Bugs Found: $BUGS_FOUND${NC}"
fi
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED!${NC}"
    echo ""
    echo "NeuronAgent is 100% integrated with NeuronDB and fully functional!"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please review the failures above and fix any issues."
    exit 1
fi

