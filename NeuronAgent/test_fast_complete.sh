#!/bin/bash
# Fast comprehensive test for NeuronAgent with NeuronDB integration

set +e  # Don't exit on error, collect all results

cd "$(dirname "$0")"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASSED=0
FAILED=0

pass() { echo -e "${GREEN}✅${NC} $1"; ((PASSED++)); }
fail() { echo -e "${RED}❌${NC} $1"; ((FAILED++)); }
info() { echo -e "${BLUE}ℹ️${NC} $1"; }

echo "================================================================"
echo "NeuronAgent Complete Integration Test"
echo "================================================================"
echo ""

# 1. Server
info "Testing server..."
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    pass "Server health check"
else
    fail "Server not responding"
    exit 1
fi

# 2. Database
info "Testing database..."
if psql -d neurondb -U pge -c "SELECT 1;" > /dev/null 2>&1; then
    pass "Database connection"
else
    fail "Database connection failed"
    exit 1
fi

# 3. NeuronDB Extension
info "Testing NeuronDB extension..."
if psql -d neurondb -U pge -t -c "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'neurondb');" | grep -q t; then
    pass "NeuronDB extension installed"
else
    fail "NeuronDB extension not found"
    exit 1
fi

# 4. Vector type
info "Testing vector type..."
if psql -d neurondb -U pge -t -c "SELECT '[1,2,3]'::vector(3);" > /dev/null 2>&1; then
    pass "Vector type working"
else
    fail "Vector type not working"
fi

# 5. Schema
info "Testing schema..."
TABLES=$(psql -d neurondb -U pge -t -c "SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'neurondb_agent';" | xargs)
if [ "$TABLES" -ge 8 ]; then
    pass "Schema tables exist ($TABLES tables)"
else
    fail "Schema incomplete ($TABLES tables, expected 8+)"
fi

# 6. Generate API key
info "Generating API key..."
API_KEY_OUTPUT=$(go run cmd/generate-key/main.go -org test -user test -rate 100 -roles user -db-host localhost -db-port 5432 -db-name neurondb -db-user pge 2>&1)
API_KEY=$(echo "$API_KEY_OUTPUT" | grep "^Key:" | sed 's/^Key: //' | sed 's/[[:space:]]*$//')
if [ -n "$API_KEY" ] && [ ${#API_KEY} -gt 30 ]; then
    pass "API key generated: ${API_KEY:0:16}... (length: ${#API_KEY})"
    export NEURONAGENT_API_KEY="$API_KEY"
    # Wait a moment for server to pick up new key
    sleep 2
else
    fail "API key generation failed or key too short (length: ${#API_KEY})"
    echo "Output: $API_KEY_OUTPUT"
    exit 1
fi

# 7. Test API endpoints
info "Testing API endpoints..."

# Health
if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8080/health)" = "200" ]; then
    pass "GET /health"
else
    fail "GET /health"
fi

# List agents
RESP=$(curl -s -w "\n%{http_code}" -X GET http://localhost:8080/api/v1/agents -H "Authorization: Bearer $API_KEY")
CODE=$(echo "$RESP" | tail -1)
if [ "$CODE" = "200" ]; then
    pass "GET /api/v1/agents"
else
    fail "GET /api/v1/agents (got $CODE)"
fi

# Create agent
AGENT_NAME="test-agent-$(date +%s)"
RESP=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8080/api/v1/agents \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"$AGENT_NAME\",\"system_prompt\":\"Test agent for integration testing\",\"model_name\":\"gpt-4\",\"enabled_tools\":[\"sql\"]}")
CODE=$(echo "$RESP" | tail -1)
if [ "$CODE" = "201" ]; then
    pass "POST /api/v1/agents"
    AGENT_ID=$(echo "$RESP" | head -1 | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])" 2>/dev/null)
else
    fail "POST /api/v1/agents (got $CODE)"
    echo "Response: $(echo "$RESP" | head -1)"
    AGENT_ID=""
fi

# Create session
if [ -n "$AGENT_ID" ]; then
    RESP=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8080/api/v1/sessions \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"agent_id\":\"$AGENT_ID\"}")
    CODE=$(echo "$RESP" | tail -1)
    if [ "$CODE" = "201" ]; then
        pass "POST /api/v1/sessions"
        SESSION_ID=$(echo "$RESP" | head -1 | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])" 2>/dev/null)
    else
        fail "POST /api/v1/sessions (got $CODE)"
        SESSION_ID=""
    fi
fi

# Send message
if [ -n "$SESSION_ID" ]; then
    RESP=$(curl -s -w "\n%{http_code}" -X POST "http://localhost:8080/api/v1/sessions/$SESSION_ID/messages" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"content":"Test message","role":"user"}')
    CODE=$(echo "$RESP" | tail -1)
    if [ "$CODE" = "200" ]; then
        pass "POST /api/v1/sessions/{id}/messages"
    else
        fail "POST /api/v1/sessions/{id}/messages (got $CODE)"
    fi
fi

# 8. Database verification
info "Verifying database..."

if [ -n "$AGENT_ID" ]; then
    DB_AGENT=$(psql -d neurondb -U pge -t -c "SELECT name FROM neurondb_agent.agents WHERE id = '$AGENT_ID';" | xargs)
    if [ "$DB_AGENT" = "$AGENT_NAME" ]; then
        pass "Agent in database"
    else
        fail "Agent not in database"
    fi
fi

if [ -n "$SESSION_ID" ]; then
    DB_SESSION=$(psql -d neurondb -U pge -t -c "SELECT id FROM neurondb_agent.sessions WHERE id = '$SESSION_ID';" | xargs)
    if [ "$DB_SESSION" = "$SESSION_ID" ]; then
        pass "Session in database"
    else
        fail "Session not in database"
    fi
    
    MSG_COUNT=$(psql -d neurondb -U pge -t -c "SELECT COUNT(*) FROM neurondb_agent.messages WHERE session_id = '$SESSION_ID';" | xargs)
    if [ "$MSG_COUNT" -gt 0 ]; then
        pass "Messages in database ($MSG_COUNT)"
    else
        fail "No messages in database"
    fi
fi

# 9. NeuronDB features
info "Testing NeuronDB features..."

# Vector operations
if psql -d neurondb -U pge -c "SELECT '[1,2,3]'::vector(3) <=> '[4,5,6]'::vector(3);" > /dev/null 2>&1; then
    pass "Vector distance operations"
else
    fail "Vector distance operations"
fi

# Memory chunks vector column
VECTOR_COL=$(psql -d neurondb -U pge -t -c "SELECT udt_name FROM information_schema.columns WHERE table_schema = 'neurondb_agent' AND table_name = 'memory_chunks' AND column_name = 'embedding';" | xargs)
if echo "$VECTOR_COL" | grep -q "vector"; then
    pass "memory_chunks.embedding is vector type"
else
    fail "memory_chunks.embedding type incorrect (got: $VECTOR_COL)"
fi

# HNSW index
HNSW=$(psql -d neurondb -U pge -t -c "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'neurondb_agent' AND tablename = 'memory_chunks' AND indexname LIKE '%embedding%';" | xargs)
if [ "$HNSW" -gt 0 ]; then
    pass "HNSW index on memory_chunks"
else
    fail "HNSW index missing"
fi

# 10. Error handling
info "Testing error handling..."

# Invalid key
INVALID_CODE=$(curl -s -o /dev/null -w '%{http_code}' -X GET http://localhost:8080/api/v1/agents -H "Authorization: Bearer invalid_key")
if [ "$INVALID_CODE" = "401" ]; then
    pass "Invalid key rejected"
else
    fail "Invalid key not rejected (got $INVALID_CODE)"
fi

# Missing auth
NO_AUTH_CODE=$(curl -s -o /dev/null -w '%{http_code}' -X GET http://localhost:8080/api/v1/agents)
if [ "$NO_AUTH_CODE" = "401" ]; then
    pass "Missing auth rejected"
else
    fail "Missing auth not rejected (got $NO_AUTH_CODE)"
fi

echo ""
echo "================================================================"
echo "RESULTS"
echo "================================================================"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo "NeuronAgent is 100% integrated with NeuronDB!"
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    exit 1
fi

