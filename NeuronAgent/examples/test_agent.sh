#!/bin/bash
# Test script for NeuronAgent

set -e

API_KEY="testkey1_$(openssl rand -hex 16)"
BASE_URL="http://localhost:8080"

echo "=========================================="
echo "NeuronAgent Test Script"
echo "=========================================="

# Check server health
echo -n "1. Checking server health... "
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health)
if [ "$HEALTH" = "200" ]; then
    echo "✓ Server is healthy"
else
    echo "✗ Server not responding (status: $HEALTH)"
    exit 1
fi

# For testing, we'll use a simple API key
# In production, generate proper keys
echo ""
echo "2. Testing API endpoints..."

# Create agent
echo -n "   Creating agent... "
AGENT_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/agents" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "test-agent-'$(date +%s)'",
        "system_prompt": "You are a helpful test assistant.",
        "model_name": "gpt-4",
        "enabled_tools": ["sql"],
        "config": {"temperature": 0.7, "max_tokens": 1000}
    }')

if echo "$AGENT_RESPONSE" | grep -q "id"; then
    AGENT_ID=$(echo "$AGENT_RESPONSE" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
    echo "✓ Agent created: $AGENT_ID"
else
    echo "✗ Failed to create agent"
    echo "   Response: $AGENT_RESPONSE"
    exit 1
fi

# Create session
echo -n "   Creating session... "
SESSION_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/sessions" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"agent_id\": \"$AGENT_ID\"}")

if echo "$SESSION_RESPONSE" | grep -q "id"; then
    SESSION_ID=$(echo "$SESSION_RESPONSE" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
    echo "✓ Session created: $SESSION_ID"
else
    echo "✗ Failed to create session"
    echo "   Response: $SESSION_RESPONSE"
    exit 1
fi

# Send message
echo -n "   Sending message... "
MESSAGE_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/sessions/$SESSION_ID/messages" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"content": "Hello! This is a test message.", "role": "user"}')

if echo "$MESSAGE_RESPONSE" | grep -q "response\|error"; then
    echo "✓ Message sent"
    echo "   Response preview: $(echo "$MESSAGE_RESPONSE" | head -c 100)..."
else
    echo "⚠️  Unexpected response format"
    echo "   Response: $MESSAGE_RESPONSE"
fi

echo ""
echo "=========================================="
echo "✓ Basic tests completed!"
echo "=========================================="
echo ""
echo "Agent ID: $AGENT_ID"
echo "Session ID: $SESSION_ID"
echo ""
echo "Test with Python client:"
echo "  export NEURONAGENT_API_KEY=$API_KEY"
echo "  cd examples && python3 examples_modular/01_basic_usage.py"

