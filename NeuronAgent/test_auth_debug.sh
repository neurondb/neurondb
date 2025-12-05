#!/bin/bash
# Debug authentication issue

cd "$(dirname "$0")"

echo "=== Authentication Debug Test ==="
echo ""

# Generate a fresh key
echo "1. Generating API key..."
API_KEY_OUTPUT=$(go run cmd/generate-key/main.go -org debug-auth -user debug-auth -rate 100 -roles user -db-host localhost -db-port 5432 -db-name neurondb -db-user pge 2>&1)
API_KEY=$(echo "$API_KEY_OUTPUT" | grep "^Key:" | sed 's/^Key: //' | sed 's/[[:space:]]*$//')
PREFIX=$(echo "$API_KEY" | cut -c1-8)

echo "   Full Key: $API_KEY"
echo "   Key Length: ${#API_KEY}"
echo "   Prefix: $PREFIX"
echo ""

# Verify in database
echo "2. Verifying in database..."
DB_KEY=$(psql -d neurondb -t -c "SELECT key_prefix FROM neurondb_agent.api_keys WHERE key_prefix = '$PREFIX';" | xargs)
if [ "$DB_KEY" = "$PREFIX" ]; then
    echo "   ✓ Key found in database"
else
    echo "   ✗ Key NOT found in database"
    exit 1
fi

# Test key verification
echo ""
echo "3. Testing key verification with Python..."
python3 << EOF
import bcrypt
import subprocess

key = "$API_KEY"
prefix = "$PREFIX"

# Get hash
result = subprocess.run(['psql', '-d', 'neurondb', '-U', 'pge', '-t', '-c', 
    f"SELECT key_hash FROM neurondb_agent.api_keys WHERE key_prefix = '{prefix}';"],
    capture_output=True, text=True)
hash_from_db = result.stdout.strip()

try:
    verified = bcrypt.checkpw(key.encode('utf-8'), hash_from_db.encode('utf-8'))
    if verified:
        print(f"   ✓ Key verification: PASSED")
    else:
        print(f"   ✗ Key verification: FAILED")
        print(f"   Key: {key}")
        print(f"   Hash: {hash_from_db[:50]}...")
except Exception as e:
    print(f"   ✗ Verification error: {e}")
EOF

# Wait for server
echo ""
echo "4. Waiting for server to be ready..."
sleep 5

# Test API
echo ""
echo "5. Testing API with key..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X GET "http://localhost:8080/api/v1/agents" -H "Authorization: Bearer $API_KEY")
echo "   HTTP Code: $HTTP_CODE"

if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✓ Authentication working!"
elif [ "$HTTP_CODE" = "401" ]; then
    echo "   ✗ Authentication failed (401)"
    echo ""
    echo "   Debugging steps:"
    echo "   1. Check server logs: tail -f /tmp/neurondb-agent-debug.log"
    echo "   2. Verify key prefix matches: psql -d neurondb -c \"SELECT key_prefix FROM neurondb_agent.api_keys WHERE key_prefix = '$PREFIX';\""
    echo "   3. Restart server after generating key"
else
    echo "   ⚠️  Unexpected code: $HTTP_CODE"
fi

echo ""
echo "=== Debug Complete ==="

