#!/usr/bin/env python3
"""Quick test script for NeuronAgent"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from neurondb_client import NeuronAgentClient, AgentManager, SessionManager

# Use test API key (in production, generate proper keys)
API_KEY = "testkey1_dummy"  # This won't work for auth, but we can test connection

print("=" * 60)
print("NeuronAgent Quick Test")
print("=" * 60)

# Initialize client
print("\n1. Initializing client...")
try:
    client = NeuronAgentClient(api_key=API_KEY)
    print("   ✅ Client created")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Health check
print("\n2. Checking server health...")
if client.health_check():
    print("   ✅ Server is healthy")
else:
    print("   ❌ Server is not responding")
    print("   Make sure NeuronAgent is running: go run cmd/agent-server/main.go")
    sys.exit(1)

# Test API (will fail auth, but tests connection)
print("\n3. Testing API connection...")
try:
    response = client.get('/api/v1/agents')
    print("   ✅ API is accessible")
except Exception as e:
    if "401" in str(e) or "Authentication" in str(e):
        print("   ⚠️  API accessible but authentication required (expected)")
        print("   Generate API key: ./scripts/generate_api_keys.sh")
    else:
        print(f"   ❌ API error: {e}")

# Show metrics
print("\n4. Client metrics:")
metrics = client.get_metrics()
print(f"   Requests: {metrics['requests']}")
print(f"   Errors: {metrics['errors']}")

print("\n" + "=" * 60)
print("✅ Basic connectivity test completed!")
print("=" * 60)
print("\nNext steps:")
print("1. Generate API key: cd NeuronAgent && ./scripts/generate_api_keys.sh")
print("2. Set environment: export NEURONAGENT_API_KEY=your_key")
print("3. Run examples: python3 examples_modular/01_basic_usage.py")

