#!/usr/bin/env python3
"""
NeuronAgent Module: Basic Agent
================================
Learn how to create and use agents with NeuronAgent.

Run: python 01_basic_agent.py

Requirements:
    - NeuronAgent server running
    - API key configured
"""

import requests
import os
import json

# NeuronAgent configuration
AGENT_URL = os.getenv('AGENT_URL', 'http://localhost:8080')
API_KEY = os.getenv('NEURONAGENT_API_KEY', 'your-api-key-here')

print("=" * 60)
print("NeuronAgent Module: Basic Agent")
print("=" * 60)

print("\nNeuronAgent provides:")
print("  - AI agent runtime")
print("  - Tool orchestration")
print("  - Session management")
print("  - Memory persistence in PostgreSQL")

print("\n1. Creating an agent...")
print("  POST /api/v1/agents")
print("  {")
print('    "name": "my_agent",')
print('    "model": "gpt-4",')
print('    "tools": ["sql", "http"]')
print("  }")

print("\n2. Sending a message to agent...")
print("  POST /api/v1/agents/{agent_id}/messages")
print("  {")
print('    "message": "What is the weather in San Francisco?"')
print("  }")

print("\n3. For complete examples, see:")
print("  - NeuronAgent/examples/")
print("  - NeuronAgent/README.md")
print("  - examples/agent-tools/")

print("\nNote: This example shows the API structure.")
print("For working examples, ensure NeuronAgent server is running:")
print("  cd NeuronAgent")
print("  ./start_server.sh")

print("\n✓ Example complete!")








