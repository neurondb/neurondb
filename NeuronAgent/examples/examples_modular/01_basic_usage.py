#!/usr/bin/env python3
"""
Example 1: Basic Usage with Modular Client

Demonstrates the simplest way to use the modular client library.
"""

import sys
import os

# Add parent directory to path to import client library
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurondb_client import NeuronAgentClient, AgentManager, SessionManager
from neurondb_client.utils.logging import setup_logging

# Setup logging
setup_logging(level="INFO")

def main():
    # Initialize client
    client = NeuronAgentClient()
    
    # Check server health
    if not client.health_check():
        print("❌ Server is not healthy")
        return
    
    print("✅ Server is healthy")
    
    # Create agent manager
    agent_mgr = AgentManager(client)
    
    # Create an agent
    print("\n📝 Creating agent...")
    agent = agent_mgr.create(
        name="basic-example-agent",
        system_prompt="You are a helpful assistant.",
        model_name="gpt-4",
        enabled_tools=['sql', 'http']
    )
    print(f"✅ Agent created: {agent['id']}")
    
    # Create session manager
    session_mgr = SessionManager(client)
    
    # Create a session
    print("\n💬 Creating session...")
    session = session_mgr.create(agent_id=agent['id'])
    print(f"✅ Session created: {session['id']}")
    
    # Send a message
    print("\n💭 Sending message...")
    response = session_mgr.send_message(
        session_id=session['id'],
        content="Hello! Can you introduce yourself?"
    )
    print(f"🤖 Response: {response['response'][:200]}...")
    print(f"   Tokens used: {response.get('tokens_used', 0)}")
    
    # Show metrics
    metrics = client.get_metrics()
    print(f"\n📊 Metrics:")
    print(f"   Total requests: {metrics['requests']}")
    print(f"   Errors: {metrics['errors']}")
    
    print("\n✅ Example completed!")

if __name__ == "__main__":
    main()

