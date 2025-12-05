#!/usr/bin/env python3
"""
Example 6: Advanced Agent Management

Demonstrates advanced agent operations:
- Finding agents
- Updating agents
- Listing agents
- Agent lifecycle management
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurondb_client import NeuronAgentClient, AgentManager
from neurondb_client.utils.logging import setup_logging

setup_logging(level="INFO")

def main():
    client = NeuronAgentClient()
    agent_mgr = AgentManager(client)
    
    # List all agents
    print("=" * 60)
    print("Listing All Agents")
    print("=" * 60)
    agents = agent_mgr.list()
    print(f"\nFound {len(agents)} agents:")
    for agent in agents:
        print(f"  - {agent['name']} ({agent['id']})")
        print(f"    Model: {agent['model_name']}")
        print(f"    Tools: {agent.get('enabled_tools', [])}")
    
    # Find agent by name
    print("\n" + "=" * 60)
    print("Finding Agent by Name")
    print("=" * 60)
    agent = agent_mgr.find_by_name("basic-example-agent")
    if agent:
        print(f"\n✓ Found agent: {agent['id']}")
        print(f"   Description: {agent.get('description', 'N/A')}")
    else:
        print("\n✗ Agent not found")
    
    # Create a new agent
    print("\n" + "=" * 60)
    print("Creating New Agent")
    print("=" * 60)
    new_agent = agent_mgr.create(
        name="advanced-example-agent",
        system_prompt="You are an advanced assistant with multiple capabilities.",
        model_name="gpt-4",
        description="Advanced example agent",
        enabled_tools=['sql', 'http', 'code'],
        config={
            'temperature': 0.7,
            'max_tokens': 2500,
            'top_p': 0.95
        }
    )
    print(f"✓ Created: {new_agent['id']}")
    
    # Update the agent
    print("\n" + "=" * 60)
    print("Updating Agent")
    print("=" * 60)
    updated = agent_mgr.update(
        agent_id=new_agent['id'],
        description="Updated advanced example agent",
        config={
            'temperature': 0.8,
            'max_tokens': 3000
        }
    )
    print(f"✓ Updated: {updated['id']}")
    print(f"   New description: {updated.get('description')}")
    print(f"   New temperature: {updated['config'].get('temperature')}")
    
    # Get agent details
    print("\n" + "=" * 60)
    print("Getting Agent Details")
    print("=" * 60)
    details = agent_mgr.get(new_agent['id'])
    print(f"\nAgent Details:")
    print(f"  ID: {details['id']}")
    print(f"  Name: {details['name']}")
    print(f"  Model: {details['model_name']}")
    print(f"  Tools: {details.get('enabled_tools', [])}")
    print(f"  Config: {details.get('config', {})}")
    print(f"  Created: {details.get('created_at', 'N/A')}")
    
    # Cleanup: Delete the agent (optional)
    print("\n" + "=" * 60)
    print("Cleanup (Optional)")
    print("=" * 60)
    response = input(f"\nDelete agent '{new_agent['name']}'? (y/n): ")
    if response.lower() == 'y':
        agent_mgr.delete(new_agent['id'])
        print(f"✓ Agent deleted")
    else:
        print("Agent kept")
    
    print("\n✓ Example completed!")

if __name__ == "__main__":
    main()

