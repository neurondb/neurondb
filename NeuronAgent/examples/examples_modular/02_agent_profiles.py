#!/usr/bin/env python3
"""
Example 2: Using Agent Profiles

Demonstrates creating agents from predefined profiles.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurondb_client import NeuronAgentClient, AgentManager
from neurondb_client.agents import AgentProfile, get_default_profile, load_profiles_from_file
from neurondb_client.utils.logging import setup_logging

setup_logging(level="INFO")

def main():
    client = NeuronAgentClient()
    agent_mgr = AgentManager(client)
    
    # Method 1: Use default profiles
    print("=" * 60)
    print("Method 1: Using Default Profiles")
    print("=" * 60)
    
    profile = get_default_profile('research_assistant')
    if profile:
        print(f"\n📋 Profile: {profile.name}")
        print(f"   Description: {profile.description}")
        print(f"   Tools: {profile.enabled_tools}")
        
        agent = agent_mgr.create_from_profile(profile)
        print(f"✅ Agent created: {agent['id']}")
    
    # Method 2: Load from file
    print("\n" + "=" * 60)
    print("Method 2: Loading from Configuration File")
    print("=" * 60)
    
    config_file = os.path.join(os.path.dirname(__file__), '..', 'agent_configs.json')
    if os.path.exists(config_file):
        profiles = load_profiles_from_file(config_file)
        print(f"\n📚 Loaded {len(profiles)} profiles")
        
        for name, profile in profiles.items():
            print(f"\n  - {name}: {profile.description}")
        
        # Create agent from loaded profile
        if 'data_analyst' in profiles:
            profile = profiles['data_analyst']
            agent = agent_mgr.create_from_profile(profile)
            print(f"\n✅ Created agent from profile: {agent['name']}")
    
    # Method 3: Create custom profile
    print("\n" + "=" * 60)
    print("Method 3: Custom Profile")
    print("=" * 60)
    
    custom_profile = AgentProfile(
        name="custom-assistant",
        description="Custom assistant for specific tasks",
        system_prompt="You are a specialized assistant for customer support.",
        model_name="gpt-4",
        enabled_tools=['sql', 'http'],
        config={
            'temperature': 0.6,
            'max_tokens': 1500
        }
    )
    
    agent = agent_mgr.create_from_profile(custom_profile)
    print(f"✅ Custom agent created: {agent['id']}")
    
    print("\n✅ Example completed!")

if __name__ == "__main__":
    main()

