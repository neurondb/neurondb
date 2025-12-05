#!/usr/bin/env python3
"""
Example 3: Conversation Management

Demonstrates using ConversationManager for managing conversations.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurondb_client import NeuronAgentClient, AgentManager, ConversationManager
from neurondb_client.utils.logging import setup_logging

setup_logging(level="INFO")

def main():
    client = NeuronAgentClient()
    agent_mgr = AgentManager(client)
    
    # Create agent
    print(" Creating agent...")
    agent = agent_mgr.create(
        name="conversation-example-agent",
        system_prompt="You are a helpful assistant. Keep responses concise.",
        model_name="gpt-4"
    )
    
    # Create conversation manager
    print("\n Starting conversation...")
    conversation = ConversationManager(
        client=client,
        agent_id=agent['id'],
        external_user_id="user-123",
        metadata={'source': 'example'}
    )
    
    conversation.start()
    print(f"✓ Conversation started: {conversation.session['id']}")
    
    # Send multiple messages
    messages = [
        "Hello! What's your name?",
        "What can you help me with?",
        "Can you explain what NeuronAgent is?"
    ]
    
    for i, msg in enumerate(messages, 1):
        print(f"\n{'='*60}")
        print(f"Message {i}: {msg}")
        print('='*60)
        
        response = conversation.send(msg)
        print(f"\nResponse: {response[:300]}...")
    
    # Show conversation history
    print("\n" + "=" * 60)
    print("Conversation History")
    print("=" * 60)
    history = conversation.get_history()
    for i, exchange in enumerate(history, 1):
        print(f"\nExchange {i}:")
        print(f"  User: {exchange['user']}")
        print(f"  Assistant: {exchange['assistant'][:150]}...")
        print(f"  Tokens: {exchange.get('tokens', 0)}")
    
    # Show total tokens
    total_tokens = conversation.get_total_tokens()
    print(f"\n Total tokens used: {total_tokens}")
    
    # Cleanup
    conversation.close()
    print("\n✓ Example completed!")

if __name__ == "__main__":
    main()

