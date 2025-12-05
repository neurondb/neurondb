#!/usr/bin/env python3
"""
Example 4: Streaming Responses

Demonstrates using WebSocket for streaming agent responses.
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
        name="streaming-example-agent",
        system_prompt="You are a helpful assistant. Provide detailed explanations.",
        model_name="gpt-4"
    )
    
    # Create conversation manager
    conversation = ConversationManager(client, agent_id=agent['id'])
    conversation.start()
    
    print("\n Streaming query: Explain how neural networks work")
    print("\n Streaming response:")
    print("-" * 60)
    
    full_response = []
    
    def on_chunk(chunk: str):
        """Handle each chunk of the response"""
        print(chunk, end='', flush=True)
        full_response.append(chunk)
    
    def on_complete(response: str):
        """Handle completion"""
        print("\n" + "-" * 60)
        print(f"✓ Stream complete. Total length: {len(response)} chars")
    
    # Stream the message
    conversation.stream(
        message="Explain how neural networks work in detail.",
        on_chunk=on_chunk,
        on_complete=on_complete
    )
    
    # Show history
    history = conversation.get_history()
    if history:
        print(f"\n Messages in history: {len(history)}")
    
    conversation.close()
    print("\n✓ Example completed!")

if __name__ == "__main__":
    main()

