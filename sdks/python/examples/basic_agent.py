#!/usr/bin/env python3
"""
Basic NeuronAgent usage example

This example demonstrates:
1. Creating an agent
2. Creating a session
3. Sending messages
4. Retrieving conversation history
"""

from neuronagent import NeuronAgentClient

def main():
    # Initialize client
    client = NeuronAgentClient(
        base_url="http://localhost:8080",
        api_key="your-api-key-here"
    )

    # Step 1: Create an agent
    print("Creating agent...")
    agent = client.agents.create_agent(
        name="example-agent",
        description="A simple example agent",
        system_prompt="You are a helpful assistant that answers questions clearly and concisely.",
        model_name="gpt-4",
        enabled_tools=["sql", "http"],
        config={
            "temperature": 0.7,
            "max_tokens": 1000
        }
    )
    print(f"Created agent: {agent.id}")

    # Step 2: Create a session
    print("\nCreating session...")
    session = client.sessions.create_session(
        agent_id=agent.id,
        metadata={"user_id": "example-user"}
    )
    print(f"Created session: {session.id}")

    # Step 3: Send messages
    print("\nSending messages...")
    
    messages = [
        "Hello! Can you help me understand vector databases?",
        "What are the key advantages of using NeuronDB?",
        "How does HNSW indexing work?"
    ]

    for message in messages:
        print(f"\nUser: {message}")
        response = client.sessions.send_message(
            session_id=session.id,
            content=message
        )
        print(f"Agent: {response.content}")

    # Step 4: Retrieve conversation history
    print("\n\nRetrieving conversation history...")
    history = client.sessions.get_messages(session_id=session.id)
    
    print(f"\nTotal messages: {len(history)}")
    for msg in history:
        print(f"\n[{msg.role}]: {msg.content[:100]}...")

    print("\nExample complete!")

if __name__ == "__main__":
    main()







