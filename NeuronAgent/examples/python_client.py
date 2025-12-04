#!/usr/bin/env python3
"""
NeuronAgent Python Client Example

A comprehensive example demonstrating how to use NeuronAgent to create
and interact with AI agents. This example shows:

1. Creating an agent with custom configuration
2. Creating a session for conversation
3. Sending messages and receiving responses
4. Using WebSocket for streaming responses
5. Working with tools (SQL, HTTP, etc.)
6. Error handling and best practices
"""

import json
import os
import sys
import time
import uuid
from typing import Dict, List, Optional
import requests
import websocket
import threading


class NeuronAgentClient:
    """Client for interacting with NeuronAgent API"""
    
    def __init__(self, base_url: str = "http://localhost:8080", api_key: str = None):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the NeuronAgent server
            api_key: API key for authentication (or set NEURONAGENT_API_KEY env var)
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.getenv('NEURONAGENT_API_KEY')
        if not self.api_key:
            raise ValueError("API key required. Set NEURONAGENT_API_KEY env var or pass api_key parameter")
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def health_check(self) -> bool:
        """Check if the server is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def create_agent(
        self,
        name: str,
        system_prompt: str,
        model_name: str = "gpt-4",
        description: Optional[str] = None,
        enabled_tools: List[str] = None,
        config: Optional[Dict] = None,
        memory_table: Optional[str] = None
    ) -> Dict:
        """
        Create a new agent
        
        Args:
            name: Unique name for the agent
            system_prompt: System prompt defining agent behavior
            model_name: LLM model to use (default: gpt-4)
            description: Optional description
            enabled_tools: List of enabled tools (e.g., ['sql', 'http'])
            config: Optional model configuration (temperature, max_tokens, etc.)
            memory_table: Optional memory table name for long-term memory
        
        Returns:
            Created agent object
        """
        if enabled_tools is None:
            enabled_tools = ['sql', 'http']
        
        if config is None:
            config = {
                'temperature': 0.7,
                'max_tokens': 2000,
                'top_p': 0.9
            }
        
        payload = {
            'name': name,
            'system_prompt': system_prompt,
            'model_name': model_name,
            'enabled_tools': enabled_tools,
            'config': config
        }
        
        if description:
            payload['description'] = description
        if memory_table:
            payload['memory_table'] = memory_table
        
        response = requests.post(
            f"{self.base_url}/api/v1/agents",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_agent(self, agent_id: str) -> Dict:
        """Get agent by ID"""
        response = requests.get(
            f"{self.base_url}/api/v1/agents/{agent_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def list_agents(self) -> List[Dict]:
        """List all agents"""
        response = requests.get(
            f"{self.base_url}/api/v1/agents",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def create_session(
        self,
        agent_id: str,
        external_user_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Create a new session for conversation
        
        Args:
            agent_id: ID of the agent to use
            external_user_id: Optional external user identifier
            metadata: Optional metadata dictionary
        
        Returns:
            Created session object
        """
        payload = {
            'agent_id': agent_id
        }
        
        if external_user_id:
            payload['external_user_id'] = external_user_id
        if metadata:
            payload['metadata'] = metadata
        
        response = requests.post(
            f"{self.base_url}/api/v1/sessions",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def send_message(
        self,
        session_id: str,
        content: str,
        stream: bool = False,
        role: str = "user",
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Send a message to the agent
        
        Args:
            session_id: Session ID
            content: Message content
            stream: Whether to stream the response
            role: Message role (default: "user")
            metadata: Optional metadata
        
        Returns:
            Response from the agent
        """
        payload = {
            'content': content,
            'role': role,
            'stream': stream
        }
        
        if metadata:
            payload['metadata'] = metadata
        
        response = requests.post(
            f"{self.base_url}/api/v1/sessions/{session_id}/messages",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_messages(self, session_id: str, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get messages from a session"""
        params = {'limit': limit, 'offset': offset}
        response = requests.get(
            f"{self.base_url}/api/v1/sessions/{session_id}/messages",
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def stream_message(self, session_id: str, content: str, on_message=None):
        """
        Stream a message using WebSocket
        
        Args:
            session_id: Session ID
            content: Message content
            on_message: Callback function(message_dict) for each message chunk
        """
        ws_url = f"ws://{self.base_url.replace('http://', '').replace('https://', '')}/ws?session_id={session_id}"
        
        def on_open(ws):
            # Send the message
            ws.send(json.dumps({'content': content}))
        
        def on_message_ws(ws, message):
            data = json.loads(message)
            if on_message:
                on_message(data)
            if data.get('complete'):
                ws.close()
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message_ws,
            on_error=on_error,
            header=[f"Authorization: Bearer {self.api_key}"]
        )
        ws.run_forever()


def example_basic_agent():
    """Example: Create and use a basic agent"""
    print("\n" + "="*60)
    print("Example 1: Basic Agent")
    print("="*60)
    
    client = NeuronAgentClient()
    
    # Check server health
    if not client.health_check():
        print("❌ Server is not healthy. Make sure NeuronAgent is running.")
        return
    
    print("✅ Server is healthy")
    
    # Create an agent
    print("\n📝 Creating agent...")
    agent = client.create_agent(
        name="example-assistant",
        system_prompt="You are a helpful, knowledgeable assistant. Answer questions clearly and concisely.",
        model_name="gpt-4",
        description="Example assistant agent",
        enabled_tools=['sql', 'http'],
        config={
            'temperature': 0.7,
            'max_tokens': 1500
        }
    )
    print(f"✅ Agent created: {agent['id']}")
    print(f"   Name: {agent['name']}")
    print(f"   Model: {agent['model_name']}")
    
    # Create a session
    print("\n💬 Creating session...")
    session = client.create_session(
        agent_id=agent['id'],
        external_user_id="example-user-123",
        metadata={'source': 'example'}
    )
    print(f"✅ Session created: {session['id']}")
    
    # Send messages
    messages = [
        "Hello! Can you introduce yourself?",
        "What can you help me with?",
        "Can you explain what NeuronAgent is?"
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"\n💭 Sending message {i}: {message[:50]}...")
        response = client.send_message(
            session_id=session['id'],
            content=message
        )
        print(f"🤖 Agent response: {response['response'][:200]}...")
        print(f"   Tokens used: {response.get('tokens_used', 'N/A')}")
        time.sleep(1)  # Small delay between messages
    
    # Get conversation history
    print("\n📜 Retrieving conversation history...")
    history = client.get_messages(session['id'], limit=10)
    print(f"✅ Retrieved {len(history)} messages")
    for msg in history[-3:]:  # Show last 3
        print(f"   [{msg['role']}]: {msg['content'][:80]}...")


def example_research_agent():
    """Example: Create a research agent with HTTP tools"""
    print("\n" + "="*60)
    print("Example 2: Research Agent with HTTP Tools")
    print("="*60)
    
    client = NeuronAgentClient()
    
    # Create a research-focused agent
    print("\n📝 Creating research agent...")
    agent = client.create_agent(
        name="research-assistant",
        system_prompt="""You are a research assistant. When users ask questions, you can:
1. Search the web using HTTP tools to find current information
2. Query databases using SQL tools to find stored data
3. Synthesize information from multiple sources
4. Provide citations and sources when possible

Always be thorough and verify information from multiple sources when possible.""",
        model_name="gpt-4",
        description="Research assistant with web search capabilities",
        enabled_tools=['http', 'sql'],
        config={
            'temperature': 0.5,
            'max_tokens': 2500,
            'top_p': 0.95
        }
    )
    print(f"✅ Research agent created: {agent['id']}")
    
    # Create session
    session = client.create_session(agent_id=agent['id'])
    print(f"✅ Session created: {session['id']}")
    
    # Example research query
    query = "What are the latest developments in AI agent frameworks?"
    print(f"\n🔍 Research query: {query}")
    print("   (Note: This will use HTTP tools if configured)")
    
    response = client.send_message(
        session_id=session['id'],
        content=query
    )
    print(f"\n📊 Research response:")
    print(f"   {response['response'][:300]}...")
    if response.get('tool_calls'):
        print(f"\n🔧 Tools used: {len(response['tool_calls'])}")
        for tool_call in response['tool_calls']:
            print(f"   - {tool_call.get('name', 'unknown')}")


def example_data_analyst_agent():
    """Example: Create a data analyst agent with SQL tools"""
    print("\n" + "="*60)
    print("Example 3: Data Analyst Agent with SQL Tools")
    print("="*60)
    
    client = NeuronAgentClient()
    
    # Create a data analyst agent
    print("\n📝 Creating data analyst agent...")
    agent = client.create_agent(
        name="data-analyst",
        system_prompt="""You are a data analyst. You help users:
1. Write and execute SQL queries
2. Analyze data and provide insights
3. Create reports and visualizations
4. Explain data patterns and trends

Always validate SQL queries before execution and explain your findings clearly.""",
        model_name="gpt-4",
        description="Data analyst with SQL query capabilities",
        enabled_tools=['sql'],
        config={
            'temperature': 0.2,  # Lower temperature for more precise SQL
            'max_tokens': 2000
        }
    )
    print(f"✅ Data analyst agent created: {agent['id']}")
    
    # Create session
    session = client.create_session(agent_id=agent['id'])
    print(f"✅ Session created: {session['id']}")
    
    # Example data query
    query = "Can you help me write a SQL query to find the top 10 customers by revenue?"
    print(f"\n💾 Data query: {query}")
    
    response = client.send_message(
        session_id=session['id'],
        content=query
    )
    print(f"\n📈 Analyst response:")
    print(f"   {response['response'][:400]}...")


def example_streaming():
    """Example: Use WebSocket for streaming responses"""
    print("\n" + "="*60)
    print("Example 4: Streaming Responses via WebSocket")
    print("="*60)
    
    client = NeuronAgentClient()
    
    # Get or create an agent
    agents = client.list_agents()
    if not agents:
        print("Creating agent for streaming example...")
        agent = client.create_agent(
            name="streaming-agent",
            system_prompt="You are a helpful assistant. Provide detailed, thoughtful responses.",
            enabled_tools=[]
        )
    else:
        agent = agents[0]
    
    # Create session
    session = client.create_session(agent_id=agent['id'])
    print(f"✅ Session created: {session['id']}")
    
    # Stream a message
    query = "Explain how neural networks work in detail."
    print(f"\n💭 Streaming query: {query}")
    print("\n📡 Streaming response:")
    print("-" * 60)
    
    full_response = []
    
    def on_message_chunk(data):
        if data.get('type') == 'response':
            content = data.get('content', '')
            if content:
                print(content, end='', flush=True)
                full_response.append(content)
        if data.get('complete'):
            print("\n" + "-" * 60)
            print(f"✅ Stream complete. Total length: {len(''.join(full_response))} chars")
    
    client.stream_message(
        session_id=session['id'],
        content=query,
        on_message=on_message_chunk
    )


def example_error_handling():
    """Example: Proper error handling"""
    print("\n" + "="*60)
    print("Example 5: Error Handling")
    print("="*60)
    
    try:
        # Try with invalid API key
        client = NeuronAgentClient(api_key="invalid-key")
        client.list_agents()
    except requests.exceptions.HTTPError as e:
        print(f"✅ Caught expected authentication error: {e.response.status_code}")
    
    try:
        # Try with valid client but invalid session
        client = NeuronAgentClient()
        client.send_message(
            session_id=str(uuid.uuid4()),  # Non-existent session
            content="Hello"
        )
    except requests.exceptions.HTTPError as e:
        print(f"✅ Caught expected session error: {e.response.status_code}")
    
    print("\n✅ Error handling examples completed")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("NeuronAgent Python Client Examples")
    print("="*60)
    print("\nMake sure NeuronAgent server is running on http://localhost:8080")
    print("Set NEURONAGENT_API_KEY environment variable with your API key")
    print("\n" + "="*60)
    
    # Check if API key is set
    if not os.getenv('NEURONAGENT_API_KEY'):
        print("\n❌ Error: NEURONAGENT_API_KEY environment variable not set")
        print("   Set it with: export NEURONAGENT_API_KEY=your_api_key")
        print("   Or generate one using: ./scripts/generate_api_keys.sh")
        sys.exit(1)
    
    try:
        # Run examples
        example_basic_agent()
        time.sleep(2)
        
        example_research_agent()
        time.sleep(2)
        
        example_data_analyst_agent()
        time.sleep(2)
        
        example_streaming()
        time.sleep(2)
        
        example_error_handling()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

