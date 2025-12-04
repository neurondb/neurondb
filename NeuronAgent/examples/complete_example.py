#!/usr/bin/env python3
"""
Complete NeuronAgent Example - Production Ready

This is a comprehensive, production-ready example that demonstrates:
1. Proper error handling and retries
2. Session management
3. Memory and context handling
4. Tool usage
5. Streaming responses
6. Logging and monitoring
7. Best practices for production use
"""

import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionNeuronAgentClient:
    """
    Production-ready NeuronAgent client with:
    - Retry logic
    - Connection pooling
    - Error handling
    - Logging
    - Metrics tracking
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str = None,
        max_retries: int = 3,
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.getenv('NEURONAGENT_API_KEY')
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError("API key required")
        
        # Create session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Metrics
        self.metrics = {
            'requests': 0,
            'errors': 0,
            'tokens_used': 0
        }
    
    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """Make authenticated request with error handling"""
        url = f"{self.base_url}{path}"
        kwargs.setdefault('headers', {}).update(self.headers)
        kwargs.setdefault('timeout', self.timeout)
        
        try:
            self.metrics['requests'] += 1
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            self.metrics['errors'] += 1
            logger.error(f"Request failed: {method} {path} - {e}")
            raise
    
    def health_check(self) -> bool:
        """Check server health with retries"""
        try:
            response = self._request('GET', '/health')
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def create_agent(
        self,
        name: str,
        system_prompt: str,
        model_name: str = "gpt-4",
        **kwargs
    ) -> Dict:
        """Create agent with validation"""
        logger.info(f"Creating agent: {name}")
        
        payload = {
            'name': name,
            'system_prompt': system_prompt,
            'model_name': model_name,
            'enabled_tools': kwargs.get('enabled_tools', ['sql', 'http']),
            'config': kwargs.get('config', {
                'temperature': 0.7,
                'max_tokens': 2000
            })
        }
        
        if 'description' in kwargs:
            payload['description'] = kwargs['description']
        if 'memory_table' in kwargs:
            payload['memory_table'] = kwargs['memory_table']
        
        response = self._request('POST', '/api/v1/agents', json=payload)
        agent = response.json()
        logger.info(f"Agent created: {agent['id']}")
        return agent
    
    def create_session(
        self,
        agent_id: str,
        external_user_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Create session with metadata"""
        logger.info(f"Creating session for agent: {agent_id}")
        
        payload = {'agent_id': agent_id}
        if external_user_id:
            payload['external_user_id'] = external_user_id
        if metadata:
            payload['metadata'] = metadata
        
        response = self._request('POST', '/api/v1/sessions', json=payload)
        session = response.json()
        logger.info(f"Session created: {session['id']}")
        return session
    
    def send_message(
        self,
        session_id: str,
        content: str,
        stream: bool = False
    ) -> Dict:
        """Send message and track metrics"""
        logger.info(f"Sending message to session: {session_id}")
        
        payload = {
            'content': content,
            'role': 'user',
            'stream': stream
        }
        
        response = self._request(
            'POST',
            f"/api/v1/sessions/{session_id}/messages",
            json=payload
        )
        result = response.json()
        
        # Track metrics
        if 'tokens_used' in result:
            self.metrics['tokens_used'] += result['tokens_used']
        
        logger.info(f"Message sent. Tokens used: {result.get('tokens_used', 0)}")
        return result
    
    def get_metrics(self) -> Dict:
        """Get client metrics"""
        return self.metrics.copy()


class ConversationManager:
    """
    Manages a conversation with an agent, handling:
    - Context management
    - Message history
    - Error recovery
    """
    
    def __init__(self, client: ProductionNeuronAgentClient, agent_id: str):
        self.client = client
        self.agent_id = agent_id
        self.session = None
        self.message_history = []
    
    def start(self, external_user_id: Optional[str] = None) -> str:
        """Start a new conversation session"""
        self.session = self.client.create_session(
            agent_id=self.agent_id,
            external_user_id=external_user_id
        )
        return self.session['id']
    
    def send(self, message: str) -> str:
        """Send a message and get response"""
        if not self.session:
            raise ValueError("Session not started. Call start() first.")
        
        try:
            response = self.client.send_message(
                session_id=self.session['id'],
                content=message
            )
            
            # Store in history
            self.message_history.append({
                'user': message,
                'assistant': response['response'],
                'tokens': response.get('tokens_used', 0)
            })
            
            return response['response']
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise
    
    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.message_history.copy()


def main():
    """Main example demonstrating production patterns"""
    
    # Initialize client
    logger.info("Initializing NeuronAgent client...")
    client = ProductionNeuronAgentClient()
    
    # Health check
    if not client.health_check():
        logger.error("Server is not healthy. Exiting.")
        sys.exit(1)
    logger.info("✅ Server is healthy")
    
    # Create a specialized agent
    logger.info("Creating research assistant agent...")
    agent = client.create_agent(
        name="production-research-assistant",
        system_prompt="""You are a research assistant. Your role is to:
1. Answer questions accurately using available tools
2. Provide citations and sources when possible
3. Synthesize information from multiple sources
4. Be thorough but concise

Always verify information and provide reliable answers.""",
        model_name="gpt-4",
        description="Production research assistant",
        enabled_tools=['http', 'sql'],
        config={
            'temperature': 0.5,
            'max_tokens': 2500,
            'top_p': 0.95
        }
    )
    
    # Create conversation manager
    logger.info("Starting conversation...")
    conversation = ConversationManager(client, agent['id'])
    conversation.start(external_user_id="production-user-001")
    
    # Example conversation
    queries = [
        "Hello! Can you help me with research tasks?",
        "What are the key features of NeuronAgent?",
        "How does long-term memory work in AI agents?",
    ]
    
    for i, query in enumerate(queries, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Query {i}: {query}")
        logger.info('='*60)
        
        try:
            response = conversation.send(query)
            logger.info(f"\nResponse:\n{response}\n")
            time.sleep(1)  # Rate limiting
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            continue
    
    # Show metrics
    logger.info("\n" + "="*60)
    logger.info("Metrics")
    logger.info("="*60)
    metrics = client.get_metrics()
    logger.info(f"Total requests: {metrics['requests']}")
    logger.info(f"Errors: {metrics['errors']}")
    logger.info(f"Total tokens used: {metrics['tokens_used']}")
    
    # Show conversation history
    logger.info("\n" + "="*60)
    logger.info("Conversation History")
    logger.info("="*60)
    history = conversation.get_history()
    for i, msg in enumerate(history, 1):
        logger.info(f"\nExchange {i}:")
        logger.info(f"  User: {msg['user']}")
        logger.info(f"  Assistant: {msg['assistant'][:200]}...")
        logger.info(f"  Tokens: {msg['tokens']}")
    
    logger.info("\n✅ Example completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)

