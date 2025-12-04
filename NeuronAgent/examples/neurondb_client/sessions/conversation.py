"""
Conversation management

Provides high-level conversation handling with:
- Message history tracking
- Context management
- Streaming support
"""

import logging
from typing import Dict, List, Optional, Callable, Any
from uuid import UUID

from ..core.client import NeuronAgentClient
from ..core.websocket import WebSocketClient
from .manager import SessionManager

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages a conversation with an agent
    
    Features:
    - Message history tracking
    - Context management
    - Streaming support
    - Error recovery
    
    Usage:
        client = NeuronAgentClient()
        conversation = ConversationManager(client, agent_id="...")
        conversation.start()
        response = conversation.send("Hello!")
    """
    
    def __init__(
        self,
        client: NeuronAgentClient,
        agent_id: str,
        external_user_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize conversation manager
        
        Args:
            client: NeuronAgentClient instance
            agent_id: Agent UUID
            external_user_id: Optional external user ID
            metadata: Optional session metadata
        """
        self.client = client
        self.agent_id = agent_id
        self.external_user_id = external_user_id
        self.metadata = metadata
        
        self.session_manager = SessionManager(client)
        self.session: Optional[Dict] = None
        self.message_history: List[Dict] = []
        self._ws_client: Optional[WebSocketClient] = None
    
    def start(self) -> str:
        """
        Start a new conversation session
        
        Returns:
            Session ID
        """
        if self.session:
            logger.warning("Session already started")
            return self.session['id']
        
        self.session = self.session_manager.create(
            agent_id=self.agent_id,
            external_user_id=self.external_user_id,
            metadata=self.metadata
        )
        
        logger.info(f"Conversation started: {self.session['id']}")
        return self.session['id']
    
    def send(
        self,
        message: str,
        role: str = "user",
        stream: bool = False
    ) -> str:
        """
        Send a message and get response
        
        Args:
            message: Message content
            role: Message role
            stream: Whether to stream response
        
        Returns:
            Agent response text
        """
        if not self.session:
            raise ValueError("Session not started. Call start() first.")
        
        try:
            response = self.session_manager.send_message(
                session_id=self.session['id'],
                content=message,
                role=role,
                stream=stream
            )
            
            # Store in history
            self.message_history.append({
                'user': message,
                'assistant': response.get('response', ''),
                'tokens': response.get('tokens_used', 0),
                'tool_calls': response.get('tool_calls', []),
                'tool_results': response.get('tool_results', [])
            })
            
            return response.get('response', '')
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise
    
    def stream(
        self,
        message: str,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Send a message with streaming response
        
        Args:
            message: Message content
            on_chunk: Callback for each chunk (chunk: str)
            on_complete: Callback when complete (full_response: str)
        
        Returns:
            Full response text
        """
        if not self.session:
            raise ValueError("Session not started. Call start() first.")
        
        if not self._ws_client:
            self._ws_client = WebSocketClient(
                self.client.base_url,
                self.client.api_key
            )
        
        full_response = []
        
        def on_message(data: Dict[str, Any]):
            if data.get('type') == 'response':
                content = data.get('content', '')
                if content:
                    full_response.append(content)
                    if on_chunk:
                        on_chunk(content)
        
        def on_complete_cb():
            response_text = ''.join(full_response)
            if on_complete:
                on_complete(response_text)
            
            # Store in history
            self.message_history.append({
                'user': message,
                'assistant': response_text,
                'tokens': 0,  # Streaming doesn't provide token count
                'streamed': True
            })
        
        self._ws_client.stream_message(
            session_id=self.session['id'],
            content=message,
            on_message=on_message,
            on_complete=on_complete_cb
        )
        
        return ''.join(full_response)
    
    def get_history(self) -> List[Dict]:
        """
        Get conversation history
        
        Returns:
            List of message exchanges
        """
        return self.message_history.copy()
    
    def refresh_history(self, limit: int = 100) -> None:
        """
        Refresh history from server
        
        Args:
            limit: Maximum number of messages to fetch
        """
        if not self.session:
            return
        
        messages = self.session_manager.get_messages(
            session_id=self.session['id'],
            limit=limit
        )
        
        # Rebuild history from messages
        self.message_history = []
        current_exchange = {}
        
        for msg in messages:
            if msg['role'] == 'user':
                if current_exchange:
                    self.message_history.append(current_exchange)
                current_exchange = {'user': msg['content']}
            elif msg['role'] == 'assistant':
                if 'user' in current_exchange:
                    current_exchange['assistant'] = msg['content']
                    current_exchange['tokens'] = msg.get('token_count', 0)
                    self.message_history.append(current_exchange)
                    current_exchange = {}
        
        if current_exchange:
            self.message_history.append(current_exchange)
    
    def get_total_tokens(self) -> int:
        """Get total tokens used in conversation"""
        return sum(msg.get('tokens', 0) for msg in self.message_history)
    
    def close(self) -> None:
        """Close conversation and cleanup"""
        if self._ws_client:
            self._ws_client.close()
        self.session = None
        self.message_history = []

