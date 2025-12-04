"""
Session management module

Provides session operations:
- Create sessions
- Get session details
- List sessions
- Message management
"""

import logging
from typing import Dict, List, Optional
from uuid import UUID

from ..core.client import NeuronAgentClient
from ..core.exceptions import NotFoundError

logger = logging.getLogger(__name__)


class SessionManager:
    """
    High-level session management
    
    Usage:
        client = NeuronAgentClient()
        manager = SessionManager(client)
        
        session = manager.create(agent_id="...")
        messages = manager.get_messages(session_id="...")
    """
    
    def __init__(self, client: NeuronAgentClient):
        """
        Initialize session manager
        
        Args:
            client: NeuronAgentClient instance
        """
        self.client = client
    
    def create(
        self,
        agent_id: str,
        external_user_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Create a new session
        
        Args:
            agent_id: Agent UUID
            external_user_id: Optional external user identifier
            metadata: Optional metadata dictionary
        
        Returns:
            Created session dictionary
        """
        logger.info(f"Creating session for agent: {agent_id}")
        
        payload = {'agent_id': agent_id}
        if external_user_id:
            payload['external_user_id'] = external_user_id
        if metadata:
            payload['metadata'] = metadata
        
        session = self.client.post('/api/v1/sessions', json_data=payload)
        logger.info(f"Session created: {session['id']}")
        return session
    
    def get(self, session_id: str) -> Dict:
        """
        Get session by ID
        
        Args:
            session_id: Session UUID
        
        Returns:
            Session dictionary
        
        Raises:
            NotFoundError: If session not found
        """
        try:
            return self.client.get(f'/api/v1/sessions/{session_id}')
        except NotFoundError:
            logger.error(f"Session not found: {session_id}")
            raise
    
    def list_for_agent(self, agent_id: str, limit: int = 50, offset: int = 0) -> List[Dict]:
        """
        List sessions for an agent
        
        Args:
            agent_id: Agent UUID
            limit: Maximum number of sessions
            offset: Offset for pagination
        
        Returns:
            List of session dictionaries
        """
        params = {'limit': limit, 'offset': offset}
        return self.client.get(f'/api/v1/agents/{agent_id}/sessions', params=params)
    
    def send_message(
        self,
        session_id: str,
        content: str,
        role: str = "user",
        stream: bool = False,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Send a message to the agent
        
        Args:
            session_id: Session UUID
            content: Message content
            role: Message role (default: "user")
            stream: Whether to stream response
            metadata: Optional metadata
        
        Returns:
            Response dictionary
        """
        logger.info(f"Sending message to session: {session_id}")
        
        payload = {
            'content': content,
            'role': role,
            'stream': stream
        }
        if metadata:
            payload['metadata'] = metadata
        
        response = self.client.post(
            f'/api/v1/sessions/{session_id}/messages',
            json_data=payload
        )
        
        tokens = response.get('tokens_used', 0)
        logger.info(f"Message sent. Tokens used: {tokens}")
        return response
    
    def get_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        Get messages from a session
        
        Args:
            session_id: Session UUID
            limit: Maximum number of messages
            offset: Offset for pagination
        
        Returns:
            List of message dictionaries
        """
        params = {'limit': limit, 'offset': offset}
        return self.client.get(
            f'/api/v1/sessions/{session_id}/messages',
            params=params
        )

