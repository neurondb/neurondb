"""
Collaboration module

Provides collaboration operations:
- Create workspace
- Get workspace
- Add participant to workspace
"""

import logging
from typing import Dict, Optional

from ..core.client import NeuronAgentClient
from ..core.exceptions import NotFoundError

logger = logging.getLogger(__name__)


class CollaborationManager:
    """
    High-level collaboration management
    
    Usage:
        client = NeuronAgentClient()
        manager = CollaborationManager(client)
        
        workspace = manager.create_workspace(name="My Workspace")
    """
    
    def __init__(self, client: NeuronAgentClient):
        """
        Initialize collaboration manager
        
        Args:
            client: NeuronAgentClient instance
        """
        self.client = client
    
    def create_workspace(
        self,
        name: str,
        description: Optional[str] = None
    ) -> Dict:
        """
        Create a collaboration workspace
        
        Args:
            name: Workspace name
            description: Optional workspace description
        
        Returns:
            Created workspace dictionary
        """
        if not name:
            raise ValueError("name is required")
        
        logger.info(f"Creating workspace: {name}")
        
        payload = {'name': name}
        if description:
            payload['description'] = description
        
        workspace = self.client.post('/api/v1/workspaces', json_data=payload)
        logger.info(f"Workspace created: {workspace['workspace_id']}")
        return workspace
    
    def get_workspace(self, workspace_id: str) -> Dict:
        """
        Get workspace by ID
        
        Args:
            workspace_id: Workspace UUID
        
        Returns:
            Workspace dictionary with state and participants
        
        Raises:
            NotFoundError: If workspace not found
        """
        try:
            return self.client.get(f'/api/v1/workspaces/{workspace_id}')
        except NotFoundError:
            logger.error(f"Workspace not found: {workspace_id}")
            raise
    
    def add_participant(
        self,
        workspace_id: str,
        role: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> Dict:
        """
        Add participant to workspace
        
        Args:
            workspace_id: Workspace UUID
            role: Participant role
            user_id: Optional user UUID
            agent_id: Optional agent UUID
        
        Returns:
            Participant dictionary
        """
        if not user_id and not agent_id:
            raise ValueError("Either user_id or agent_id must be provided")
        
        logger.info(f"Adding participant to workspace: {workspace_id}")
        
        payload = {'role': role}
        if user_id:
            payload['user_id'] = user_id
        if agent_id:
            payload['agent_id'] = agent_id
        
        result = self.client.post(
            f'/api/v1/workspaces/{workspace_id}/participants',
            json_data=payload
        )
        logger.info(f"Participant added to workspace: {workspace_id}")
        return result



