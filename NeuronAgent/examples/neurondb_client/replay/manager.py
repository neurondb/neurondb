"""
Replay and snapshots module

Provides snapshot and replay operations:
- Create execution snapshots
- List snapshots
- Get snapshot details
- Delete snapshots
- Replay execution from snapshot
"""

import logging
from typing import Dict, List, Optional

from ..core.client import NeuronAgentClient
from ..core.exceptions import NotFoundError

logger = logging.getLogger(__name__)


class ReplayManager:
    """
    High-level replay and snapshot management
    
    Usage:
        client = NeuronAgentClient()
        manager = ReplayManager(client)
        
        snapshot = manager.create_snapshot(
            session_id="...",
            user_message="Hello"
        )
    """
    
    def __init__(self, client: NeuronAgentClient):
        """
        Initialize replay manager
        
        Args:
            client: NeuronAgentClient instance
        """
        self.client = client
    
    def create_snapshot(
        self,
        session_id: str,
        user_message: str,
        deterministic_mode: bool = False
    ) -> Dict:
        """
        Create an execution snapshot
        
        Args:
            session_id: Session UUID
            user_message: User message to snapshot
            deterministic_mode: Whether to use deterministic mode
        
        Returns:
            Created snapshot dictionary
        """
        logger.info(f"Creating snapshot for session: {session_id}")
        
        payload = {
            'user_message': user_message,
            'deterministic_mode': deterministic_mode
        }
        
        snapshot = self.client.post(
            f'/api/v1/sessions/{session_id}/snapshots',
            json_data=payload
        )
        logger.info(f"Snapshot created: {snapshot['id']}")
        return snapshot
    
    def list_by_session(self, session_id: str) -> List[Dict]:
        """
        List snapshots for a session
        
        Args:
            session_id: Session UUID
        
        Returns:
            List of snapshot dictionaries
        """
        return self.client.get(f'/api/v1/sessions/{session_id}/snapshots')
    
    def list_by_agent(self, agent_id: str) -> List[Dict]:
        """
        List snapshots for an agent
        
        Args:
            agent_id: Agent UUID
        
        Returns:
            List of snapshot dictionaries
        """
        return self.client.get(f'/api/v1/agents/{agent_id}/snapshots')
    
    def get(self, snapshot_id: str) -> Dict:
        """
        Get snapshot by ID
        
        Args:
            snapshot_id: Snapshot UUID
        
        Returns:
            Snapshot dictionary
        
        Raises:
            NotFoundError: If snapshot not found
        """
        try:
            return self.client.get(f'/api/v1/snapshots/{snapshot_id}')
        except NotFoundError:
            logger.error(f"Snapshot not found: {snapshot_id}")
            raise
    
    def delete(self, snapshot_id: str) -> None:
        """
        Delete a snapshot
        
        Args:
            snapshot_id: Snapshot UUID
        """
        logger.info(f"Deleting snapshot: {snapshot_id}")
        self.client.delete(f'/api/v1/snapshots/{snapshot_id}')
        logger.info(f"Snapshot deleted: {snapshot_id}")
    
    def replay(self, snapshot_id: str) -> Dict:
        """
        Replay execution from a snapshot
        
        Args:
            snapshot_id: Snapshot UUID
        
        Returns:
            Replay response dictionary with:
            - session_id
            - agent_id
            - user_message
            - final_answer
            - tool_calls
            - tool_results
            - tokens_used
        """
        logger.info(f"Replaying snapshot: {snapshot_id}")
        result = self.client.post(f'/api/v1/snapshots/{snapshot_id}/replay')
        logger.info(f"Snapshot replayed: {snapshot_id}")
        return result



