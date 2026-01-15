"""
Plans and reflections module

Provides plan operations:
- List plans
- Get plan
- Update plan status
"""

import logging
from typing import Dict, List, Optional

from ..core.client import NeuronAgentClient
from ..core.exceptions import NotFoundError

logger = logging.getLogger(__name__)


class PlanManager:
    """
    High-level plan management
    
    Usage:
        client = NeuronAgentClient()
        manager = PlanManager(client)
        
        plans = manager.list(agent_id="...")
        plan = manager.get(plan_id="...")
    """
    
    def __init__(self, client: NeuronAgentClient):
        """
        Initialize plan manager
        
        Args:
            client: NeuronAgentClient instance
        """
        self.client = client
    
    def list(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """
        List plans with optional filters
        
        Args:
            agent_id: Filter by agent ID (optional)
            session_id: Filter by session ID (optional)
            limit: Maximum number of plans
            offset: Offset for pagination
        
        Returns:
            List of plan dictionaries
        """
        params = {'limit': limit, 'offset': offset}
        if agent_id:
            params['agent_id'] = agent_id
        if session_id:
            params['session_id'] = session_id
        
        return self.client.get('/api/v1/plans', params=params)
    
    def get(self, plan_id: str) -> Dict:
        """
        Get plan by ID
        
        Args:
            plan_id: Plan UUID
        
        Returns:
            Plan dictionary
        
        Raises:
            NotFoundError: If plan not found
        """
        try:
            return self.client.get(f'/api/v1/plans/{plan_id}')
        except NotFoundError:
            logger.error(f"Plan not found: {plan_id}")
            raise
    
    def update_status(
        self,
        plan_id: str,
        status: str,
        result: Optional[Dict] = None
    ) -> Dict:
        """
        Update plan status
        
        Args:
            plan_id: Plan UUID
            status: New status (created, executing, completed, failed, cancelled)
            result: Optional result dictionary
        
        Returns:
            Updated plan dictionary
        """
        valid_statuses = ['created', 'executing', 'completed', 'failed', 'cancelled']
        if status not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")
        
        logger.info(f"Updating plan status: {plan_id} -> {status}")
        
        payload = {'status': status}
        if result:
            payload['result'] = result
        
        updated = self.client.put(f'/api/v1/plans/{plan_id}', json_data=payload)
        logger.info(f"Plan status updated: {plan_id}")
        return updated



