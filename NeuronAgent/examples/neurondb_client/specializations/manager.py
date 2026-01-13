"""
Agent specializations module

Provides specialization operations:
- Create specialization
- Get specialization
- List specializations
- Update specialization
- Delete specialization
"""

import logging
from typing import Dict, List, Optional

from ..core.client import NeuronAgentClient
from ..core.exceptions import NotFoundError

logger = logging.getLogger(__name__)


class SpecializationManager:
    """
    High-level specialization management
    
    Usage:
        client = NeuronAgentClient()
        manager = SpecializationManager(client)
        
        spec = manager.create(
            agent_id="...",
            specialization_type="coding",
            capabilities=["python", "javascript"]
        )
    """
    
    def __init__(self, client: NeuronAgentClient):
        """
        Initialize specialization manager
        
        Args:
            client: NeuronAgentClient instance
        """
        self.client = client
    
    def create(
        self,
        agent_id: str,
        specialization_type: str,
        capabilities: List[str],
        config: Optional[Dict] = None
    ) -> Dict:
        """
        Create agent specialization
        
        Args:
            agent_id: Agent UUID
            specialization_type: Type (planning, research, coding, execution, analysis, general)
            capabilities: List of capabilities
            config: Optional configuration
        
        Returns:
            Created specialization dictionary
        """
        valid_types = ['planning', 'research', 'coding', 'execution', 'analysis', 'general']
        if specialization_type not in valid_types:
            raise ValueError(f"specialization_type must be one of {valid_types}")
        
        logger.info(f"Creating specialization for agent: {agent_id}")
        
        payload = {
            'specialization_type': specialization_type,
            'capabilities': capabilities
        }
        
        if config:
            payload['config'] = config
        
        spec = self.client.post(
            f'/api/v1/agents/{agent_id}/specialization',
            json_data=payload
        )
        logger.info(f"Specialization created: {spec['id']}")
        return spec
    
    def get(self, agent_id: str) -> Dict:
        """
        Get agent specialization
        
        Args:
            agent_id: Agent UUID
        
        Returns:
            Specialization dictionary
        
        Raises:
            NotFoundError: If specialization not found
        """
        try:
            return self.client.get(f'/api/v1/agents/{agent_id}/specialization')
        except NotFoundError:
            logger.error(f"Specialization not found for agent: {agent_id}")
            raise
    
    def list(self, specialization_type: Optional[str] = None) -> List[Dict]:
        """
        List specializations
        
        Args:
            specialization_type: Filter by type (optional)
        
        Returns:
            List of specialization dictionaries
        """
        params = {}
        if specialization_type:
            params['specialization_type'] = specialization_type
        
        return self.client.get('/api/v1/specializations', params=params)
    
    def update(
        self,
        agent_id: str,
        specialization_type: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ) -> Dict:
        """
        Update agent specialization
        
        Args:
            agent_id: Agent UUID
            specialization_type: New type (optional)
            capabilities: New capabilities (optional)
            config: New config (optional)
        
        Returns:
            Updated specialization dictionary
        """
        logger.info(f"Updating specialization for agent: {agent_id}")
        
        # Get current specialization
        spec = self.get(agent_id)
        
        # Update fields
        if specialization_type:
            valid_types = ['planning', 'research', 'coding', 'execution', 'analysis', 'general']
            if specialization_type not in valid_types:
                raise ValueError(f"specialization_type must be one of {valid_types}")
            spec['specialization_type'] = specialization_type
        if capabilities:
            spec['capabilities'] = capabilities
        if config:
            spec['config'] = config
        
        updated = self.client.put(
            f'/api/v1/agents/{agent_id}/specialization',
            json_data=spec
        )
        logger.info(f"Specialization updated: {agent_id}")
        return updated
    
    def delete(self, agent_id: str) -> None:
        """
        Delete agent specialization
        
        Args:
            agent_id: Agent UUID
        """
        logger.info(f"Deleting specialization for agent: {agent_id}")
        self.client.delete(f'/api/v1/agents/{agent_id}/specialization')
        logger.info(f"Specialization deleted: {agent_id}")


