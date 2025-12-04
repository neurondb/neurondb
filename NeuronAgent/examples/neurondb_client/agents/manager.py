"""
Agent management module

Provides high-level agent operations:
- Create agents from profiles
- List and search agents
- Update and delete agents
- Agent configuration management
"""

import logging
from typing import Dict, List, Optional
from uuid import UUID

from ..core.client import NeuronAgentClient
from ..core.exceptions import NotFoundError
from .profile import AgentProfile

logger = logging.getLogger(__name__)


class AgentManager:
    """
    High-level agent management
    
    Usage:
        client = NeuronAgentClient()
        manager = AgentManager(client)
        
        # Create from profile
        agent = manager.create_from_profile(profile)
        
        # Find agent
        agent = manager.find_by_name("my-agent")
    """
    
    def __init__(self, client: NeuronAgentClient):
        """
        Initialize agent manager
        
        Args:
            client: NeuronAgentClient instance
        """
        self.client = client
    
    def create(
        self,
        name: str,
        system_prompt: str,
        model_name: str = "gpt-4",
        description: Optional[str] = None,
        enabled_tools: Optional[List[str]] = None,
        config: Optional[Dict] = None,
        memory_table: Optional[str] = None
    ) -> Dict:
        """
        Create a new agent
        
        Args:
            name: Unique agent name
            system_prompt: System prompt
            model_name: LLM model name
            description: Optional description
            enabled_tools: List of enabled tools
            config: Model configuration
            memory_table: Optional memory table
        
        Returns:
            Created agent dictionary
        """
        logger.info(f"Creating agent: {name}")
        
        payload = {
            'name': name,
            'system_prompt': system_prompt,
            'model_name': model_name,
        }
        
        if description:
            payload['description'] = description
        if enabled_tools:
            payload['enabled_tools'] = enabled_tools
        if config:
            payload['config'] = config
        if memory_table:
            payload['memory_table'] = memory_table
        
        agent = self.client.post('/api/v1/agents', json_data=payload)
        logger.info(f"Agent created: {agent['id']}")
        return agent
    
    def create_from_profile(self, profile: AgentProfile) -> Dict:
        """
        Create agent from a profile
        
        Args:
            profile: AgentProfile instance
        
        Returns:
            Created agent dictionary
        """
        return self.create(**profile.to_dict())
    
    def get(self, agent_id: str) -> Dict:
        """
        Get agent by ID
        
        Args:
            agent_id: Agent UUID
        
        Returns:
            Agent dictionary
        
        Raises:
            NotFoundError: If agent not found
        """
        try:
            return self.client.get(f'/api/v1/agents/{agent_id}')
        except NotFoundError:
            logger.error(f"Agent not found: {agent_id}")
            raise
    
    def list(self, limit: Optional[int] = None) -> List[Dict]:
        """
        List all agents
        
        Args:
            limit: Optional limit on number of agents
        
        Returns:
            List of agent dictionaries
        """
        agents = self.client.get('/api/v1/agents')
        if limit:
            agents = agents[:limit]
        return agents
    
    def find_by_name(self, name: str) -> Optional[Dict]:
        """
        Find agent by name
        
        Args:
            name: Agent name
        
        Returns:
            Agent dictionary or None if not found
        """
        agents = self.list()
        for agent in agents:
            if agent.get('name') == name:
                return agent
        return None
    
    def update(
        self,
        agent_id: str,
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        description: Optional[str] = None,
        enabled_tools: Optional[List[str]] = None,
        config: Optional[Dict] = None,
        memory_table: Optional[str] = None
    ) -> Dict:
        """
        Update an agent
        
        Args:
            agent_id: Agent UUID
            name: New name (optional)
            system_prompt: New system prompt (optional)
            model_name: New model name (optional)
            description: New description (optional)
            enabled_tools: New enabled tools (optional)
            config: New config (optional)
            memory_table: New memory table (optional)
        
        Returns:
            Updated agent dictionary
        """
        logger.info(f"Updating agent: {agent_id}")
        
        # Get current agent
        agent = self.get(agent_id)
        
        # Update fields
        if name:
            agent['name'] = name
        if system_prompt:
            agent['system_prompt'] = system_prompt
        if model_name:
            agent['model_name'] = model_name
        if description is not None:
            agent['description'] = description
        if enabled_tools:
            agent['enabled_tools'] = enabled_tools
        if config:
            agent['config'] = config
        if memory_table is not None:
            agent['memory_table'] = memory_table
        
        updated = self.client.put(f'/api/v1/agents/{agent_id}', json_data=agent)
        logger.info(f"Agent updated: {agent_id}")
        return updated
    
    def delete(self, agent_id: str) -> None:
        """
        Delete an agent
        
        Args:
            agent_id: Agent UUID
        """
        logger.info(f"Deleting agent: {agent_id}")
        self.client.delete(f'/api/v1/agents/{agent_id}')
        logger.info(f"Agent deleted: {agent_id}")

