"""
Tools management module

Provides tool operations:
- List available tools
- Get tool details
- Create custom tools
- Update tools
- Delete tools
"""

import logging
from typing import Dict, List, Optional

from ..core.client import NeuronAgentClient
from ..core.exceptions import NotFoundError

logger = logging.getLogger(__name__)


class ToolManager:
    """
    High-level tool management
    
    Usage:
        client = NeuronAgentClient()
        manager = ToolManager(client)
        
        tools = manager.list()
        tool = manager.get("sql")
    """
    
    def __init__(self, client: NeuronAgentClient):
        """
        Initialize tool manager
        
        Args:
            client: NeuronAgentClient instance
        """
        self.client = client
    
    def list(self) -> List[Dict]:
        """
        List all available tools
        
        Returns:
            List of tool dictionaries
        """
        return self.client.get('/api/v1/tools')
    
    def get(self, name: str) -> Dict:
        """
        Get tool by name
        
        Args:
            name: Tool name
        
        Returns:
            Tool dictionary
        
        Raises:
            NotFoundError: If tool not found
        """
        try:
            return self.client.get(f'/api/v1/tools/{name}')
        except NotFoundError:
            logger.error(f"Tool not found: {name}")
            raise
    
    def create(
        self,
        name: str,
        description: str,
        tool_type: str,
        config: Optional[Dict] = None,
        schema: Optional[Dict] = None
    ) -> Dict:
        """
        Create a new custom tool
        
        Args:
            name: Tool name
            description: Tool description
            tool_type: Tool type
            config: Optional tool configuration
            schema: Optional tool schema
        
        Returns:
            Created tool dictionary
        """
        logger.info(f"Creating tool: {name}")
        
        payload = {
            'name': name,
            'description': description,
            'tool_type': tool_type
        }
        
        if config:
            payload['config'] = config
        if schema:
            payload['schema'] = schema
        
        tool = self.client.post('/api/v1/tools', json_data=payload)
        logger.info(f"Tool created: {name}")
        return tool
    
    def update(
        self,
        name: str,
        description: Optional[str] = None,
        config: Optional[Dict] = None,
        schema: Optional[Dict] = None
    ) -> Dict:
        """
        Update an existing tool
        
        Args:
            name: Tool name
            description: New description (optional)
            config: New config (optional)
            schema: New schema (optional)
        
        Returns:
            Updated tool dictionary
        """
        logger.info(f"Updating tool: {name}")
        
        # Get current tool
        tool = self.get(name)
        
        # Update fields
        if description:
            tool['description'] = description
        if config:
            tool['config'] = config
        if schema:
            tool['schema'] = schema
        
        updated = self.client.put(f'/api/v1/tools/{name}', json_data=tool)
        logger.info(f"Tool updated: {name}")
        return updated
    
    def delete(self, name: str) -> None:
        """
        Delete a tool
        
        Args:
            name: Tool name
        """
        logger.info(f"Deleting tool: {name}")
        self.client.delete(f'/api/v1/tools/{name}')
        logger.info(f"Tool deleted: {name}")



