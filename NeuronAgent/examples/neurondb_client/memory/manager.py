"""
Memory management module

Provides memory operations:
- List memory chunks
- Get memory chunk
- Delete memory chunk
- Search memory
- Summarize memory
"""

import logging
from typing import Dict, List, Optional

from ..core.client import NeuronAgentClient
from ..core.exceptions import NotFoundError

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    High-level memory management
    
    Usage:
        client = NeuronAgentClient()
        manager = MemoryManager(client)
        
        chunks = manager.list_chunks(agent_id="...")
        results = manager.search(agent_id="...", query="...")
    """
    
    def __init__(self, client: NeuronAgentClient):
        """
        Initialize memory manager
        
        Args:
            client: NeuronAgentClient instance
        """
        self.client = client
    
    def list_chunks(
        self,
        agent_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """
        List memory chunks for an agent
        
        Args:
            agent_id: Agent UUID
            limit: Maximum number of chunks
            offset: Offset for pagination
        
        Returns:
            List of memory chunk dictionaries
        """
        params = {'limit': limit, 'offset': offset}
        return self.client.get(
            f'/api/v1/agents/{agent_id}/memory',
            params=params
        )
    
    def get_chunk(self, chunk_id: int) -> Dict:
        """
        Get memory chunk by ID
        
        Args:
            chunk_id: Memory chunk ID
        
        Returns:
            Memory chunk dictionary
        
        Raises:
            NotFoundError: If chunk not found
        """
        try:
            return self.client.get(f'/api/v1/memory/{chunk_id}')
        except NotFoundError:
            logger.error(f"Memory chunk not found: {chunk_id}")
            raise
    
    def delete_chunk(self, chunk_id: int) -> None:
        """
        Delete a memory chunk
        
        Args:
            chunk_id: Memory chunk ID
        """
        logger.info(f"Deleting memory chunk: {chunk_id}")
        self.client.delete(f'/api/v1/memory/{chunk_id}')
        logger.info(f"Memory chunk deleted: {chunk_id}")
    
    def search(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search memory chunks by query text
        
        Args:
            agent_id: Agent UUID
            query: Search query text
            top_k: Number of results to return (1-100, default: 5)
        
        Returns:
            List of matching memory chunk dictionaries with similarity scores
        """
        if not query:
            raise ValueError("query is required")
        if top_k <= 0:
            top_k = 5
        if top_k > 100:
            top_k = 100
        
        logger.info(f"Searching memory for agent: {agent_id}")
        
        payload = {
            'query': query,
            'top_k': top_k
        }
        
        results = self.client.post(
            f'/api/v1/agents/{agent_id}/memory/search',
            json_data=payload
        )
        logger.info(f"Memory search completed: {len(results)} results")
        return results
    
    def summarize(
        self,
        memory_id: int,
        max_length: Optional[int] = None
    ) -> Dict:
        """
        Summarize a memory chunk
        
        Args:
            memory_id: Memory chunk ID
            max_length: Optional maximum summary length
        
        Returns:
            Summary dictionary
        """
        logger.info(f"Summarizing memory: {memory_id}")
        
        payload = {}
        if max_length:
            payload['max_length'] = max_length
        
        result = self.client.post(
            f'/api/v1/memory/{memory_id}/summarize',
            json_data=payload
        )
        logger.info(f"Memory summarized: {memory_id}")
        return result



