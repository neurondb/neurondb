"""
Webhooks management module

Provides webhook operations:
- Create webhook
- List webhooks
- Get webhook
- Update webhook
- Delete webhook
- List webhook deliveries
"""

import logging
from typing import Dict, List, Optional

from ..core.client import NeuronAgentClient
from ..core.exceptions import NotFoundError

logger = logging.getLogger(__name__)


class WebhookManager:
    """
    High-level webhook management
    
    Usage:
        client = NeuronAgentClient()
        manager = WebhookManager(client)
        
        webhook = manager.create(
            url="https://example.com/webhook",
            events=["message.sent", "agent.created"]
        )
    """
    
    def __init__(self, client: NeuronAgentClient):
        """
        Initialize webhook manager
        
        Args:
            client: NeuronAgentClient instance
        """
        self.client = client
    
    def create(
        self,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        enabled: bool = True,
        timeout_seconds: int = 30,
        retry_count: int = 3,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Create a webhook
        
        Args:
            url: Webhook URL
            events: List of events to subscribe to
            secret: Optional webhook secret
            enabled: Whether webhook is enabled
            timeout_seconds: Request timeout in seconds
            retry_count: Number of retry attempts
            metadata: Optional metadata dictionary
        
        Returns:
            Created webhook dictionary
        """
        if not url:
            raise ValueError("url is required")
        if not events:
            raise ValueError("events are required")
        
        logger.info(f"Creating webhook: {url}")
        
        payload = {
            'url': url,
            'events': events,
            'enabled': enabled,
            'timeout_seconds': timeout_seconds,
            'retry_count': retry_count
        }
        
        if secret:
            payload['secret'] = secret
        if metadata:
            payload['metadata'] = metadata
        
        webhook = self.client.post('/api/v1/webhooks', json_data=payload)
        logger.info(f"Webhook created: {webhook['id']}")
        return webhook
    
    def list(self) -> List[Dict]:
        """
        List all webhooks
        
        Returns:
            List of webhook dictionaries
        """
        return self.client.get('/api/v1/webhooks')
    
    def get(self, webhook_id: str) -> Dict:
        """
        Get webhook by ID
        
        Args:
            webhook_id: Webhook UUID
        
        Returns:
            Webhook dictionary
        
        Raises:
            NotFoundError: If webhook not found
        """
        try:
            return self.client.get(f'/api/v1/webhooks/{webhook_id}')
        except NotFoundError:
            logger.error(f"Webhook not found: {webhook_id}")
            raise
    
    def update(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        events: Optional[List[str]] = None,
        secret: Optional[str] = None,
        enabled: Optional[bool] = None,
        timeout_seconds: Optional[int] = None,
        retry_count: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Update a webhook
        
        Args:
            webhook_id: Webhook UUID
            url: New URL (optional)
            events: New events list (optional)
            secret: New secret (optional)
            enabled: New enabled status (optional)
            timeout_seconds: New timeout (optional)
            retry_count: New retry count (optional)
            metadata: New metadata (optional)
        
        Returns:
            Updated webhook dictionary
        """
        logger.info(f"Updating webhook: {webhook_id}")
        
        # Get current webhook
        webhook = self.get(webhook_id)
        
        # Update fields
        payload = {}
        if url:
            payload['url'] = url
        if events:
            payload['events'] = events
        if secret is not None:
            payload['secret'] = secret
        if enabled is not None:
            payload['enabled'] = enabled
        if timeout_seconds:
            payload['timeout_seconds'] = timeout_seconds
        if retry_count:
            payload['retry_count'] = retry_count
        if metadata:
            payload['metadata'] = metadata
        
        updated = self.client.put(f'/api/v1/webhooks/{webhook_id}', json_data=payload)
        logger.info(f"Webhook updated: {webhook_id}")
        return updated
    
    def delete(self, webhook_id: str) -> None:
        """
        Delete a webhook
        
        Args:
            webhook_id: Webhook UUID
        """
        logger.info(f"Deleting webhook: {webhook_id}")
        self.client.delete(f'/api/v1/webhooks/{webhook_id}')
        logger.info(f"Webhook deleted: {webhook_id}")
    
    def list_deliveries(
        self,
        webhook_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """
        List webhook deliveries
        
        Args:
            webhook_id: Webhook UUID
            limit: Maximum number of deliveries
            offset: Offset for pagination
        
        Returns:
            List of delivery dictionaries
        """
        params = {'limit': limit, 'offset': offset}
        return self.client.get(
            f'/api/v1/webhooks/{webhook_id}/deliveries',
            params=params
        )



