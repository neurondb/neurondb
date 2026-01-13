"""
Workflow management module

Provides workflow operations:
- Workflow schedule CRUD
- Workflow execution
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from ..core.client import NeuronAgentClient
from ..core.exceptions import NotFoundError

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    High-level workflow management
    
    Usage:
        client = NeuronAgentClient()
        manager = WorkflowManager(client)
        
        schedule = manager.create_schedule(
            workflow_id="...",
            cron_expression="0 0 * * *",
            timezone="UTC"
        )
    """
    
    def __init__(self, client: NeuronAgentClient):
        """
        Initialize workflow manager
        
        Args:
            client: NeuronAgentClient instance
        """
        self.client = client
    
    def create_schedule(
        self,
        workflow_id: str,
        cron_expression: str,
        timezone: str = "UTC",
        enabled: bool = True,
        next_run_at: Optional[datetime] = None
    ) -> Dict:
        """
        Create or update workflow schedule
        
        Args:
            workflow_id: Workflow UUID
            cron_expression: Cron expression for schedule
            timezone: Timezone (default: UTC)
            enabled: Whether schedule is enabled
            next_run_at: Optional next run time
        
        Returns:
            Workflow schedule dictionary
        """
        logger.info(f"Creating schedule for workflow: {workflow_id}")
        
        payload = {
            'cron_expression': cron_expression,
            'timezone': timezone,
            'enabled': enabled
        }
        
        if next_run_at:
            payload['next_run_at'] = next_run_at.isoformat()
        
        schedule = self.client.post(
            f'/api/v1/workflows/{workflow_id}/schedule',
            json_data=payload
        )
        logger.info(f"Schedule created: {schedule['id']}")
        return schedule
    
    def get_schedule(self, workflow_id: str) -> Dict:
        """
        Get workflow schedule
        
        Args:
            workflow_id: Workflow UUID
        
        Returns:
            Workflow schedule dictionary
        
        Raises:
            NotFoundError: If schedule not found
        """
        try:
            return self.client.get(f'/api/v1/workflows/{workflow_id}/schedule')
        except NotFoundError:
            logger.error(f"Schedule not found for workflow: {workflow_id}")
            raise
    
    def update_schedule(
        self,
        workflow_id: str,
        cron_expression: Optional[str] = None,
        timezone: Optional[str] = None,
        enabled: Optional[bool] = None,
        next_run_at: Optional[datetime] = None
    ) -> Dict:
        """
        Update workflow schedule
        
        Args:
            workflow_id: Workflow UUID
            cron_expression: New cron expression (optional)
            timezone: New timezone (optional)
            enabled: New enabled status (optional)
            next_run_at: New next run time (optional)
        
        Returns:
            Updated workflow schedule dictionary
        """
        logger.info(f"Updating schedule for workflow: {workflow_id}")
        
        # Get current schedule
        schedule = self.get_schedule(workflow_id)
        
        # Update fields
        if cron_expression:
            schedule['cron_expression'] = cron_expression
        if timezone:
            schedule['timezone'] = timezone
        if enabled is not None:
            schedule['enabled'] = enabled
        if next_run_at:
            schedule['next_run_at'] = next_run_at.isoformat()
        
        updated = self.client.put(
            f'/api/v1/workflows/{workflow_id}/schedule',
            json_data=schedule
        )
        logger.info(f"Schedule updated: {workflow_id}")
        return updated
    
    def delete_schedule(self, workflow_id: str) -> None:
        """
        Delete workflow schedule
        
        Args:
            workflow_id: Workflow UUID
        """
        logger.info(f"Deleting schedule for workflow: {workflow_id}")
        self.client.delete(f'/api/v1/workflows/{workflow_id}/schedule')
        logger.info(f"Schedule deleted: {workflow_id}")
    
    def list_schedules(self) -> List[Dict]:
        """
        List all workflow schedules
        
        Returns:
            List of workflow schedule dictionaries
        """
        return self.client.get('/api/v1/workflow-schedules')


