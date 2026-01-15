"""
Budget management module

Provides budget operations:
- Get budget status
- Set budget limits
- Update budget
- Budget usage tracking
"""

import logging
from typing import Dict, Optional
from datetime import datetime

from ..core.client import NeuronAgentClient
from ..core.exceptions import NotFoundError

logger = logging.getLogger(__name__)


class BudgetManager:
    """
    High-level budget management
    
    Usage:
        client = NeuronAgentClient()
        manager = BudgetManager(client)
        
        budget = manager.get_budget(agent_id="...", period_type="monthly")
        manager.set_budget(agent_id="...", budget_amount=1000.0, period_type="monthly")
    """
    
    def __init__(self, client: NeuronAgentClient):
        """
        Initialize budget manager
        
        Args:
            client: NeuronAgentClient instance
        """
        self.client = client
    
    def get_budget(
        self,
        agent_id: str,
        period_type: str = "monthly"
    ) -> Dict:
        """
        Get budget status for an agent
        
        Args:
            agent_id: Agent UUID
            period_type: Period type (daily, weekly, monthly, yearly, total)
        
        Returns:
            Budget status dictionary with:
            - agent_id
            - period_type
            - budget_set (bool)
            - budget_amount (if set)
            - spent (if budget is set)
            - remaining (if budget is set)
            - usage_percentage (if budget is set)
        """
        valid_periods = ['daily', 'weekly', 'monthly', 'yearly', 'total']
        if period_type not in valid_periods:
            raise ValueError(f"period_type must be one of {valid_periods}")
        
        params = {'period_type': period_type}
        return self.client.get(
            f'/api/v1/agents/{agent_id}/budget',
            params=params
        )
    
    def set_budget(
        self,
        agent_id: str,
        budget_amount: float,
        period_type: str = "monthly",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Set budget for an agent
        
        Args:
            agent_id: Agent UUID
            budget_amount: Budget amount
            period_type: Period type (daily, weekly, monthly, yearly, total)
            start_date: Optional start date
            end_date: Optional end date
            metadata: Optional metadata dictionary
        
        Returns:
            Created budget dictionary
        """
        valid_periods = ['daily', 'weekly', 'monthly', 'yearly', 'total']
        if period_type not in valid_periods:
            raise ValueError(f"period_type must be one of {valid_periods}")
        
        if budget_amount < 0:
            raise ValueError("budget_amount must be >= 0")
        
        logger.info(f"Setting budget for agent: {agent_id}")
        
        payload = {
            'budget_amount': budget_amount,
            'period_type': period_type
        }
        
        if start_date:
            payload['start_date'] = start_date.isoformat()
        if end_date:
            payload['end_date'] = end_date.isoformat()
        if metadata:
            payload['metadata'] = metadata
        
        budget = self.client.post(
            f'/api/v1/agents/{agent_id}/budget',
            json_data=payload
        )
        logger.info(f"Budget set: {budget['id']}")
        return budget
    
    def update_budget(
        self,
        agent_id: str,
        budget_amount: Optional[float] = None,
        period_type: str = "monthly",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Update existing budget
        
        Args:
            agent_id: Agent UUID
            budget_amount: New budget amount (optional)
            period_type: Period type (default: monthly)
            start_date: New start date (optional)
            end_date: New end date (optional)
            metadata: New metadata (optional)
        
        Returns:
            Updated budget dictionary
        """
        logger.info(f"Updating budget for agent: {agent_id}")
        
        params = {'period_type': period_type}
        payload = {}
        
        if budget_amount is not None:
            if budget_amount < 0:
                raise ValueError("budget_amount must be >= 0")
            payload['budget_amount'] = budget_amount
        
        if start_date:
            payload['start_date'] = start_date.isoformat()
        if end_date:
            payload['end_date'] = end_date.isoformat()
        if metadata:
            payload['metadata'] = metadata
        
        updated = self.client.put(
            f'/api/v1/agents/{agent_id}/budget',
            json_data=payload,
            params=params
        )
        logger.info(f"Budget updated: {agent_id}")
        return updated



