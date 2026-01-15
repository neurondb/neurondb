"""
Evaluation framework module

Provides evaluation operations:
- Evaluation task management
- Evaluation run execution
- Results retrieval
"""

import logging
from typing import Dict, List, Optional
from uuid import UUID

from ..core.client import NeuronAgentClient
from ..core.exceptions import NotFoundError

logger = logging.getLogger(__name__)


class EvaluationManager:
    """
    High-level evaluation management
    
    Usage:
        client = NeuronAgentClient()
        manager = EvaluationManager(client)
        
        task = manager.create_task(
            task_type="end_to_end",
            input="What is 2+2?",
            expected_output="4"
        )
    """
    
    def __init__(self, client: NeuronAgentClient):
        """
        Initialize evaluation manager
        
        Args:
            client: NeuronAgentClient instance
        """
        self.client = client
    
    def create_task(
        self,
        task_type: str,
        input_text: str,
        expected_output: Optional[str] = None,
        expected_tool_sequence: Optional[Dict] = None,
        golden_sql_side_effects: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Create an evaluation task
        
        Args:
            task_type: Task type (tool_sequence, sql_side_effect, retrieval, end_to_end)
            input_text: Input text for the task
            expected_output: Expected output (optional)
            expected_tool_sequence: Expected tool sequence (optional)
            golden_sql_side_effects: Golden SQL side effects (optional)
            metadata: Optional metadata
        
        Returns:
            Created evaluation task dictionary
        """
        valid_types = ['tool_sequence', 'sql_side_effect', 'retrieval', 'end_to_end']
        if task_type not in valid_types:
            raise ValueError(f"task_type must be one of {valid_types}")
        
        logger.info(f"Creating evaluation task: {task_type}")
        
        payload = {
            'task_type': task_type,
            'input': input_text
        }
        
        if expected_output:
            payload['expected_output'] = expected_output
        if expected_tool_sequence:
            payload['expected_tool_sequence'] = expected_tool_sequence
        if golden_sql_side_effects:
            payload['golden_sql_side_effects'] = golden_sql_side_effects
        if metadata:
            payload['metadata'] = metadata
        
        task = self.client.post('/api/v1/eval/tasks', json_data=payload)
        logger.info(f"Evaluation task created: {task['id']}")
        return task
    
    def list_tasks(
        self,
        task_type: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Dict]:
        """
        List evaluation tasks
        
        Args:
            task_type: Filter by task type (optional)
            limit: Maximum number of tasks (optional)
            offset: Offset for pagination (optional)
        
        Returns:
            List of evaluation task dictionaries
        """
        params = {}
        if task_type:
            params['task_type'] = task_type
        if limit:
            params['limit'] = limit
        if offset:
            params['offset'] = offset
        
        return self.client.get('/api/v1/eval/tasks', params=params)
    
    def get_task(self, task_id: str) -> Dict:
        """
        Get evaluation task by ID
        
        Args:
            task_id: Task UUID
        
        Returns:
            Evaluation task dictionary
        
        Raises:
            NotFoundError: If task not found
        """
        try:
            return self.client.get(f'/api/v1/eval/tasks/{task_id}')
        except NotFoundError:
            logger.error(f"Evaluation task not found: {task_id}")
            raise
    
    def create_run(
        self,
        dataset_version: str,
        agent_id: Optional[str] = None,
        total_tasks: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Create an evaluation run
        
        Args:
            dataset_version: Dataset version identifier
            agent_id: Optional agent UUID to evaluate
            total_tasks: Optional total number of tasks
            metadata: Optional metadata
        
        Returns:
            Created evaluation run dictionary
        """
        logger.info(f"Creating evaluation run: {dataset_version}")
        
        payload = {'dataset_version': dataset_version}
        
        if agent_id:
            payload['agent_id'] = agent_id
        if total_tasks:
            payload['total_tasks'] = total_tasks
        if metadata:
            payload['metadata'] = metadata
        
        run = self.client.post('/api/v1/eval/runs', json_data=payload)
        logger.info(f"Evaluation run created: {run['id']}")
        return run
    
    def list_runs(
        self,
        dataset_version: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> List[Dict]:
        """
        List evaluation runs
        
        Args:
            dataset_version: Filter by dataset version (optional)
            agent_id: Filter by agent ID (optional)
        
        Returns:
            List of evaluation run dictionaries
        """
        params = {}
        if dataset_version:
            params['dataset_version'] = dataset_version
        if agent_id:
            params['agent_id'] = agent_id
        
        return self.client.get('/api/v1/eval/runs', params=params)
    
    def get_run(self, run_id: str) -> Dict:
        """
        Get evaluation run by ID
        
        Args:
            run_id: Run UUID
        
        Returns:
            Evaluation run dictionary
        
        Raises:
            NotFoundError: If run not found
        """
        try:
            return self.client.get(f'/api/v1/eval/runs/{run_id}')
        except NotFoundError:
            logger.error(f"Evaluation run not found: {run_id}")
            raise
    
    def update_run(
        self,
        run_id: str,
        score: Optional[float] = None,
        passed_tasks: Optional[int] = None,
        failed_tasks: Optional[int] = None
    ) -> Dict:
        """
        Update evaluation run
        
        Args:
            run_id: Run UUID
            score: New score (optional)
            passed_tasks: New passed tasks count (optional)
            failed_tasks: New failed tasks count (optional)
        
        Returns:
            Updated evaluation run dictionary
        """
        logger.info(f"Updating evaluation run: {run_id}")
        
        payload = {}
        if score is not None:
            payload['score'] = score
        if passed_tasks is not None:
            payload['passed_tasks'] = passed_tasks
        if failed_tasks is not None:
            payload['failed_tasks'] = failed_tasks
        
        updated = self.client.put(f'/api/v1/eval/runs/{run_id}', json_data=payload)
        logger.info(f"Evaluation run updated: {run_id}")
        return updated
    
    def execute_run(self, run_id: str) -> Dict:
        """
        Execute an evaluation run
        
        Args:
            run_id: Run UUID
        
        Returns:
            Evaluation run dictionary with execution results
        """
        logger.info(f"Executing evaluation run: {run_id}")
        result = self.client.post(f'/api/v1/eval/runs/{run_id}/execute')
        logger.info(f"Evaluation run executed: {run_id}")
        return result
    
    def get_results(self, run_id: str) -> List[Dict]:
        """
        Get evaluation run results
        
        Args:
            run_id: Run UUID
        
        Returns:
            List of evaluation task result dictionaries
        """
        return self.client.get(f'/api/v1/eval/runs/{run_id}/results')



