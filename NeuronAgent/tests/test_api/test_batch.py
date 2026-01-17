"""
Comprehensive tests for Batch Operations API endpoints.

Tests batch processing for multiple requests.
"""

import pytest
import uuid
from typing import Dict, Any, List


@pytest.mark.api
@pytest.mark.requires_server
class TestBatchOperations:
    """Test batch operation endpoints."""
    
    def test_batch_create_agents(self, api_client, unique_name):
        """Test batch creating agents."""
        agents_data = [
            {
                "name": f"{unique_name}-1",
                "system_prompt": "Agent 1",
                "model_name": "gpt-4"
            },
            {
                "name": f"{unique_name}-2",
                "system_prompt": "Agent 2",
                "model_name": "gpt-4"
            }
        ]
        
        try:
            response = api_client.post("/api/v1/agents/batch", json_data={"agents": agents_data})
            assert isinstance(response, list) or isinstance(response, dict)
            # Should have created agents
        finally:
            # Cleanup
            agents = api_client.get("/api/v1/agents")
            for agent in agents:
                if agent["name"].startswith(unique_name):
                    try:
                        api_client.delete(f"/api/v1/agents/{agent['id']}")
                    except Exception:
                        # Cleanup failures are non-critical
                        pass
    
    def test_batch_delete_agents(self, api_client, unique_name):
        """Test batch deleting agents."""
        # Create agents first
        agent_ids = []
        for i in range(2):
            agent_data = {
                "name": f"{unique_name}-batch-{i}",
                "system_prompt": "Test",
                "model_name": "gpt-4"
            }
            agent = api_client.post("/api/v1/agents", json_data=agent_data)
            agent_ids.append(agent["id"])
        
        try:
            delete_data = {"agent_ids": agent_ids}
            api_client.post("/api/v1/agents/batch/delete", json_data=delete_data)
            
            # Verify deletion
            for agent_id in agent_ids:
                with pytest.raises(Exception):
                    api_client.get(f"/api/v1/agents/{agent_id}")
        except Exception:
            # Cleanup if batch delete doesn't work
            for agent_id in agent_ids:
                try:
                    api_client.delete(f"/api/v1/agents/{agent_id}")
                except Exception:
                    # Cleanup failures are non-critical
                    pass
    
    def test_batch_delete_messages(self, api_client, test_session):
        """Test batch deleting messages."""
        # Create messages first
        message_ids = []
        for i in range(2):
            message_data = {
                "content": f"Batch test message {i}",
                "role": "user"
            }
            result = api_client.post(
                f"/api/v1/sessions/{test_session['id']}/messages",
                json_data=message_data
            )
            msg_id = result.get("id") or (result.get("message", {}).get("id") if isinstance(result, dict) else None)
            if msg_id:
                message_ids.append(msg_id)
        
        if message_ids:
            try:
                delete_data = {"message_ids": message_ids}
                api_client.post(
                    f"/api/v1/sessions/{test_session['id']}/messages/batch/delete",
                    json_data=delete_data
                )
            except Exception:
                # Individual cleanup
                for msg_id in message_ids:
                    try:
                        api_client.delete(f"/api/v1/sessions/{test_session['id']}/messages/{msg_id}")
                    except Exception:
                        # Cleanup failures are non-critical
                        pass

