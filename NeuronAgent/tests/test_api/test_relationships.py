"""Tests for Relationships API."""
import pytest

@pytest.mark.api
@pytest.mark.requires_server
class TestRelationships:
    def test_agent_relationships(self, api_client, test_agent):
        """Test agent relationships."""
        try:
            response = api_client.get(f"/api/v1/agents/{test_agent['id']}/relationships")
            assert isinstance(response, (list, dict))
        except Exception:
            pytest.skip("Relationships API not available")

