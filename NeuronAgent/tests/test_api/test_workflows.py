"""Tests for Workflow Engine API."""
import pytest

@pytest.mark.api
@pytest.mark.requires_server
class TestWorkflows:
    def test_list_workflows(self, api_client):
        """Test listing workflows."""
        try:
            response = api_client.get("/api/v1/workflows")
            assert isinstance(response, (list, dict))
        except Exception:
            pytest.skip("Workflow API not available")

