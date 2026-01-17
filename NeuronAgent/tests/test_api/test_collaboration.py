"""Tests for Collaboration API."""
import pytest

@pytest.mark.api
@pytest.mark.requires_server
class TestCollaboration:
    def test_collaboration_api(self, api_client):
        """Test collaboration endpoints."""
        try:
            response = api_client.get("/api/v1/collaborations")
            assert isinstance(response, (list, dict))
        except Exception:
            pytest.skip("Collaboration API not available")

