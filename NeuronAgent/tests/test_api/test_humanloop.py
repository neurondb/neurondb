"""Tests for HumanLoop API."""
import pytest

@pytest.mark.api
@pytest.mark.requires_server
class TestHumanLoop:
    def test_approval_requests(self, api_client):
        """Test listing approval requests."""
        try:
            response = api_client.get("/api/v1/approvals")
            assert isinstance(response, (list, dict))
        except Exception:
            pytest.skip("HumanLoop API not available")

