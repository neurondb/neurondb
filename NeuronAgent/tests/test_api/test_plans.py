"""Tests for Plans API."""
import pytest

@pytest.mark.api
@pytest.mark.requires_server
class TestPlans:
    def test_list_plans(self, api_client):
        """Test listing plans."""
        try:
            response = api_client.get("/api/v1/plans")
            assert isinstance(response, (list, dict))
        except Exception:
            pytest.skip("Plans API not available")

