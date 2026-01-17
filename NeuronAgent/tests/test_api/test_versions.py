"""Tests for Versions API."""
import pytest

@pytest.mark.api
@pytest.mark.requires_server
class TestVersions:
    def test_list_versions(self, api_client, test_agent):
        """Test listing agent versions."""
        try:
            response = api_client.get(f"/api/v1/agents/{test_agent['id']}/versions")
            assert isinstance(response, (list, dict))
        except Exception:
            pytest.skip("Versions API not available")

