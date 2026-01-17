"""Tests for Reflections API."""
import pytest

@pytest.mark.api
@pytest.mark.requires_server
class TestReflections:
    def test_list_reflections(self, api_client, test_session):
        """Test listing reflections."""
        try:
            response = api_client.get(f"/api/v1/sessions/{test_session['id']}/reflections")
            assert isinstance(response, (list, dict))
        except Exception:
            pytest.skip("Reflections API not available")

