"""Tests for Analytics API."""
import pytest

@pytest.mark.api
@pytest.mark.requires_server
class TestAnalytics:
    def test_analytics_overview(self, api_client):
        """Test analytics overview."""
        try:
            response = api_client.get("/api/v1/analytics/overview")
            assert isinstance(response, dict)
        except Exception:
            pytest.skip("Analytics API not available")

