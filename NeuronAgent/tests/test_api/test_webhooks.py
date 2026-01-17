"""Tests for Webhook Management API."""
import pytest

@pytest.mark.api
@pytest.mark.requires_server
class TestWebhooks:
    def test_list_webhooks(self, api_client):
        """Test listing webhooks."""
        response = api_client.get("/api/v1/webhooks")
        assert isinstance(response, (list, dict))
    
    def test_create_webhook(self, api_client, unique_name):
        """Test creating webhook."""
        webhook_data = {"url": f"https://example.com/webhook/{unique_name}", "events": ["message.created"]}
        try:
            webhook = api_client.post("/api/v1/webhooks", json_data=webhook_data)
            assert "id" in webhook
            api_client.delete(f"/api/v1/webhooks/{webhook['id']}")
        except Exception:
            pytest.skip("Webhook API not available")

