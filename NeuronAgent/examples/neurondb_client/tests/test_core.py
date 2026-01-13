"""
Unit tests for core SDK components
"""

import pytest
from unittest.mock import Mock, patch
from neurondb_client import (
    NeuronAgentClient,
    AuthenticationError,
    NotFoundError,
    ServerError,
    ConnectionError,
    TimeoutError
)


class TestNeuronAgentClient:
    """Test NeuronAgentClient"""
    
    def test_init_with_env_vars(self, monkeypatch):
        """Test client initialization with environment variables"""
        monkeypatch.setenv('NEURONAGENT_BASE_URL', 'http://test:8080')
        monkeypatch.setenv('NEURONAGENT_API_KEY', 'test-key')
        
        client = NeuronAgentClient()
        assert client.base_url == 'http://test:8080'
        assert client.api_key == 'test-key'
    
    def test_init_without_api_key(self):
        """Test that missing API key raises error"""
        with pytest.raises(ValueError, match="API key required"):
            NeuronAgentClient(api_key=None)
    
    @patch('neurondb_client.core.client.requests.Session')
    def test_get_request(self, mock_session):
        """Test GET request"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test'}
        mock_session.return_value.request.return_value = mock_response
        
        client = NeuronAgentClient(api_key='test-key')
        result = client.get('/api/v1/agents')
        assert result == {'data': 'test'}
    
    @patch('neurondb_client.core.client.requests.Session')
    def test_post_request(self, mock_session):
        """Test POST request"""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'id': '123'}
        mock_session.return_value.request.return_value = mock_response
        
        client = NeuronAgentClient(api_key='test-key')
        result = client.post('/api/v1/agents', json_data={'name': 'test'})
        assert result == {'id': '123'}
    
    @patch('neurondb_client.core.client.requests.Session')
    def test_authentication_error(self, mock_session):
        """Test authentication error handling"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_session.return_value.request.return_value = mock_response
        
        client = NeuronAgentClient(api_key='test-key')
        with pytest.raises(AuthenticationError):
            client.get('/api/v1/agents')
    
    @patch('neurondb_client.core.client.requests.Session')
    def test_not_found_error(self, mock_session):
        """Test not found error handling"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_session.return_value.request.return_value = mock_response
        
        client = NeuronAgentClient(api_key='test-key')
        with pytest.raises(NotFoundError):
            client.get('/api/v1/agents/invalid-id')
    
    @patch('neurondb_client.core.client.requests.Session')
    def test_server_error(self, mock_session):
        """Test server error handling"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {'error': 'Internal error'}
        mock_session.return_value.request.return_value = mock_response
        
        client = NeuronAgentClient(api_key='test-key')
        with pytest.raises(ServerError) as exc_info:
            client.get('/api/v1/agents')
        assert exc_info.value.status_code == 500
    
    @patch('neurondb_client.core.client.requests.Session')
    def test_health_check(self, mock_session):
        """Test health check"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.return_value.get.return_value = mock_response
        
        client = NeuronAgentClient(api_key='test-key')
        assert client.health_check() is True
    
    def test_metrics_collection(self):
        """Test metrics collection"""
        client = NeuronAgentClient(api_key='test-key')
        metrics = client.get_metrics()
        assert 'requests' in metrics
        assert 'errors' in metrics
        assert 'average_request_time' in metrics


class TestExceptions:
    """Test exception classes"""
    
    def test_authentication_error(self):
        """Test AuthenticationError"""
        error = AuthenticationError("Invalid key")
        assert str(error) == "Invalid key"
    
    def test_not_found_error(self):
        """Test NotFoundError"""
        error = NotFoundError("Agent", "123")
        assert "Agent not found" in str(error)
        assert "123" in str(error)
    
    def test_server_error(self):
        """Test ServerError"""
        error = ServerError(500, "Internal error")
        assert error.status_code == 500
        assert "500" in str(error)


