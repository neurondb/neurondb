"""
Core NeuronAgent HTTP client

Provides low-level HTTP client functionality with:
- Retry logic
- Connection pooling
- Error handling
- Request/response logging
"""

import json
import logging
import os
import time
from typing import Dict, Optional, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import (
    NeuronAgentError,
    AuthenticationError,
    NotFoundError,
    ServerError,
    ConnectionError,
    TimeoutError
)

logger = logging.getLogger(__name__)


class NeuronAgentClient:
    """
    Core HTTP client for NeuronAgent API
    
    Features:
    - Automatic retries with exponential backoff
    - Connection pooling
    - Request/response logging
    - Comprehensive error handling
    - Metrics collection
    """
    
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        max_retries: int = 3,
        timeout: int = 30,
        retry_backoff: float = 1.0,
        enable_logging: bool = True
    ):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of NeuronAgent server (default: http://localhost:8080)
            api_key: API key for authentication
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            retry_backoff: Backoff multiplier for retries
            enable_logging: Enable request/response logging
        """
        self.base_url = (base_url or os.getenv('NEURONAGENT_BASE_URL') or 
                        'http://localhost:8080').rstrip('/')
        self.api_key = api_key or os.getenv('NEURONAGENT_API_KEY')
        self.timeout = timeout
        self.enable_logging = enable_logging
        
        if not self.api_key:
            raise ValueError(
                "API key required. Set NEURONAGENT_API_KEY env var or pass api_key parameter"
            )
        
        # Create session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'NeuronAgent-Python-Client/1.0.0'
        }
        
        # Metrics
        self._metrics = {
            'requests': 0,
            'errors': 0,
            'tokens_used': 0,
            'total_time': 0.0
        }
    
    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        **kwargs
    ) -> requests.Response:
        """
        Make authenticated HTTP request
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., '/api/v1/agents')
            params: Query parameters
            json_data: JSON request body
            **kwargs: Additional request arguments
        
        Returns:
            Response object
        
        Raises:
            AuthenticationError: If authentication fails
            NotFoundError: If resource not found
            ServerError: If server error occurs
            ConnectionError: If connection fails
            TimeoutError: If request times out
        """
        url = f"{self.base_url}{path}"
        headers = {**self.headers, **kwargs.pop('headers', {})}
        
        start_time = time.time()
        
        try:
            if self.enable_logging:
                logger.debug(f"{method} {url}")
            
            response = self.session.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=self.timeout,
                **kwargs
            )
            
            elapsed = time.time() - start_time
            self._metrics['requests'] += 1
            self._metrics['total_time'] += elapsed
            
            if self.enable_logging:
                logger.debug(f"Response: {response.status_code} ({elapsed:.2f}s)")
            
            # Handle errors
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code == 404:
                raise NotFoundError("Resource", path)
            elif response.status_code >= 500:
                self._metrics['errors'] += 1
                raise ServerError(response.status_code, "Server error")
            elif response.status_code >= 400:
                self._metrics['errors'] += 1
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', 'Request failed')
                except:
                    error_msg = response.text or 'Request failed'
                raise ServerError(response.status_code, error_msg)
            
            return response
            
        except requests.exceptions.Timeout:
            self._metrics['errors'] += 1
            raise TimeoutError(f"Request to {url} timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError as e:
            self._metrics['errors'] += 1
            raise ConnectionError(f"Failed to connect to {self.base_url}: {e}")
        except (AuthenticationError, NotFoundError, ServerError):
            raise
        except Exception as e:
            self._metrics['errors'] += 1
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise NeuronAgentError(f"Request failed: {e}") from e
    
    def get(self, path: str, params: Optional[Dict] = None, **kwargs) -> Dict:
        """GET request"""
        response = self._request('GET', path, params=params, **kwargs)
        return response.json()
    
    def post(self, path: str, json_data: Optional[Dict] = None, **kwargs) -> Dict:
        """POST request"""
        response = self._request('POST', path, json_data=json_data, **kwargs)
        return response.json()
    
    def put(self, path: str, json_data: Optional[Dict] = None, **kwargs) -> Dict:
        """PUT request"""
        response = self._request('PUT', path, json_data=json_data, **kwargs)
        return response.json()
    
    def delete(self, path: str, **kwargs) -> None:
        """DELETE request"""
        self._request('DELETE', path, **kwargs)
    
    def health_check(self) -> bool:
        """
        Check if server is healthy
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics"""
        avg_time = 0.0
        if self._metrics['requests'] > 0:
            avg_time = self._metrics['total_time'] / self._metrics['requests']
        
        return {
            **self._metrics,
            'average_request_time': avg_time,
            'error_rate': (
                self._metrics['errors'] / self._metrics['requests']
                if self._metrics['requests'] > 0 else 0.0
            )
        }
    
    def reset_metrics(self) -> None:
        """Reset metrics"""
        self._metrics = {
            'requests': 0,
            'errors': 0,
            'tokens_used': 0,
            'total_time': 0.0
        }
    
    def close(self) -> None:
        """Close the session"""
        self.session.close()

