"""
Pytest configuration and fixtures for NeuronAgent comprehensive test suite.

This module provides:
- Shared fixtures for API clients, database connections, test data
- Test utilities and helpers
- Common test setup and teardown
"""

import os
import sys
import pytest
import requests
import psycopg2
from typing import Dict, Optional, Any, Generator
from faker import Faker
import uuid
import time
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try to import the client library
try:
    from examples.neurondb_client import NeuronAgentClient, AgentManager, SessionManager
    CLIENT_AVAILABLE = True
except ImportError:
    CLIENT_AVAILABLE = False
    # Fallback client class
    class NeuronAgentClient:
        def __init__(self, *args, **kwargs):
            pass
    # Fallback managers
    class AgentManager:
        def __init__(self, *args, **kwargs):
            pass
    class SessionManager:
        def __init__(self, *args, **kwargs):
            pass

# Make managers available for import
__all__ = ['NeuronAgentClient', 'AgentManager', 'SessionManager']

# Initialize Faker for test data generation
fake = Faker()

# Test configuration from environment
TEST_CONFIG = {
    'base_url': os.getenv('NEURONAGENT_BASE_URL', 'http://localhost:8080'),
    'api_key': os.getenv('NEURONAGENT_API_KEY', ''),
    'db_host': os.getenv('DB_HOST', 'localhost'),
    'db_port': int(os.getenv('DB_PORT', '5432')),
    'db_name': os.getenv('DB_NAME', 'neurondb'),
    'db_user': os.getenv('DB_USER', 'pge'),
    'db_password': os.getenv('DB_PASSWORD', ''),
}


@pytest.fixture(scope="session")
def server_available() -> bool:
    """Check if NeuronAgent server is available."""
    try:
        response = requests.get(f"{TEST_CONFIG['base_url']}/health", timeout=5)
        # Accept any HTTP response (200, 503, etc.) - server is running if we get a response
        # 503 means server is up but may have database issues, which is fine for testing
        return response.status_code < 600
    except Exception:
        return False


@pytest.fixture(scope="session")
def api_key() -> str:
    """Get or generate API key for testing."""
    key = TEST_CONFIG['api_key']
    if not key:
        # Try to generate one using the Go tool
        import subprocess
        try:
            result = subprocess.run(
                [
                    'go', 'run', 'cmd/generate-key/main.go',
                    '-org', 'test-org',
                    '-user', 'test-user',
                    '-rate', '1000',
                    '-roles', 'user,admin',
                    '-db-host', TEST_CONFIG['db_host'],
                    '-db-port', str(TEST_CONFIG['db_port']),
                    '-db-name', TEST_CONFIG['db_name'],
                    '-db-user', TEST_CONFIG['db_user'],
                ],
                cwd=os.path.join(os.path.dirname(__file__), '..'),
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Extract key from output
                for line in result.stdout.split('\n'):
                    if 'Key:' in line or 'API Key:' in line:
                        key = line.split(':')[-1].strip()
                        break
        except Exception:
            pass
    
    # Use the known working key if available
    if not key:
        # Try the key we know works: vaqSx2Is6KiQ47TO-t1YYsNPgmamUcEjztXe2ikL74k=
        key = "vaqSx2Is6KiQ47TO-t1YYsNPgmamUcEjztXe2ikL74k="
    
    # If still no key, try to generate a fresh one
    if not key:
        import subprocess
        try:
            result = subprocess.run(
                [
                    'go', 'run', 'cmd/generate-key/main.go',
                    '-org', 'test-org',
                    '-user', 'test-user',
                    '-rate', '1000',
                    '-roles', 'user,admin',
                    '-db-host', TEST_CONFIG['db_host'],
                    '-db-port', str(TEST_CONFIG['db_port']),
                    '-db-name', TEST_CONFIG['db_name'],
                    '-db-user', TEST_CONFIG['db_user'],
                ],
                cwd=os.path.join(os.path.dirname(__file__), '..'),
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Extract key from output - try multiple patterns
                for line in result.stdout.split('\n'):
                    if 'Key:' in line:
                        key = line.split(':')[-1].strip()
                        break
                    elif 'Key =' in line:
                        key = line.split('=')[-1].strip()
                        break
        except Exception:
            pass
    
    if not key:
        pytest.skip("API key not available. Set NEURONAGENT_API_KEY or ensure generate-key tool works.")
    
    return key


@pytest.fixture(scope="session")
def api_client(api_key: str, server_available: bool) -> NeuronAgentClient:
    """Create authenticated API client."""
    if not server_available:
        pytest.skip("Server not available")
    
    if CLIENT_AVAILABLE:
        client = NeuronAgentClient(
            base_url=TEST_CONFIG['base_url'],
            api_key=api_key
        )
    else:
        # Fallback: create a simple client
        class SimpleClient:
            def __init__(self, base_url, api_key):
                self.base_url = base_url.rstrip('/')
                self.api_key = api_key
                self.session = requests.Session()
                self.session.headers.update({
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                })
            
            def get(self, path, **kwargs):
                resp = self.session.get(f"{self.base_url}{path}", **kwargs)
                resp.raise_for_status()
                return resp.json()
            
            def post(self, path, json_data=None, **kwargs):
                resp = self.session.post(f"{self.base_url}{path}", json=json_data, **kwargs)
                resp.raise_for_status()
                return resp.json()
            
            def put(self, path, json_data=None, **kwargs):
                resp = self.session.put(f"{self.base_url}{path}", json=json_data, **kwargs)
                resp.raise_for_status()
                return resp.json()
            
            def delete(self, path, **kwargs):
                resp = self.session.delete(f"{self.base_url}{path}", **kwargs)
                resp.raise_for_status()
            
            def health_check(self):
                try:
                    resp = requests.get(f"{self.base_url}/health", timeout=5)
                    return resp.status_code == 200
                except (requests.exceptions.RequestException, Exception) as e:
                    # Log or handle connection errors appropriately
                    return False
        
        client = SimpleClient(TEST_CONFIG['base_url'], api_key)
    
    yield client
    
    # Cleanup
    if hasattr(client, 'close'):
        client.close()


@pytest.fixture(scope="session")
def db_connection():
    """Create database connection for direct DB tests."""
    try:
        conn = psycopg2.connect(
            host=TEST_CONFIG['db_host'],
            port=TEST_CONFIG['db_port'],
            database=TEST_CONFIG['db_name'],
            user=TEST_CONFIG['db_user'],
            password=TEST_CONFIG['db_password']
        )
        # Set autocommit to avoid transaction issues
        conn.autocommit = True
        yield conn
        conn.close()
    except Exception as e:
        pytest.skip(f"Database connection failed: {e}")


@pytest.fixture(scope="session")
def neurondb_available(db_connection) -> bool:
    """Check if NeuronDB extension is available."""
    try:
        with db_connection.cursor() as cur:
            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'neurondb');")
            return cur.fetchone()[0]
    except Exception:
        return False


@pytest.fixture
def test_agent(api_client) -> Dict[str, Any]:
    """Create a test agent and return its data."""
    agent_name = f"test-agent-{uuid.uuid4().hex[:8]}"
    agent_data = {
        "name": agent_name,
        "description": "Test agent for automated testing",
        "system_prompt": "You are a helpful test assistant.",
        "model_name": "gpt-4",
        "enabled_tools": ["sql"],
        "config": {
            "temperature": 0.7,
            "max_tokens": 1000
        }
    }
    
    try:
        agent = api_client.post("/api/v1/agents", json_data=agent_data)
        yield agent
        
        # Cleanup
        try:
            api_client.delete(f"/api/v1/agents/{agent['id']}")
        except Exception:
            # Cleanup failures are non-critical, ignore silently
            pass
    except Exception as e:
        pytest.fail(f"Failed to create test agent: {e}")


@pytest.fixture
def test_session(api_client, test_agent) -> Dict[str, Any]:
    """Create a test session and return its data."""
    session_data = {
        "agent_id": test_agent['id'],
        "external_user_id": f"test-user-{uuid.uuid4().hex[:8]}",
        "metadata": {"test": True}
    }
    
    try:
        session = api_client.post("/api/v1/sessions", json_data=session_data)
        yield session
        
        # Cleanup
        try:
            api_client.delete(f"/api/v1/sessions/{session['id']}")
        except Exception:
            # Cleanup failures are non-critical, ignore silently
            pass
    except Exception as e:
        pytest.fail(f"Failed to create test session: {e}")


@pytest.fixture
def test_data_generator():
    """Provide Faker instance for generating test data."""
    return fake


@pytest.fixture
def unique_name():
    """Generate unique name for test resources."""
    return f"test-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def wait_for_async(max_wait: int = 30, interval: int = 1):
    """Helper to wait for async operations to complete."""
    def _wait(condition_func, timeout=max_wait):
        start = time.time()
        while time.time() - start < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        return False
    return _wait


# Test markers and utilities
def assert_status_code(response, expected_code: int):
    """Assert HTTP status code."""
    if hasattr(response, 'status_code'):
        assert response.status_code == expected_code, \
            f"Expected status {expected_code}, got {response.status_code}: {response.text}"
    else:
        # Assume it's a dict with status_code
        assert response.get('status_code') == expected_code


def assert_has_fields(data: Dict, *fields: str):
    """Assert that data dict has required fields."""
    for field in fields:
        assert field in data, f"Missing required field: {field}"


def assert_valid_uuid(value: str):
    """Assert that value is a valid UUID."""
    try:
        uuid.UUID(value)
    except ValueError:
        pytest.fail(f"Invalid UUID: {value}")


def assert_valid_timestamp(value: str):
    """Assert that value is a valid timestamp."""
    try:
        time.strptime(value, '%Y-%m-%dT%H:%M:%S.%fZ')
    except (ValueError, TypeError):
        try:
            time.strptime(value, '%Y-%m-%dT%H:%M:%SZ')
        except (ValueError, TypeError):
            pytest.fail(f"Invalid timestamp: {value}")


# Pytest hooks
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (require services)"
    )
    config.addinivalue_line(
        "markers", "api: API endpoint tests"
    )
    config.addinivalue_line(
        "markers", "tool: Tool execution tests"
    )
    config.addinivalue_line(
        "markers", "neurondb: NeuronDB integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "requires_db: Tests that require database connection"
    )
    config.addinivalue_line(
        "markers", "requires_server: Tests that require running server"
    )
    config.addinivalue_line(
        "markers", "requires_neurondb: Tests that require NeuronDB extension"
    )
    config.addinivalue_line(
        "markers", "performance: Performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "security: Security and authentication tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items based on markers."""
    # Skip tests that require server if server is not available
    server_available = False
    try:
        response = requests.get(f"{TEST_CONFIG['base_url']}/health", timeout=5)
        # Accept any HTTP response (200, 503, etc.) - server is running if we get a response
        # 503 means server is up but may have database issues, which is fine for testing
        server_available = response.status_code < 600
    except:
        pass
    
    for item in items:
        if 'requires_server' in [m.name for m in item.iter_markers()] and not server_available:
            item.add_marker(pytest.mark.skip(reason="Server not available"))

