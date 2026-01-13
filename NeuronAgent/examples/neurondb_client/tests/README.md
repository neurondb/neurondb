# NeuronAgent SDK Test Suite

## Overview

Comprehensive test suite for the NeuronAgent Python SDK covering unit tests, integration tests, and example verification.

## Test Structure

```
tests/
├── __init__.py
├── test_core.py          # Core client tests
├── test_agents.py        # Agent management tests
├── test_sessions.py      # Session management tests
├── test_tools.py         # Tool management tests
├── test_workflows.py     # Workflow tests
├── test_budgets.py       # Budget tests
├── test_evaluation.py    # Evaluation tests
├── test_replay.py        # Replay tests
├── test_memory.py        # Memory tests
├── test_webhooks.py      # Webhook tests
├── test_vfs.py          # VFS tests
├── test_collaboration.py # Collaboration tests
└── test_integration.py   # Integration tests
```

## Running Tests

### Unit Tests

```bash
# Run all unit tests
pytest tests/test_core.py -v

# Run with coverage
pytest tests/ --cov=neurondb_client --cov-report=html
```

### Integration Tests

Integration tests require a running NeuronAgent server:

```bash
# Set environment variables
export NEURONAGENT_BASE_URL=http://localhost:8080
export NEURONAGENT_API_KEY=your_api_key

# Run integration tests
pytest tests/test_integration.py -v
```

### All Tests

```bash
# Run all tests
pytest tests/ -v

# Run with verbose output
pytest tests/ -vv

# Run specific test file
pytest tests/test_agents.py -v
```

## Test Requirements

Install test dependencies:

```bash
pip install pytest pytest-cov pytest-mock
```

## Test Coverage

Target: > 80% coverage

Current coverage areas:
- ✅ Core client functionality
- ✅ Error handling
- ✅ All manager classes
- ✅ Integration scenarios

## Writing New Tests

### Unit Test Example

```python
def test_create_agent(agent_manager):
    """Test agent creation"""
    agent = agent_manager.create(
        name="test-agent",
        system_prompt="Test",
        model_name="gpt-4"
    )
    assert agent['name'] == "test-agent"
    assert 'id' in agent
```

### Integration Test Example

```python
@pytest.mark.integration
def test_end_to_end_workflow(client, test_agent):
    """Test complete workflow"""
    # Create session
    session_mgr = SessionManager(client)
    session = session_mgr.create(agent_id=test_agent['id'])
    
    # Send message
    response = session_mgr.send_message(
        session_id=session['id'],
        content="Hello"
    )
    assert 'response' in response
```

## Continuous Integration

Tests should be run:
- On every commit
- Before releases
- In CI/CD pipeline

## Notes

- Integration tests require running NeuronAgent server
- Some tests may require NeuronDB extension
- Mock tests don't require server
- Clean up test data after tests


