# NeuronMCP Test Suite

Comprehensive test suite for NeuronMCP covering all 100+ tools, 9 resources, MCP protocol endpoints, and integration scenarios.

## Test Structure

```
tests/
├── test_protocol.py          # MCP protocol endpoint tests
├── test_tools_postgresql.py   # PostgreSQL tools tests (27 tools)
├── test_resources.py          # Resources tests (9 resources)
├── test_comprehensive.py      # Comprehensive tool tests (100+ tools)
├── test_dataloading.py        # Dataset loading comprehensive tests
└── run_all_tests.py          # Test runner script
```

## Test Categories

### 1. Protocol Tests (`test_protocol.py`)
- MCP protocol initialization
- tools/list endpoint
- tools/call endpoint
- tools/search endpoint
- tools/call_batch endpoint
- resources/list endpoint
- resources/read endpoint
- prompts/list endpoint
- progress/get endpoint

### 2. PostgreSQL Tools Tests (`test_tools_postgresql.py`)
- Server Information Tools (5 tools)
- Database Object Management Tools (8 tools)
- User and Role Management Tools (3 tools)
- Performance and Statistics Tools (7 tools)
- Size and Storage Tools (4 tools)

### 3. Resources Tests (`test_resources.py`)
- All 9 resources
- Resource listing
- Resource reading
- Invalid resource handling

### 4. Comprehensive Tool Tests (`test_comprehensive.py`)
- All 100+ tools across all categories
- Vector operations
- Embedding tools
- ML operations
- Analytics tools
- RAG operations
- And more...

### 5. Dataset Loading Tests (`test_dataloading.py`)
- HuggingFace dataset loading (basic, with embeddings, streaming, configs)
- URL dataset loading (CSV, JSON, auto-format detection, compression)
- Local file loading (CSV, JSON, JSONL, auto-format)
- GitHub repository loading
- CSV-specific options (delimiters, headers, skip rows)
- Table management (if_exists, load_mode)
- Embedding features (auto-embedding, custom dimensions, index creation)
- Parameter validation and error handling
- 30+ test cases covering all dataloading features

## Running Tests

### Prerequisites

1. **Database Setup**: Ensure PostgreSQL with NeuronDB extension is running
2. **Configuration**: Create `neuronmcp_server.json` in the NeuronMCP root directory
3. **Python Dependencies**: Install required Python packages:
   ```bash
   pip install -r client/requirements.txt
   ```

### Running All Tests

```bash
# Run all Python tests
make test-python

# Or run directly
python3 tests/run_all_tests.py
```

### Running Individual Test Suites

```bash
# Protocol tests
python3 tests/test_protocol.py

# PostgreSQL tools tests
python3 tests/test_tools_postgresql.py

# Resources tests
python3 tests/test_resources.py

# Comprehensive tests
python3 tests/test_comprehensive.py

# Dataset loading tests
python3 tests/test_dataloading.py
```

### Running Go Tests

```bash
# All Go tests
make test

# Unit tests only
make test-unit

# Integration tests only
make test-integration

# Fast tests (no race detector)
make test-fast
```

## Test Configuration

Create `neuronmcp_server.json` in the NeuronMCP root directory:

```json
{
  "mcpServers": {
    "neurondb": {
      "command": "./bin/neurondb-mcp",
      "env": {
        "NEURONDB_HOST": "localhost",
        "NEURONDB_PORT": "5432",
        "NEURONDB_DATABASE": "neurondb",
        "NEURONDB_USER": "neurondb",
        "NEURONDB_PASSWORD": "neurondb"
      }
    }
  }
}
```

## Test Results

Tests report results in the following format:

- ✅ **Passed**: Test completed successfully
- ❌ **Failed**: Test failed with an error
- ⚠️ **Configuration Needed**: Test requires database connection or configuration
- ⏭️ **Skipped**: Test was skipped (not available or not applicable)

## Test Coverage

The test suite aims for:

- **100% Tool Coverage**: All 100+ tools tested
- **100% Resource Coverage**: All 9 resources tested
- **100% Protocol Coverage**: All MCP protocol endpoints tested
- **>80% Code Coverage**: Unit tests cover >80% of codebase

## Continuous Integration

Tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    make test-all
```

## Troubleshooting

### Database Connection Issues

If tests fail with connection errors:

1. Verify PostgreSQL is running
2. Check database credentials in `neuronmcp_server.json`
3. Ensure NeuronDB extension is installed
4. Verify database user has required permissions

### Test Timeouts

If tests timeout:

1. Increase timeout in `run_all_tests.py`
2. Run tests individually to identify slow tests
3. Check database performance

### Missing Tools

If tools are not found:

1. Verify NeuronMCP server is built: `make build`
2. Check tool registration in server logs
3. Verify NeuronDB extension is properly installed

## Contributing

When adding new tools or features:

1. Add corresponding tests to appropriate test file
2. Update test counts in documentation
3. Ensure tests pass before submitting PR
4. Update this README if test structure changes



