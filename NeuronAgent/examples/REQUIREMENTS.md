# Requirements Documentation

Complete guide to dependencies for the NeuronAgent Python client library.

## Overview

The NeuronAgent client library has different requirement sets depending on your use case:

- **Minimal**: Basic HTTP/WebSocket client only
- **Standard**: Full client library with examples
- **NeuronDB Integration**: Full client + NeuronDB database integration
- **Development**: All dependencies for development and testing

## Quick Start

### Basic Installation

```bash
# Install core dependencies only
pip install -r requirements-minimal.txt

# Or install standard requirements
pip install -r requirements.txt
```

### With NeuronDB Integration

```bash
# Install with NeuronDB support
pip install -r requirements.txt
pip install -r requirements-neurondb.txt

# Or install everything at once
pip install -r requirements.txt -r requirements-neurondb.txt
```

### For Development

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Requirements Files

### `requirements.txt` (Standard)

Core dependencies for the client library:

- `requests>=2.31.0` - HTTP client library
- `websocket-client>=1.6.0` - WebSocket client
- `urllib3>=2.0.0` - HTTP library with retry support

**Installation:**
```bash
pip install -r requirements.txt
```

**Use when:**
- Using the client library for API calls
- Running examples
- Basic agent operations

### `requirements-minimal.txt`

Absolute minimum dependencies:

- `requests>=2.31.0`
- `websocket-client>=1.6.0`
- `urllib3>=2.0.0`

**Installation:**
```bash
pip install -r requirements-minimal.txt
```

**Use when:**
- Minimal installation needed
- Limited dependencies required
- Basic HTTP/WebSocket only

### `requirements-neurondb.txt`

NeuronDB integration dependencies:

- `psycopg2-binary>=2.9.7` - PostgreSQL adapter
- `numpy>=1.24.0` - Numerical operations
- `pandas>=2.0.0` - Data processing
- `scipy>=1.11.0` - Scientific computing
- Additional optional packages

**Installation:**
```bash
pip install -r requirements.txt
pip install -r requirements-neurondb.txt
```

**Use when:**
- Working with NeuronDB database directly
- Using vector search features
- Processing embeddings
- Running ML models
- Data analysis operations

### `requirements-dev.txt`

Development and testing dependencies:

- Type checking: `mypy`, `types-requests`
- Code formatting: `black`, `isort`
- Linting: `flake8`, `pylint`, `ruff`
- Testing: `pytest`, `pytest-cov`, `pytest-asyncio`
- Documentation: `sphinx`, `sphinx-rtd-theme`
- Development tools: `ipython`, `ipdb`, `pre-commit`

**Installation:**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Use when:**
- Developing the client library
- Running tests
- Contributing code
- Building documentation

## Dependency Details

### Core Dependencies

#### requests (>=2.31.0)

**Purpose:** HTTP client library for API requests

**Used in:**
- `core/client.py` - Main HTTP client
- All API operations

**Features:**
- Retry logic
- Connection pooling
- Session management
- Authentication

**Why this version:**
- 2.31.0+ includes security fixes
- Better error handling
- Improved connection pooling

#### websocket-client (>=1.6.0)

**Purpose:** WebSocket client for streaming responses

**Used in:**
- `core/websocket.py` - WebSocket client
- Streaming message responses

**Features:**
- Real-time streaming
- Message handling
- Connection management

**Why this version:**
- 1.6.0+ includes bug fixes
- Better error handling
- Improved reconnection logic

#### urllib3 (>=2.0.0)

**Purpose:** HTTP library with retry support

**Used in:**
- `core/client.py` - Retry strategy
- Connection pooling

**Why this version:**
- 2.0.0+ includes security fixes
- Better retry mechanisms
- Improved connection handling

### NeuronDB Integration Dependencies

#### psycopg2-binary (>=2.9.7)

**Purpose:** PostgreSQL adapter for Python

**Used for:**
- Direct database connections
- Vector operations
- Embedding storage
- SQL queries

**Why this version:**
- 2.9.7+ includes performance improvements
- Better error messages
- Improved connection handling

**Installation notes:**
- On Linux: May need `postgresql-dev` package
- On macOS: May need PostgreSQL via Homebrew
- On Windows: Pre-built wheels available

#### numpy (>=1.24.0)

**Purpose:** Numerical operations and vector handling

**Used for:**
- Vector operations
- Array handling
- Embedding processing
- Numerical computations

**Why this version:**
- 1.24.0+ includes performance improvements
- Better type hints
- Improved array operations

#### pandas (>=2.0.0)

**Purpose:** Data processing and analysis

**Used for:**
- Data manipulation
- DataFrame operations
- Data analysis

**Why this version:**
- 2.0.0+ includes major improvements
- Better performance
- Improved API

## Python Version Compatibility

All packages are tested with:

- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11
- Python 3.12

**Minimum:** Python 3.8

**Recommended:** Python 3.10+

## Platform Compatibility

### Linux

All packages work on Linux. Some notes:

- `psycopg2-binary` may need `postgresql-dev` package
- GPU packages available for CUDA/ROCm

### macOS

All packages work on macOS. Some notes:

- `psycopg2-binary` may need PostgreSQL via Homebrew
- Apple Silicon (M1/M2) supported

### Windows

All packages work on Windows. Some notes:

- `psycopg2-binary` has pre-built wheels
- Some optional packages may have limited Windows support

## Installation Methods

### pip

```bash
# Standard installation
pip install -r requirements.txt

# With NeuronDB
pip install -r requirements.txt -r requirements-neurondb.txt

# Development
pip install -r requirements.txt -r requirements-dev.txt
```

### pip with virtual environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install
pip install -r requirements.txt
```

### conda

```bash
# Create conda environment
conda create -n neurondb python=3.10
conda activate neurondb

# Install with conda-forge
conda install -c conda-forge requests websocket-client numpy psycopg2

# Or use pip
pip install -r requirements.txt
```

## Troubleshooting

### psycopg2-binary Installation Issues

**Linux:**
```bash
sudo apt-get install postgresql-dev python3-dev
pip install psycopg2-binary
```

**macOS:**
```bash
brew install postgresql
pip install psycopg2-binary
```

**Windows:**
```bash
# Should work automatically with pre-built wheels
pip install psycopg2-binary
```

### numpy Installation Issues

**If pip fails:**
```bash
# Use conda instead
conda install numpy

# Or install build dependencies first
pip install wheel setuptools
pip install numpy
```

### websocket-client SSL Issues

**If SSL errors occur:**
```bash
# Ensure SSL is enabled
python -c "import ssl; print(ssl.OPENSSL_VERSION)"

# Update certificates (Linux)
sudo update-ca-certificates
```

### Version Conflicts

**If you have version conflicts:**
```bash
# Create fresh virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Optional Dependencies

Some features use optional dependencies:

### Environment Variables

```bash
pip install python-dotenv
```

### Progress Bars

```bash
pip install tqdm
```

### JSON Validation

```bash
pip install jsonschema
```

### Hugging Face Models

```bash
pip install transformers torch
```

### ONNX Runtime

```bash
pip install onnxruntime
```

## Version Pinning Strategy

We use version ranges (e.g., `>=2.31.0,<3.0.0`) to:

1. **Allow security updates** within major versions
2. **Prevent breaking changes** from major version upgrades
3. **Ensure compatibility** with tested versions

## Security Considerations

- All packages are pinned to versions with known security fixes
- Regular updates recommended for security patches
- Use `pip list --outdated` to check for updates

## Updating Dependencies

```bash
# Check for outdated packages
pip list --outdated

# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade requests
```

## Support

For dependency issues:

1. Check this documentation
2. Review package-specific documentation
3. Check GitHub issues
4. Contact support@neurondb.ai

## License Compatibility

All dependencies are compatible with the project license (AGPL v3).

Most dependencies use permissive licenses (MIT, Apache 2.0, BSD).

