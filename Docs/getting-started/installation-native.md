# Native Installation Guide

Complete guide for installing and running NeuronMCP, NeuronAgent, and NeuronDesktop without Docker.

## Overview

This guide covers installing the NeuronDB ecosystem components (NeuronMCP, NeuronAgent, NeuronDesktop) directly on your system without using Docker containers. This approach provides:

- Direct system integration
- System service support (systemd/launchd)
- Native performance
- Full control over configuration

## Prerequisites

### System Requirements

- **Operating System**: Linux (systemd) or macOS (launchd)
- **RAM**: Minimum 2GB, recommended 4GB+
- **Disk Space**: Minimum 500MB for binaries and dependencies

### Required Software

#### All Components

- **Go**: Version 1.23 or later
  - Download: https://golang.org/dl/
  - Verify: `go version`
- **PostgreSQL**: Version 16, 17, or 18
  - Must have `pg_config` in PATH
  - Verify: `pg_config --version`
- **Make**: Build utility
  - Verify: `make --version`

#### NeuronDesktop Only

- **Node.js**: Version 18 or later
  - Download: https://nodejs.org/
  - Verify: `node --version`
- **npm**: Usually included with Node.js
  - Verify: `npm --version`

### PostgreSQL Setup

Before installing components, ensure PostgreSQL is installed and running:

```bash
# Check PostgreSQL status (Linux)
sudo systemctl status postgresql

# Check PostgreSQL status (macOS)
brew services list | grep postgresql

# Test connection
psql -U postgres -c "SELECT version();"
```

### NeuronDB Extension

The components require the NeuronDB PostgreSQL extension to be installed. See [NeuronDB Installation Guide](../../NeuronDB/docs/getting-started/installation.md) for details.

## Quick Installation

### Automated Installation (Recommended)

Use the unified installation script to install all components:

```bash
# Install all components
sudo ./scripts/install-components.sh

# Install specific components
sudo ./scripts/install-components.sh neuronmcp neuronagent

# Install with services enabled
sudo ./scripts/install-components.sh --enable-service
```

### Component-Specific Installation

Install individual components using their specific scripts:

```bash
# Install NeuronMCP
sudo ./scripts/install-neuronmcp.sh

# Install NeuronAgent
sudo ./scripts/install-neuronagent.sh

# Install NeuronDesktop
sudo ./scripts/install-neurondesktop.sh
```

## Manual Installation

### Step 1: Build from Source

#### NeuronMCP

```bash
cd NeuronMCP
make build
# Binary will be in bin/neurondb-mcp
```

#### NeuronAgent

```bash
cd NeuronAgent
make build
# Binary will be in bin/neuronagent
```

#### NeuronDesktop

```bash
cd NeuronDesktop
make build-api
# Binary will be in bin/neurondesktop
```

### Step 2: Install Binaries

```bash
# Copy binaries to system path
sudo cp NeuronMCP/bin/neurondb-mcp /usr/local/bin/
sudo cp NeuronAgent/bin/neuronagent /usr/local/bin/
sudo cp NeuronDesktop/bin/neurondesktop /usr/local/bin/

# Make executable
sudo chmod +x /usr/local/bin/neurondb-mcp
sudo chmod +x /usr/local/bin/neuronagent
sudo chmod +x /usr/local/bin/neurondesktop
```

### Step 3: Database Setup

#### NeuronMCP

```bash
cd NeuronMCP
./scripts/neuronmcp-setup.sh
```

#### NeuronAgent

```bash
cd NeuronAgent
./scripts/neuronagent-setup.sh
```

#### NeuronDesktop

```bash
cd NeuronDesktop
./scripts/neurondesktop-setup.sh
```

### Step 4: Configuration

Create environment configuration files:

```bash
# Create configuration directory
sudo mkdir -p /etc/neurondb

# Copy example configurations
sudo cp scripts/config/neuronmcp.env.example /etc/neurondb/neuronmcp.env
sudo cp scripts/config/neuronagent.env.example /etc/neurondb/neuronagent.env
sudo cp scripts/config/neurondesktop.env.example /etc/neurondb/neurondesktop.env

# Edit configurations
sudo nano /etc/neurondb/neuronmcp.env
sudo nano /etc/neurondb/neuronagent.env
sudo nano /etc/neurondb/neurondesktop.env
```

Update the configuration files with your database credentials and settings. See the [Configuration Guide](#configuration) section below.

### Step 5: Service Installation (Optional)

See [Service Management Guide](installation-services.md) for detailed instructions on setting up system services.

## Configuration

### Environment Variables

Each component can be configured via environment variables. Configuration files are located in `/etc/neurondb/` (or custom location).

#### NeuronMCP

Key environment variables:
- `NEURONDB_HOST`: Database host (default: localhost)
- `NEURONDB_PORT`: Database port (default: 5432)
- `NEURONDB_DATABASE`: Database name (default: neurondb)
- `NEURONDB_USER`: Database user
- `NEURONDB_PASSWORD`: Database password
- `NEURONDB_LOG_LEVEL`: Log level (debug, info, warn, error)

See `scripts/config/neuronmcp.env.example` for complete list.

#### NeuronAgent

Key environment variables:
- `DB_HOST`: Database host (default: localhost)
- `DB_PORT`: Database port (default: 5432)
- `DB_NAME`: Database name (default: neurondb)
- `DB_USER`: Database user
- `DB_PASSWORD`: Database password
- `SERVER_PORT`: Server port (default: 8080)
- `LOG_LEVEL`: Log level (debug, info, warn, error)

See `scripts/config/neuronagent.env.example` for complete list.

#### NeuronDesktop

Key environment variables:
- `DB_HOST`: Database host (default: localhost)
- `DB_PORT`: Database port (default: 5432)
- `DB_NAME`: Database name (default: neurondesk)
- `DB_USER`: Database user
- `DB_PASSWORD`: Database password
- `SERVER_PORT`: API server port (default: 8081)
- `NEURONDB_HOST`: NeuronDB connection for MCP
- `NEURONMCP_BINARY_PATH`: Path to neurondb-mcp binary

See `scripts/config/neurondesktop.env.example` for complete list.

### Configuration Files

Some components also support configuration files:

- **NeuronMCP**: `mcp-config.json` (optional, see `NeuronMCP/mcp-config.json.example`)
- **NeuronAgent**: `config.yaml` (optional, see `NeuronAgent/configs/config.yaml.example`)

Environment variables override configuration file values.

## Running Components

### Manual Execution

Run components directly from the command line:

```bash
# NeuronMCP (stdio-based, requires MCP client)
neurondb-mcp

# NeuronAgent (HTTP server)
neuronagent

# NeuronDesktop API (HTTP server)
neurondesktop
```

### As System Services

See [Service Management Guide](installation-services.md) for running as system services.

## Verification

### Check Binary Installation

```bash
which neurondb-mcp
which neuronagent
which neurondesktop

# Check versions (if supported)
neurondb-mcp --version 2>/dev/null || echo "Version check not available"
```

### Test Database Connection

```bash
# Test PostgreSQL connection
psql -h localhost -U neurondb -d neurondb -c "SELECT 1;"

# Verify NeuronDB extension
psql -h localhost -U neurondb -d neurondb -c "SELECT neurondb.version();"
```

### Test Services

```bash
# Test NeuronAgent health endpoint
curl http://localhost:8080/health

# Test NeuronDesktop API health endpoint
curl http://localhost:8081/health
```

## Troubleshooting

### Build Failures

**Error: Go version too old**
- Install Go 1.23 or later
- Verify: `go version`

**Error: pg_config not found**
- Install PostgreSQL development headers
- Linux: `sudo apt-get install postgresql-server-dev-16` (or 17, 18)
- macOS: `brew install postgresql`

**Error: Node.js not found (NeuronDesktop)**
- Install Node.js 18 or later
- Verify: `node --version`

### Database Connection Errors

**Error: Connection refused**
- Verify PostgreSQL is running: `sudo systemctl status postgresql`
- Check database host and port in configuration
- Verify firewall settings

**Error: Authentication failed**
- Verify database credentials in configuration files
- Check PostgreSQL user permissions
- Verify password is correct

### Service Issues

**Service fails to start**
- Check logs: `sudo journalctl -u neuronagent -n 50` (Linux)
- Verify binary paths in service files
- Check configuration file permissions
- Verify database is accessible

**Service crashes immediately**
- Check error logs
- Verify all environment variables are set
- Check for port conflicts
- Verify database connection

### Permission Errors

**Error: Permission denied**
- Ensure binaries are executable: `chmod +x /usr/local/bin/neurondb-mcp`
- Check configuration file permissions
- Verify service user has appropriate permissions
- Check working directory permissions

## Uninstallation

### Remove Binaries

```bash
sudo rm /usr/local/bin/neurondb-mcp
sudo rm /usr/local/bin/neuronagent
sudo rm /usr/local/bin/neurondesktop
```

### Remove Service Files

**Linux (systemd):**
```bash
sudo systemctl stop neuronmcp neuronagent neurondesktop-api
sudo systemctl disable neuronmcp neuronagent neurondesktop-api
sudo rm /etc/systemd/system/neuron*.service
sudo systemctl daemon-reload
```

**macOS (launchd):**
```bash
launchctl unload ~/Library/LaunchAgents/com.neurondb.*.plist
rm ~/Library/LaunchAgents/com.neurondb.*.plist
```

### Remove Configuration

```bash
# Optional: Remove configuration files
sudo rm -rf /etc/neurondb

# Note: Database data is not removed automatically
# Remove manually if needed:
# psql -U postgres -c "DROP DATABASE neurondb;"
# psql -U postgres -c "DROP DATABASE neurondesk;"
```

## Next Steps

- [Service Management Guide](installation-services.md) - Running components as services
- [NeuronMCP README](../../NeuronMCP/README.md) - NeuronMCP documentation
- [NeuronAgent README](../../NeuronAgent/README.md) - NeuronAgent documentation
- [NeuronDesktop README](../../NeuronDesktop/README.md) - NeuronDesktop documentation
- [Configuration Examples](../reference/configuration.md) - Configuration examples and best practices

## Related Documentation

- [Docker Installation Guide](../deployment/docker.md) - Docker-based installation
- [NeuronDB Installation](../../NeuronDB/docs/getting-started/installation.md) - PostgreSQL extension installation
- [Production Deployment](../deployment/production.md) - Production deployment guide



