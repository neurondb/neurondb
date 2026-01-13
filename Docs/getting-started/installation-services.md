# Service Management Guide

Guide for running NeuronMCP, NeuronAgent, and NeuronDesktop as system services on Linux (systemd) and macOS (launchd).

## Overview

Running components as system services provides:
- Automatic startup on boot
- Automatic restart on failure
- Centralized logging
- Process management
- Resource limits

## Prerequisites

- Components installed (see [Native Installation Guide](installation-native.md))
- Systemd (Linux) or launchd (macOS)
- Appropriate permissions (root for system-level services)

## Quick Start

### Using Installation Scripts

The installation scripts can automatically install service configurations:

```bash
# Install components with services enabled
sudo ./scripts/install-components.sh --enable-service

# Or install individual components
sudo ./scripts/install-neuronmcp.sh --enable-service
sudo ./scripts/install-neuronagent.sh --enable-service
sudo ./scripts/install-neurondesktop.sh --enable-service
```

### Manual Service Installation

If you've already installed the binaries manually, see the platform-specific sections below.

## Linux (systemd)

### Installation

1. Copy service files to systemd directory:
   ```bash
   sudo cp scripts/services/systemd/*.service /etc/systemd/system/
   ```

2. Create configuration directory:
   ```bash
   sudo mkdir -p /etc/neurondb
   ```

3. Create configuration files:
   ```bash
   sudo cp scripts/config/neuronmcp.env.example /etc/neurondb/neuronmcp.env
   sudo cp scripts/config/neuronagent.env.example /etc/neurondb/neuronagent.env
   sudo cp scripts/config/neurondesktop.env.example /etc/neurondb/neurondesktop.env
   ```

4. Edit configuration files with your settings:
   ```bash
   sudo nano /etc/neurondb/neuronmcp.env
   sudo nano /etc/neurondb/neuronagent.env
   sudo nano /etc/neurondb/neurondesktop.env
   ```

5. Update service files if binaries are in non-standard locations:
   ```bash
   sudo nano /etc/systemd/system/neuronmcp.service
   # Update ExecStart path if needed
   ```

6. Create service user (if not exists):
   ```bash
   sudo useradd -r -s /bin/false neurondb
   sudo mkdir -p /opt/neurondb
   sudo chown neurondb:neurondb /opt/neurondb
   ```

7. Reload systemd and enable services:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable neuronmcp neuronagent neurondesktop-api
   sudo systemctl start neuronmcp neuronagent neurondesktop-api
   ```

### Management

Use the service management script or systemctl directly:

```bash
# Using management script
./scripts/manage-services.sh start
./scripts/manage-services.sh status
./scripts/manage-services.sh logs neuronagent
./scripts/manage-services.sh restart neuronmcp

# Using systemctl directly
sudo systemctl status neuronmcp
sudo systemctl restart neuronagent
sudo systemctl stop neurondesktop-api
sudo systemctl start neurondesktop-api
```

### Common Commands

```bash
# Check status
sudo systemctl status neuronmcp
sudo systemctl status neuronagent
sudo systemctl status neurondesktop-api

# View logs
sudo journalctl -u neuronmcp -f
sudo journalctl -u neuronagent -n 50
sudo journalctl -u neurondesktop-api --since "1 hour ago"

# Restart services
sudo systemctl restart neuronmcp
sudo systemctl restart neuronagent
sudo systemctl restart neurondesktop-api

# Enable/disable services
sudo systemctl enable neuronmcp
sudo systemctl disable neuronagent
```

For detailed instructions, see [`scripts/services/systemd/README.md`](../../scripts/services/systemd/README.md).

## macOS (launchd)

### User-Level Services (Recommended for Development)

1. Create directories:
   ```bash
   mkdir -p ~/Library/Logs/neurondb
   mkdir -p ~/usr/local/var/neurondb
   ```

2. Copy plist files:
   ```bash
   cp scripts/services/launchd/*.plist ~/Library/LaunchAgents/
   ```

3. Edit plist files with your configuration:
   ```bash
   nano ~/Library/LaunchAgents/com.neurondb.neuronmcp.plist
   # Update ProgramArguments, EnvironmentVariables, paths
   ```

4. Load and start services:
   ```bash
   launchctl load ~/Library/LaunchAgents/com.neurondb.neuronmcp.plist
   launchctl load ~/Library/LaunchAgents/com.neurondb.neuronagent.plist
   launchctl load ~/Library/LaunchAgents/com.neurondb.neurondesktop-api.plist
   
   launchctl start com.neurondb.neuronmcp
   launchctl start com.neurondb.neuronagent
   launchctl start com.neurondb.neurondesktop-api
   ```

### System-Level Services (Requires Root)

1. Create directories:
   ```bash
   sudo mkdir -p /usr/local/var/log/neurondb
   sudo mkdir -p /usr/local/var/neurondb
   sudo chown $USER:admin /usr/local/var/log/neurondb
   sudo chown $USER:admin /usr/local/var/neurondb
   ```

2. Copy plist files to LaunchDaemons:
   ```bash
   sudo cp scripts/services/launchd/*.plist /Library/LaunchDaemons/
   sudo chown root:wheel /Library/LaunchDaemons/com.neurondb.*.plist
   ```

3. Edit plist files (as root):
   ```bash
   sudo nano /Library/LaunchDaemons/com.neurondb.neuronmcp.plist
   ```

4. Load and start services:
   ```bash
   sudo launchctl load /Library/LaunchDaemons/com.neurondb.neuronmcp.plist
   sudo launchctl load /Library/LaunchDaemons/com.neurondb.neuronagent.plist
   sudo launchctl load /Library/LaunchDaemons/com.neurondb.neurondesktop-api.plist
   
   sudo launchctl start com.neurondb.neuronmcp
   sudo launchctl start com.neurondb.neuronagent
   sudo launchctl start com.neurondb.neurondesktop-api
   ```

### Management

```bash
# Check status
launchctl list | grep neurondb

# View logs
tail -f ~/Library/Logs/neurondb/neuronmcp.log
tail -f ~/Library/Logs/neurondb/neuronagent.log

# Stop services
launchctl stop com.neurondb.neuronmcp
launchctl stop com.neurondb.neuronagent

# Unload services
launchctl unload ~/Library/LaunchAgents/com.neurondb.neuronmcp.plist
```

For detailed instructions, see [`scripts/services/launchd/README.md`](../../scripts/services/launchd/README.md).

## Service Management Script

A unified script is available for managing services across platforms:

```bash
# Start all services
./scripts/manage-services.sh start

# Start specific service
./scripts/manage-services.sh start neuronmcp

# Check status
./scripts/manage-services.sh status

# View logs
./scripts/manage-services.sh logs neuronagent

# Restart service
./scripts/manage-services.sh restart neuronagent

# Enable services (start on boot)
./scripts/manage-services.sh enable

# Health check
./scripts/manage-services.sh health
```

## Configuration

Service configurations use environment files:

- **Linux (systemd)**: `/etc/neurondb/*.env` (referenced in service files)
- **macOS (launchd)**: Environment variables in plist files or wrapper scripts

Update configuration files and restart services to apply changes:

```bash
# Linux
sudo nano /etc/neurondb/neuronmcp.env
sudo systemctl restart neuronmcp

# macOS (edit plist)
nano ~/Library/LaunchAgents/com.neurondb.neuronmcp.plist
launchctl unload ~/Library/LaunchAgents/com.neurondb.neuronmcp.plist
launchctl load ~/Library/LaunchAgents/com.neurondb.neuronmcp.plist
```

## Troubleshooting

### Service Won't Start

1. Check logs:
   ```bash
   # Linux
   sudo journalctl -u neuronmcp -n 50
   
   # macOS
   tail -f ~/Library/Logs/neurondb/neuronmcp.error.log
   ```

2. Verify binary exists and is executable:
   ```bash
   which neurondb-mcp
   ls -l /usr/local/bin/neurondb-mcp
   ```

3. Test running binary manually:
   ```bash
   /usr/local/bin/neurondb-mcp
   ```

4. Check configuration file syntax and paths

### Service Crashes

1. Check error logs for crash reasons
2. Verify database connectivity
3. Check for port conflicts
4. Verify all environment variables are set correctly
5. Check resource limits (memory, file descriptors)

### Permission Errors

1. Verify service user has appropriate permissions
2. Check file permissions on binaries and config files
3. Ensure working directory is writable
4. Check log directory permissions

### Database Connection Issues

1. Verify PostgreSQL is running
2. Check database credentials in configuration
3. Test connection manually:
   ```bash
   psql -h localhost -U neurondb -d neurondb -c "SELECT 1;"
   ```
4. Check firewall settings

## Security Considerations

### Service User

- Run services as non-root user (e.g., `neurondb`)
- Use minimal permissions
- Set appropriate umask

### Configuration Files

- Restrict permissions on config files:
  ```bash
  sudo chmod 600 /etc/neurondb/*.env
  sudo chown root:neurondb /etc/neurondb/*.env
  ```

### Network Security

- Use firewall rules to restrict access
- Use SSL/TLS for database connections in production
- Bind services to localhost when possible

## Next Steps

- [Native Installation Guide](installation-native.md) - Installing components
- [Configuration Guide](../reference/configuration.md) - Configuration options
- Component-specific documentation:
  - [NeuronMCP README](../../NeuronMCP/README.md)
  - [NeuronAgent README](../../NeuronAgent/README.md)
  - [NeuronDesktop README](../../NeuronDesktop/README.md)



