# NeuronMCP Service Files

This directory contains service files for running NeuronMCP as a system service on Linux (systemd) and macOS (launchd).

## Directory Structure

```
services/
├── systemd/
│   └── neuronmcp.service    # Linux systemd service file
├── launchd/
│   └── com.neurondb.neuronmcp.plist  # macOS launchd service file
└── README.md                  # This file
```

## Linux (systemd) Installation

### Prerequisites

1. Build NeuronMCP binary:
   ```bash
   cd NeuronMCP
   make build
   ```

2. Install binary to system path:
   ```bash
   sudo cp bin/neurondb-mcp /usr/local/bin/neurondb-mcp
   sudo chmod +x /usr/local/bin/neurondb-mcp
   ```

3. Create service user (if not exists):
   ```bash
   sudo useradd -r -s /bin/false neurondb
   sudo mkdir -p /opt/neurondb
   sudo chown neurondb:neurondb /opt/neurondb
   ```

### Installation Steps

1. Copy service file:
   ```bash
   sudo cp services/systemd/neuronmcp.service /etc/systemd/system/
   ```

2. Create configuration directory and environment file:
   ```bash
   sudo mkdir -p /etc/neurondb
   sudo nano /etc/neurondb/neuronmcp.env
   ```

   Add your configuration:
   ```bash
   NEURONDB_HOST=localhost
   NEURONDB_PORT=5432
   NEURONDB_DATABASE=neurondb
   NEURONDB_USER=neurondb
   NEURONDB_PASSWORD=your_password
   NEURONDB_LOG_LEVEL=info
   NEURONDB_LOG_FORMAT=text
   NEURONDB_LOG_OUTPUT=stderr
   NEURONDB_ENABLE_GPU=false
   ```

3. Set proper permissions:
   ```bash
   sudo chmod 600 /etc/neurondb/neuronmcp.env
   sudo chown root:root /etc/neurondb/neuronmcp.env
   ```

4. Reload systemd and enable service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable neuronmcp
   sudo systemctl start neuronmcp
   ```

### Management

- **Check status**: `sudo systemctl status neuronmcp`
- **View logs**: `sudo journalctl -u neuronmcp -f`
- **Restart**: `sudo systemctl restart neuronmcp`
- **Stop**: `sudo systemctl stop neuronmcp`
- **Disable**: `sudo systemctl disable neuronmcp`

## macOS (launchd) Installation

### Prerequisites

1. Build NeuronMCP binary:
   ```bash
   cd NeuronMCP
   make build
   ```

2. Install binary to system path:
   ```bash
   sudo cp bin/neurondb-mcp /usr/local/bin/neurondb-mcp
   sudo chmod +x /usr/local/bin/neurondb-mcp
   ```

3. Create directories:
   ```bash
   sudo mkdir -p /usr/local/var/log/neurondb
   sudo mkdir -p /usr/local/var/neurondb
   sudo chown $USER:admin /usr/local/var/log/neurondb
   sudo chown $USER:admin /usr/local/var/neurondb
   ```

### User-level Installation (Recommended for development)

1. Create log directory:
   ```bash
   mkdir -p ~/Library/Logs/neurondb
   ```

2. Copy plist file:
   ```bash
   cp services/launchd/com.neurondb.neuronmcp.plist ~/Library/LaunchAgents/
   ```

3. Edit plist file to update:
   - Binary path (if not in `/usr/local/bin/`)
   - Environment variables (database credentials, ports, etc.)
   - Working directory
   - Log paths

4. Load and start service:
   ```bash
   launchctl load ~/Library/LaunchAgents/com.neurondb.neuronmcp.plist
   launchctl start com.neurondb.neuronmcp
   ```

### System-level Installation (Requires root)

1. Copy plist file:
   ```bash
   sudo cp services/launchd/com.neurondb.neuronmcp.plist /Library/LaunchDaemons/
   ```

2. Edit plist file (as root):
   ```bash
   sudo nano /Library/LaunchDaemons/com.neurondb.neuronmcp.plist
   ```

3. Set ownership:
   ```bash
   sudo chown root:wheel /Library/LaunchDaemons/com.neurondb.neuronmcp.plist
   ```

4. Load and start service:
   ```bash
   sudo launchctl load /Library/LaunchDaemons/com.neurondb.neuronmcp.plist
   sudo launchctl start com.neurondb.neuronmcp
   ```

### Management

- **Check status**: `launchctl list | grep neuronmcp`
- **View logs**: 
  - User-level: `tail -f ~/Library/Logs/neurondb/neuronmcp.log`
  - System-level: `tail -f /usr/local/var/log/neurondb/neuronmcp.log`
- **Stop**: `launchctl stop com.neurondb.neuronmcp`
- **Unload**: `launchctl unload ~/Library/LaunchAgents/com.neurondb.neuronmcp.plist`
- **Restart**: `launchctl stop com.neurondb.neuronmcp && launchctl start com.neurondb.neuronmcp`

## Configuration

### Environment Variables

NeuronMCP supports the following environment variables:

- `NEURONDB_HOST` - Database host (default: localhost)
- `NEURONDB_PORT` - Database port (default: 5432)
- `NEURONDB_DATABASE` - Database name (default: neurondb)
- `NEURONDB_USER` - Database user (default: pgedge)
- `NEURONDB_PASSWORD` - Database password (required)
- `NEURONDB_LOG_LEVEL` - Log level: debug, info, warn, error (default: info)
- `NEURONDB_LOG_FORMAT` - Log format: text, json (default: text)
- `NEURONDB_LOG_OUTPUT` - Log output: stdout, stderr (default: stderr)
- `NEURONDB_ENABLE_GPU` - Enable GPU features: true, false (default: false)

## Troubleshooting

### Service fails to start

1. Check logs:
   - Linux: `sudo journalctl -u neuronmcp -n 50`
   - macOS: `tail -f ~/Library/Logs/neurondb/neuronmcp.error.log`

2. Verify binary exists and is executable:
   ```bash
   ls -l /usr/local/bin/neurondb-mcp
   ```

3. Test running the binary manually:
   ```bash
   /usr/local/bin/neurondb-mcp
   ```

### Database connection errors

Ensure PostgreSQL is running and credentials are correct:
```bash
psql -h localhost -U neurondb -d neurondb -c "SELECT 1;"
```

### Permission errors

- Linux: Ensure the neurondb user has appropriate permissions
- macOS: Ensure log directories are writable

## Customization

### Changing Binary Location

Update the `ExecStart` path in the service file (Linux) or `ProgramArguments` in the plist file (macOS).

### Resource Limits

Adjust memory and CPU limits in the service files:
- Linux: Edit `MemoryLimit` and `CPUQuota` in the systemd service file
- macOS: Use `Nice` key in plist file to adjust priority
