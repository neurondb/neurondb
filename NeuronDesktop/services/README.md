# NeuronDesktop Service Files

This directory contains service files for running NeuronDesktop API server as a system service on Linux (systemd) and macOS (launchd).

## Directory Structure

```
services/
├── systemd/
│   └── neurondesktop.service    # Linux systemd service file
├── launchd/
│   └── com.neurondb.neurondesktop.plist  # macOS launchd service file
└── README.md                  # This file
```

## Linux (systemd) Installation

### Prerequisites

1. Build NeuronDesktop binary:
   ```bash
   cd NeuronDesktop
   make build
   ```

2. Install binary to system path:
   ```bash
   sudo cp bin/neurondesktop /usr/local/bin/neurondesktop
   sudo chmod +x /usr/local/bin/neurondesktop
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
   sudo cp services/systemd/neurondesktop.service /etc/systemd/system/
   ```

2. Create configuration directory and environment file:
   ```bash
   sudo mkdir -p /etc/neurondb
   sudo nano /etc/neurondb/neurondesktop.env
   ```

   Add your configuration:
   ```bash
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=neurondesk
   DB_USER=neurondb
   DB_PASSWORD=your_password
   SERVER_HOST=0.0.0.0
   SERVER_PORT=8081
   LOG_LEVEL=info
   LOG_FORMAT=json
   NEURONDB_HOST=localhost
   NEURONDB_PORT=5432
   NEURONDB_DATABASE=neurondb
   NEURONDB_USER=neurondb
   NEURONDB_PASSWORD=your_password
   NEURONMCP_BINARY_PATH=/usr/local/bin/neurondb-mcp
   ```

3. Set proper permissions:
   ```bash
   sudo chmod 600 /etc/neurondb/neurondesktop.env
   sudo chown root:root /etc/neurondb/neurondesktop.env
   ```

4. Reload systemd and enable service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable neurondesktop
   sudo systemctl start neurondesktop
   ```

### Management

- **Check status**: `sudo systemctl status neurondesktop`
- **View logs**: `sudo journalctl -u neurondesktop -f`
- **Restart**: `sudo systemctl restart neurondesktop`
- **Stop**: `sudo systemctl stop neurondesktop`
- **Disable**: `sudo systemctl disable neurondesktop`

## macOS (launchd) Installation

### Prerequisites

1. Build NeuronDesktop binary:
   ```bash
   cd NeuronDesktop
   make build
   ```

2. Install binary to system path:
   ```bash
   sudo cp bin/neurondesktop /usr/local/bin/neurondesktop
   sudo chmod +x /usr/local/bin/neurondesktop
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
   cp services/launchd/com.neurondb.neurondesktop.plist ~/Library/LaunchAgents/
   ```

3. Edit plist file to update:
   - Binary path (if not in `/usr/local/bin/`)
   - Environment variables (database credentials, ports, etc.)
   - Working directory
   - Log paths
   - NeuronMCP binary path

4. Load and start service:
   ```bash
   launchctl load ~/Library/LaunchAgents/com.neurondb.neurondesktop.plist
   launchctl start com.neurondb.neurondesktop
   ```

### System-level Installation (Requires root)

1. Copy plist file:
   ```bash
   sudo cp services/launchd/com.neurondb.neurondesktop.plist /Library/LaunchDaemons/
   ```

2. Edit plist file (as root):
   ```bash
   sudo nano /Library/LaunchDaemons/com.neurondb.neurondesktop.plist
   ```

3. Set ownership:
   ```bash
   sudo chown root:wheel /Library/LaunchDaemons/com.neurondb.neurondesktop.plist
   ```

4. Load and start service:
   ```bash
   sudo launchctl load /Library/LaunchDaemons/com.neurondb.neurondesktop.plist
   sudo launchctl start com.neurondb.neurondesktop
   ```

### Management

- **Check status**: `launchctl list | grep neurondesktop`
- **View logs**: 
  - User-level: `tail -f ~/Library/Logs/neurondb/neurondesktop.log`
  - System-level: `tail -f /usr/local/var/log/neurondb/neurondesktop.log`
- **Stop**: `launchctl stop com.neurondb.neurondesktop`
- **Unload**: `launchctl unload ~/Library/LaunchAgents/com.neurondb.neurondesktop.plist`
- **Restart**: `launchctl stop com.neurondb.neurondesktop && launchctl start com.neurondb.neurondesktop`

## Configuration

### Environment Variables

NeuronDesktop supports the following environment variables:

- `DB_HOST` - Desktop database host (default: localhost)
- `DB_PORT` - Desktop database port (default: 5432)
- `DB_NAME` - Desktop database name (default: neurondesk)
- `DB_USER` - Desktop database user (default: neurondb)
- `DB_PASSWORD` - Desktop database password (required)
- `SERVER_HOST` - Server bind address (default: 0.0.0.0)
- `SERVER_PORT` - Server port (default: 8081)
- `LOG_LEVEL` - Log level: debug, info, warn, error (default: info)
- `LOG_FORMAT` - Log format: json, console (default: json)
- `NEURONDB_HOST` - NeuronDB database host (default: localhost)
- `NEURONDB_PORT` - NeuronDB database port (default: 5432)
- `NEURONDB_DATABASE` - NeuronDB database name (default: neurondb)
- `NEURONDB_USER` - NeuronDB database user (default: neurondb)
- `NEURONDB_PASSWORD` - NeuronDB database password (required)
- `NEURONMCP_BINARY_PATH` - Path to NeuronMCP binary (optional, auto-detected if not set)

## Troubleshooting

### Service fails to start

1. Check logs:
   - Linux: `sudo journalctl -u neurondesktop -n 50`
   - macOS: `tail -f ~/Library/Logs/neurondb/neurondesktop.error.log`

2. Verify binary exists and is executable:
   ```bash
   ls -l /usr/local/bin/neurondesktop
   ```

3. Test running the binary manually:
   ```bash
   /usr/local/bin/neurondesktop
   ```

### Database connection errors

Ensure PostgreSQL is running and credentials are correct:
```bash
psql -h localhost -U neurondb -d neurondesk -c "SELECT 1;"
psql -h localhost -U neurondb -d neurondb -c "SELECT 1;"
```

### NeuronMCP binary not found

NeuronDesktop can auto-detect NeuronMCP binary, but you can explicitly set it:
```bash
export NEURONMCP_BINARY_PATH=/usr/local/bin/neurondb-mcp
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

## Frontend

Note: This service file only runs the NeuronDesktop API backend. For the frontend, you may need to run it separately or use a separate service file.
