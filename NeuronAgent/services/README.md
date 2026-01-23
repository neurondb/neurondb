# NeuronAgent Service Files

This directory contains service files for running NeuronAgent as a system service on Linux (systemd) and macOS (launchd).

## Directory Structure

```
services/
├── systemd/
│   └── neuronagent.service    # Linux systemd service file
├── launchd/
│   └── com.neurondb.neuronagent.plist  # macOS launchd service file
└── README.md                  # This file
```

## Linux (systemd) Installation

### Prerequisites

1. Build NeuronAgent binary:
   ```bash
   cd NeuronAgent
   make build
   ```

2. Install binary to system path:
   ```bash
   sudo cp bin/neuronagent /usr/local/bin/neuronagent
   sudo chmod +x /usr/local/bin/neuronagent
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
   sudo cp services/systemd/neuronagent.service /etc/systemd/system/
   ```

2. Create configuration directory and environment file:
   ```bash
   sudo mkdir -p /etc/neurondb
   sudo nano /etc/neurondb/neuronagent.env
   ```

   Add your configuration:
   ```bash
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=neurondb
   DB_USER=neurondb
   DB_PASSWORD=your_password
   SERVER_HOST=0.0.0.0
   SERVER_PORT=8080
   LOG_LEVEL=info
   LOG_FORMAT=json
   ```

3. Set proper permissions:
   ```bash
   sudo chmod 600 /etc/neurondb/neuronagent.env
   sudo chown root:root /etc/neurondb/neuronagent.env
   ```

4. Reload systemd and enable service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable neuronagent
   sudo systemctl start neuronagent
   ```

### Management

- **Check status**: `sudo systemctl status neuronagent`
- **View logs**: `sudo journalctl -u neuronagent -f`
- **Restart**: `sudo systemctl restart neuronagent`
- **Stop**: `sudo systemctl stop neuronagent`
- **Disable**: `sudo systemctl disable neuronagent`

## macOS (launchd) Installation

### Prerequisites

1. Build NeuronAgent binary:
   ```bash
   cd NeuronAgent
   make build
   ```

2. Install binary to system path:
   ```bash
   sudo cp bin/neuronagent /usr/local/bin/neuronagent
   sudo chmod +x /usr/local/bin/neuronagent
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
   cp services/launchd/com.neurondb.neuronagent.plist ~/Library/LaunchAgents/
   ```

3. Edit plist file to update:
   - Binary path (if not in `/usr/local/bin/`)
   - Environment variables (database credentials, ports, etc.)
   - Working directory
   - Log paths

4. Load and start service:
   ```bash
   launchctl load ~/Library/LaunchAgents/com.neurondb.neuronagent.plist
   launchctl start com.neurondb.neuronagent
   ```

### System-level Installation (Requires root)

1. Copy plist file:
   ```bash
   sudo cp services/launchd/com.neurondb.neuronagent.plist /Library/LaunchDaemons/
   ```

2. Edit plist file (as root):
   ```bash
   sudo nano /Library/LaunchDaemons/com.neurondb.neuronagent.plist
   ```

3. Set ownership:
   ```bash
   sudo chown root:wheel /Library/LaunchDaemons/com.neurondb.neuronagent.plist
   ```

4. Load and start service:
   ```bash
   sudo launchctl load /Library/LaunchDaemons/com.neurondb.neuronagent.plist
   sudo launchctl start com.neurondb.neuronagent
   ```

### Management

- **Check status**: `launchctl list | grep neuronagent`
- **View logs**: 
  - User-level: `tail -f ~/Library/Logs/neurondb/neuronagent.log`
  - System-level: `tail -f /usr/local/var/log/neurondb/neuronagent.log`
- **Stop**: `launchctl stop com.neurondb.neuronagent`
- **Unload**: `launchctl unload ~/Library/LaunchAgents/com.neurondb.neuronagent.plist`
- **Restart**: `launchctl stop com.neurondb.neuronagent && launchctl start com.neurondb.neuronagent`

## Configuration

### Environment Variables

NeuronAgent supports the following environment variables:

- `DB_HOST` - Database host (default: localhost)
- `DB_PORT` - Database port (default: 5432)
- `DB_NAME` - Database name (default: neurondb)
- `DB_USER` - Database user (default: postgres)
- `DB_PASSWORD` - Database password (required)
- `SERVER_HOST` - Server bind address (default: 0.0.0.0)
- `SERVER_PORT` - Server port (default: 8080)
- `LOG_LEVEL` - Log level: debug, info, warn, error (default: info)
- `LOG_FORMAT` - Log format: json, console (default: json)
- `CONFIG_PATH` - Path to YAML configuration file (optional)

### Configuration File

You can also use a YAML configuration file. See `configs/config.yaml.example` for an example.

## Troubleshooting

### Service fails to start

1. Check logs:
   - Linux: `sudo journalctl -u neuronagent -n 50`
   - macOS: `tail -f ~/Library/Logs/neurondb/neuronagent.error.log`

2. Verify binary exists and is executable:
   ```bash
   ls -l /usr/local/bin/neuronagent
   ```

3. Test running the binary manually:
   ```bash
   /usr/local/bin/neuronagent
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
