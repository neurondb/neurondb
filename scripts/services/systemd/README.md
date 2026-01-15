# systemd Service Files

This directory contains systemd service unit files for running NeuronDB ecosystem components as system services on Linux.

## Installation

1. Copy the service files to the systemd directory:
   ```bash
   sudo cp scripts/services/systemd/*.service /etc/systemd/system/
   ```

2. Create the configuration directory:
   ```bash
   sudo mkdir -p /etc/neurondb
   ```

3. Copy and configure environment files:
   ```bash
   sudo cp scripts/config/neuronmcp.env.example /etc/neurondb/neuronmcp.env
   sudo cp scripts/config/neuronagent.env.example /etc/neurondb/neuronagent.env
   sudo cp scripts/config/neurondesktop.env.example /etc/neurondb/neurondesktop.env
   
   # Edit the files with your configuration
   sudo nano /etc/neurondb/neuronmcp.env
   sudo nano /etc/neurondb/neuronagent.env
   sudo nano /etc/neurondb/neurondesktop.env
   ```

4. Create the user and group (if not exists):
   ```bash
   sudo useradd -r -s /bin/false neurondb
   sudo mkdir -p /opt/neurondb
   sudo chown neurondb:neurondb /opt/neurondb
   ```

5. Ensure binaries are installed:
   ```bash
   # Binaries should be in /usr/local/bin/ or update ExecStart paths
   ls -l /usr/local/bin/neurondb-mcp
   ls -l /usr/local/bin/neuronagent
   ls -l /usr/local/bin/neurondesktop
   ```

6. Reload systemd daemon:
   ```bash
   sudo systemctl daemon-reload
   ```

7. Enable and start services:
   ```bash
   sudo systemctl enable neuronmcp
   sudo systemctl enable neuronagent
   sudo systemctl enable neurondesktop-api
   
   sudo systemctl start neuronmcp
   sudo systemctl start neuronagent
   sudo systemctl start neurondesktop-api
   ```

## Management

### Check status
```bash
sudo systemctl status neuronmcp
sudo systemctl status neuronagent
sudo systemctl status neurondesktop-api
```

### View logs
```bash
sudo journalctl -u neuronmcp -f
sudo journalctl -u neuronagent -f
sudo journalctl -u neurondesktop-api -f
```

### Restart services
```bash
sudo systemctl restart neuronmcp
sudo systemctl restart neuronagent
sudo systemctl restart neurondesktop-api
```

### Stop services
```bash
sudo systemctl stop neuronmcp
sudo systemctl stop neuronagent
sudo systemctl stop neurondesktop-api
```

### Disable services
```bash
sudo systemctl disable neuronmcp
sudo systemctl disable neuronagent
sudo systemctl disable neurondesktop-api
```

## Customization

### Changing User/Group

If you want to run services as a different user, edit the service files:
```bash
sudo nano /etc/systemd/system/neuronmcp.service
```

Change:
```ini
User=your_user
Group=your_group
```

Then reload:
```bash
sudo systemctl daemon-reload
sudo systemctl restart neuronmcp
```

### Changing Binary Location

If binaries are installed in a different location, update the `ExecStart` path in the service files.

### Resource Limits

Adjust memory and CPU limits in the service files:
```ini
MemoryLimit=1G
CPUQuota=200%
```

Reload systemd after changes:
```bash
sudo systemctl daemon-reload
```

## Troubleshooting

### Service fails to start

1. Check logs:
   ```bash
   sudo journalctl -u neuronmcp -n 50
   ```

2. Verify environment file exists and is readable:
   ```bash
   sudo ls -l /etc/neurondb/neuronmcp.env
   ```

3. Verify binary exists and is executable:
   ```bash
   ls -l /usr/local/bin/neurondb-mcp
   ```

4. Test running the binary manually:
   ```bash
   sudo -u neurondb /usr/local/bin/neurondb-mcp
   ```

### Database connection errors

Ensure PostgreSQL is running and the database credentials in the environment file are correct:
```bash
sudo systemctl status postgresql
psql -h localhost -U neurondb -d neurondb -c "SELECT 1;"
```

### Permission errors

Ensure the neurondb user has appropriate permissions:
```bash
sudo chown -R neurondb:neurondb /opt/neurondb
sudo chmod 600 /etc/neurondb/*.env
```




