# launchd Service Files (macOS)

This directory contains launchd plist files for running NeuronDB ecosystem components as background services on macOS.

## Installation

### User-level (Recommended for development)

1. Create the log directory:
   ```bash
   mkdir -p ~/Library/Logs/neurondb
   ```

2. Create the working directory:
   ```bash
   mkdir -p ~/usr/local/var/neurondb
   ```

3. Copy plist files to LaunchAgents:
   ```bash
   cp scripts/services/launchd/*.plist ~/Library/LaunchAgents/
   ```

4. Edit plist files to update paths and environment variables:
   ```bash
   nano ~/Library/LaunchAgents/com.neurondb.neuronmcp.plist
   ```

   Update:
   - Binary paths (if not in `/usr/local/bin/`)
   - Environment variables (database credentials, ports, etc.)
   - Working directory
   - Log paths

5. Load and start services:
   ```bash
   launchctl load ~/Library/LaunchAgents/com.neurondb.neuronmcp.plist
   launchctl load ~/Library/LaunchAgents/com.neurondb.neuronagent.plist
   launchctl load ~/Library/LaunchAgents/com.neurondb.neurondesktop-api.plist
   
   launchctl start com.neurondb.neuronmcp
   launchctl start com.neurondb.neuronagent
   launchctl start com.neurondb.neurondesktop-api
   ```

### System-level (Requires root)

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
   ```

3. Edit plist files (as root):
   ```bash
   sudo nano /Library/LaunchDaemons/com.neurondb.neuronmcp.plist
   ```

4. Set ownership:
   ```bash
   sudo chown root:wheel /Library/LaunchDaemons/com.neurondb.*.plist
   ```

5. Load and start services:
   ```bash
   sudo launchctl load /Library/LaunchDaemons/com.neurondb.neuronmcp.plist
   sudo launchctl load /Library/LaunchDaemons/com.neurondb.neuronagent.plist
   sudo launchctl load /Library/LaunchDaemons/com.neurondb.neurondesktop-api.plist
   
   sudo launchctl start com.neurondb.neuronmcp
   sudo launchctl start com.neurondb.neuronagent
   sudo launchctl start com.neurondb.neurondesktop-api
   ```

## Management

### Check status
```bash
launchctl list | grep neurondb
```

### View logs
```bash
# User-level logs
tail -f ~/Library/Logs/neurondb/neuronmcp.log
tail -f ~/Library/Logs/neurondb/neuronmcp.error.log

# System-level logs
tail -f /usr/local/var/log/neurondb/neuronmcp.log
tail -f /usr/local/var/log/neurondb/neuronmcp.error.log

# System console logs
log show --predicate 'process == "neurondb-mcp"' --last 1h
```

### Stop services
```bash
# User-level
launchctl stop com.neurondb.neuronmcp
launchctl stop com.neurondb.neuronagent
launchctl stop com.neurondb.neurondesktop-api

# System-level
sudo launchctl stop com.neurondb.neuronmcp
sudo launchctl stop com.neurondb.neuronagent
sudo launchctl stop com.neurondb.neurondesktop-api
```

### Unload services
```bash
# User-level
launchctl unload ~/Library/LaunchAgents/com.neurondb.neuronmcp.plist
launchctl unload ~/Library/LaunchAgents/com.neurondb.neuronagent.plist
launchctl unload ~/Library/LaunchAgents/com.neurondb.neurondesktop-api.plist

# System-level
sudo launchctl unload /Library/LaunchDaemons/com.neurondb.neuronmcp.plist
sudo launchctl unload /Library/LaunchDaemons/com.neurondb.neuronagent.plist
sudo launchctl unload /Library/LaunchDaemons/com.neurondb.neurondesktop-api.plist
```

### Restart services
```bash
launchctl stop com.neurondb.neuronmcp && launchctl start com.neurondb.neuronmcp
```

## Customization

### Changing Binary Location

Edit the plist file and update the `ProgramArguments` array:
```xml
<key>ProgramArguments</key>
<array>
    <string>/path/to/your/neurondb-mcp</string>
</array>
```

Then reload:
```bash
launchctl unload ~/Library/LaunchAgents/com.neurondb.neuronmcp.plist
launchctl load ~/Library/LaunchAgents/com.neurondb.neuronmcp.plist
```

### Using Environment File

launchd doesn't support EnvironmentFile directly. You can:

1. Source environment variables in a wrapper script
2. Use a tool like `envdir` or `direnv`
3. Manually set all environment variables in the plist file

### Log Rotation

macOS automatically rotates logs, but you can configure log rotation by:

1. Using `log` command filters
2. Installing a log rotation tool like `newsyslog`
3. Creating a separate cron job or launchd agent for log rotation

## Troubleshooting

### Service fails to start

1. Check logs:
   ```bash
   tail -f ~/Library/Logs/neurondb/neuronmcp.error.log
   ```

2. Verify plist syntax:
   ```bash
   plutil -lint ~/Library/LaunchAgents/com.neurondb.neuronmcp.plist
   ```

3. Check if service is loaded:
   ```bash
   launchctl list | grep neurondb
   ```

4. Test running the binary manually:
   ```bash
   /usr/local/bin/neurondb-mcp
   ```

### Service starts but exits immediately

1. Check KeepAlive is set to true
2. Check error logs for crash reasons
3. Verify database connection and credentials
4. Ensure all required environment variables are set

### Permission errors

1. Ensure log directory is writable:
   ```bash
   chmod 755 ~/Library/Logs/neurondb
   ```

2. Ensure working directory is writable:
   ```bash
   chmod 755 ~/usr/local/var/neurondb
   ```

3. For system-level, ensure proper ownership:
   ```bash
   sudo chown root:wheel /Library/LaunchDaemons/com.neurondb.*.plist
   ```

## Notes

- User-level services (LaunchAgents) start when the user logs in
- System-level services (LaunchDaemons) start at boot
- KeepAlive=true ensures the service restarts if it crashes
- RunAtLoad=true starts the service immediately when loaded
- Environment variables in plist files are limited; consider using wrapper scripts for complex configurations



