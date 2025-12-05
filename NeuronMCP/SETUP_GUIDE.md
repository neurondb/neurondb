# NeuronMCP Setup Guide

Complete setup guide for NeuronMCP on macOS, Windows, and Linux.

## Prerequisites

- PostgreSQL 16, 17, or 18
- NeuronDB extension installed
- Go 1.23+ (for building from source)
- MCP-compatible client (Claude Desktop, etc.)

## Installation

### Build from Source

```bash
cd NeuronMCP
go build ./cmd/neurondb-mcp
```

The binary will be created at `./neurondb-mcp` (or `./neurondb-mcp.exe` on Windows).

### Using Pre-built Binary

Download the appropriate binary for your platform from releases.

## Configuration

### Environment Variables

Set these environment variables:

```bash
export NEURONDB_HOST=localhost
export NEURONDB_PORT=5432
export NEURONDB_DATABASE=neurondb
export NEURONDB_USER=neurondb
export NEURONDB_PASSWORD=your_password
```

### Configuration File

Create `mcp-config.json`:

```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "neurondb",
    "user": "neurondb",
    "password": "your_password"
  },
  "server": {
    "name": "neurondb-mcp-server",
    "version": "1.0.0"
  },
  "logging": {
    "level": "info",
    "format": "text"
  }
}
```

## Claude Desktop Setup

### macOS

1. Create configuration file:
   ```bash
   mkdir -p ~/Library/Application\ Support/Claude
   cp claude_desktop_config.macos.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

2. Edit the configuration file and update the path to `neurondb-mcp`:
   ```json
   {
     "mcpServers": {
       "neurondb": {
         "command": "/path/to/neurondb-mcp",
         "env": {
           "NEURONDB_HOST": "localhost",
           "NEURONDB_PORT": "5432",
           "NEURONDB_DATABASE": "neurondb",
           "NEURONDB_USER": "neurondb",
           "NEURONDB_PASSWORD": "your_password"
         }
       }
     }
   }
   ```

3. Restart Claude Desktop

### Windows

1. Create configuration file:
   ```
   %APPDATA%\Claude\claude_desktop_config.json
   ```

2. Copy content from `claude_desktop_config.windows.json` and update paths

3. Restart Claude Desktop

### Linux

1. Create configuration file:
   ```bash
   mkdir -p ~/.config/Claude
   cp claude_desktop_config.linux.json ~/.config/Claude/claude_desktop_config.json
   ```

2. Edit the configuration file and update the path to `neurondb-mcp`

3. Restart Claude Desktop

## Testing

### Test Connection

```bash
./neurondb-mcp-client ./neurondb-mcp tools/list
```

### Test Tool Execution

```bash
./neurondb-mcp-client ./neurondb-mcp tools/call '{"name":"postgresql_version","arguments":{}}'
```

## Troubleshooting

### Database Connection Failed

1. Verify PostgreSQL is running:
   ```bash
   psql -h localhost -p 5432 -U neurondb -d neurondb -c "SELECT 1;"
   ```

2. Check environment variables:
   ```bash
   env | grep NEURONDB
   ```

3. Verify NeuronDB extension is installed:
   ```sql
   SELECT * FROM pg_extension WHERE extname = 'neurondb';
   ```

### Claude Desktop Not Connecting

1. Check configuration file path and format
2. Verify binary path is correct and executable
3. Check Claude Desktop logs
4. Test binary manually:
   ```bash
   echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | ./neurondb-mcp
   ```

### Stdio Issues

- Ensure stdin/stdout are not redirected
- Use `-i` flag with Docker: `docker run -i --rm neurondb-mcp:latest`
- Do not pipe output: `./neurondb-mcp > output.log` (incorrect)

## Security

- Store credentials securely via environment variables
- Use TLS/SSL for encrypted database connections
- Run with non-root user in production
- No network endpoints (stdio only)

## Performance

- Connection pooling is enabled by default
- Query timeouts are set appropriately
- GPU acceleration is available when configured

