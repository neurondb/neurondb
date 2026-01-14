# NeuronMCP Setup for NeuronDesktop

## Status: ✅ Configured

NeuronMCP has been built and configured to work with NeuronDesktop.

## Configuration Details

### Profile Created
- **Name:** NeuronMCP Profile
- **Profile ID:** `ce91cab1-bbfb-4405-8d38-1b908eca0b6c`
- **MCP Binary:** `neurondb-mcp` (or full path to binary)
- **Database:** `postgresql://neurondb:neurondb@localhost:5432/neurondb`

### MCP Configuration
```json
{
  "command": "neurondb-mcp",
  "args": [],
  "env": {
    "NEURONDB_HOST": "localhost",
    "NEURONDB_PORT": "5432",
    "NEURONDB_DATABASE": "neurondb",
    "NEURONDB_USER": "neurondb",
    "NEURONDB_PASSWORD": "neurondb"
  }
}
```

## How It Works

1. **NeuronDesktop** spawns the NeuronMCP server process using the configured command
2. **NeuronMCP** connects to NeuronDB PostgreSQL database using environment variables
3. Communication happens via **stdio** (JSON-RPC 2.0 protocol)
4. NeuronDesktop proxies MCP requests/responses to/from the frontend

## Accessing in NeuronDesktop

1. Open NeuronDesktop: http://localhost:3000
2. Go to **Settings** page
3. Select the **NeuronMCP Profile** from the profile selector
4. Go to **MCP Console** page
5. The profile should automatically connect to NeuronMCP

## Testing the Connection

### Via API (requires API key):
```bash
# Get API key from Settings page first
curl http://localhost:8081/api/v1/profiles/{profile-id}/mcp/tools \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Via Frontend:
1. Navigate to http://localhost:3000/mcp
2. Select "NeuronMCP Profile" from dropdown
3. Click "Load Tools" - should show all NeuronMCP tools

## Troubleshooting

### MCP Server Not Starting
- Check binary exists: `which neurondb-mcp` or check your installation path
- Check binary is executable: `chmod +x /path/to/neurondb-mcp`
- Check database connection: `psql -d neurondb -c "SELECT 1;"`

### Connection Errors
- Verify NeuronDB extension is installed: `psql -d neurondb -c "SELECT neurondb.version();"`
- Check environment variables in profile match your database setup
- Check backend logs: `tail -f /tmp/neurondesk-api.log`

### Profile Not Showing
- Refresh the page
- Check database: `psql -d neurondesk -c "SELECT * FROM profiles;"`
- Create profile via Settings page if needed

## Rebuilding NeuronMCP

If you need to rebuild NeuronMCP:

```bash
cd /path/to/neurondb2/NeuronMCP
go build -o bin/neurondb-mcp ./cmd/neurondb-mcp
```

## Updating Profile Configuration

### Via Database:
```sql
UPDATE profiles 
SET mcp_config = '{
  "command": "neurondb-mcp",
  "args": [],
  "env": {
    "NEURONDB_HOST": "localhost",
    "NEURONDB_PORT": "5432",
    "NEURONDB_DATABASE": "neurondb",
    "NEURONDB_USER": "neurondb",
    "NEURONDB_PASSWORD": "neurondb"
  }
}'::jsonb
WHERE name = 'NeuronMCP Profile';
```

### Via Frontend:
1. Go to Settings page
2. Select profile
3. Update MCP Configuration section
4. Click "Save MCP Configuration"

## Next Steps

1. ✅ NeuronMCP built and configured
2. ✅ Profile created in NeuronDesktop
3. ⏭️ Test connection in MCP Console
4. ⏭️ Use tools from NeuronMCP in NeuronDesktop



