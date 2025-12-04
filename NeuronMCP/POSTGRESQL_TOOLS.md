# PostgreSQL Tools for NeuronMCP

## Overview

Three new tools have been added to NeuronMCP for PostgreSQL version and statistics monitoring:

1. **postgresql_version** - Get PostgreSQL server version information
2. **postgresql_stats** - Get comprehensive PostgreSQL server statistics
3. **postgresql_databases** - List all PostgreSQL databases

## Tools

### 1. postgresql_version

Get PostgreSQL server version and build information.

**Parameters**: None

**Returns**:
- `version` - Full PostgreSQL version string
- `pg_version` - PostgreSQL version function output
- `server_version` - Server version string
- `server_version_num` - Numeric version number
- `major_version` - Major version number
- `minor_version` - Minor version number
- `patch_version` - Patch version number

**Example**:
```bash
./bin/neurondb-mcp-client -c neuronmcp_server.json -e "postgresql_version"
```

### 2. postgresql_stats

Get comprehensive PostgreSQL server statistics including database size, connection info, table stats, and performance metrics.

**Parameters**:
- `include_database_stats` (boolean, default: true) - Include database-level statistics
- `include_table_stats` (boolean, default: true) - Include table statistics
- `include_connection_stats` (boolean, default: true) - Include connection statistics
- `include_performance_stats` (boolean, default: true) - Include performance metrics

**Returns**:
- `database` - Database statistics (size, schemas, etc.)
- `connections` - Connection statistics (active, idle, max, usage %)
- `tables` - Table statistics (count, sizes, indexes)
- `performance` - Performance metrics (scans, inserts, updates, deletes, cache hit ratios)
- `server` - Server configuration and uptime

**Examples**:
```bash
# Get all statistics
./bin/neurondb-mcp-client -c neuronmcp_server.json -e "postgresql_stats"

# Get only database statistics
./bin/neurondb-mcp-client -c neuronmcp_server.json -e "postgresql_stats:include_database_stats=true,include_table_stats=false,include_connection_stats=false,include_performance_stats=false"

# Get only performance statistics
./bin/neurondb-mcp-client -c neuronmcp_server.json -e "postgresql_stats:include_database_stats=false,include_table_stats=false,include_connection_stats=false,include_performance_stats=true"
```

### 3. postgresql_databases

List all PostgreSQL databases with their sizes and connection counts.

**Parameters**:
- `include_system` (boolean, default: false) - Include system databases (template0, template1, postgres)

**Returns**:
- `databases` - Array of database objects with:
  - `name` - Database name
  - `size_bytes` - Database size in bytes
  - `size_pretty` - Human-readable database size
  - `connections` - Number of active connections
  - `collation` - Database collation
  - `ctype` - Database character type
- `count` - Total number of databases

**Examples**:
```bash
# List user databases only
./bin/neurondb-mcp-client -c neuronmcp_server.json -e "postgresql_databases"

# List all databases including system
./bin/neurondb-mcp-client -c neuronmcp_server.json -e "postgresql_databases:include_system=true"
```

## Statistics Details

### Database Statistics
- Current database name
- Database size (bytes and human-readable)
- Total number of databases
- Number of user schemas

### Connection Statistics
- Active connections
- Idle connections
- Idle in transaction connections
- Total connections
- Maximum connections
- Connection usage percentage

### Table Statistics
- Number of user tables
- Total number of tables
- Total size of user tables (bytes and human-readable)
- Number of user indexes
- Total number of indexes

### Performance Statistics
- Total sequential scans
- Total index scans
- Total inserts, updates, deletes
- Total live tuples
- Total dead tuples
- Dead tuple percentage
- Number of tables vacuumed
- Number of tables analyzed
- Heap cache hit ratio
- Index cache hit ratio

### Server Information
- Server version
- Shared buffers setting
- Effective cache size
- Work memory setting
- Maintenance work memory
- Maximum connections
- Checkpoint timeout
- Server start time
- Server uptime

## Integration

All tools are automatically registered when the MCP server starts. They are available through:
- MCP client (`neurondb-mcp-client`)
- Claude Desktop
- Any MCP-compatible client

## Requirements

- PostgreSQL database connection (configured in `neuronmcp_server.json`)
- Appropriate database permissions to query system catalogs

## Error Handling

Tools gracefully handle:
- Database connection errors
- Missing permissions
- Query execution errors

All errors are returned in the standard MCP error format with detailed messages.

