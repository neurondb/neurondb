# Component Integration Guide

Guide for integrating and connecting all NeuronDB ecosystem components.

## Overview

All components in the NeuronDB ecosystem connect to the same NeuronDB PostgreSQL instance. Services operate independently and can run separately, but work best when integrated together.

## Integration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Clients                                │
│  (Web, Mobile, CLI, MCP Clients)                          │
└──────┬──────────────┬───────────────┬──────────────────┘
       │              │               │
┌──────▼──────┐  ┌───▼────┐  ┌───────▼────────┐
│ NeuronDesktop│  │NeuronAgent│  │   NeuronMCP   │
│  (Web UI)    │  │ (REST API)│  │  (MCP Server)  │
└──────┬───────┘  └────┬────┘  └───────┬────────┘
       │              │               │
       └──────────────┴───────────────┘
                    │
            ┌───────▼────────┐
            │   NeuronDB     │
            │  (PostgreSQL)   │
            │  + Extension   │
            └────────────────┘
```

## Database Setup

### Step 1: Create Database

```bash
createdb neurondb
```

### Step 2: Install NeuronDB Extension

```sql
psql -d neurondb -c "CREATE EXTENSION neurondb;"
```

### Step 3: Run Component Migrations

**NeuronAgent:**
```bash
cd NeuronAgent
./scripts/neuronagent-migrate.sh
```
This runs all migrations including `migrations/001_initial_schema.sql` and subsequent migrations.

**NeuronDesktop:**
```bash
cd NeuronDesktop
createdb neurondesk
./scripts/neurondesktop-setup.sh
```
This runs all migrations including `api/migrations/001_initial_schema.sql` and subsequent migrations.

**NeuronMCP:**
```bash
cd NeuronMCP
./scripts/neuronmcp-setup.sh
```

## Component Configuration

### NeuronDB Configuration

All components connect to NeuronDB using PostgreSQL connection parameters:

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=neurondb
export DB_USER=neurondb
export DB_PASSWORD=neurondb
```

### NeuronAgent Configuration

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=neurondb
export DB_USER=neurondb
export DB_PASSWORD=neurondb
export SERVER_PORT=8080
```

### NeuronMCP Configuration

```bash
export NEURONDB_HOST=localhost
export NEURONDB_PORT=5432
export NEURONDB_DATABASE=neurondb
export NEURONDB_USER=neurondb
export NEURONDB_PASSWORD=neurondb
```

### NeuronDesktop Configuration

**Backend:**
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_USER=neurondesk
export DB_PASSWORD=neurondesk
export DB_NAME=neurondesk
export SERVER_PORT=8081
```

**Profile Configuration (in UI):**
- NeuronDB connection: PostgreSQL connection string
- NeuronAgent endpoint: `http://localhost:8080`
- NeuronAgent API key: API key from NeuronAgent
- NeuronMCP command: `neurondb-mcp`
- NeuronMCP environment: Environment variables for NeuronMCP

## Integration Workflows

### Workflow 1: Vector Search via NeuronDesktop

1. User accesses NeuronDesktop web interface
2. NeuronDesktop connects to NeuronDB via PostgreSQL
3. User performs vector search through UI
4. NeuronDesktop executes query on NeuronDB
5. Results displayed in web interface

### Workflow 2: Agent with Memory

1. Client sends request to NeuronAgent
2. NeuronAgent retrieves context from NeuronDB using vector search
3. Agent processes request with context
4. Agent stores new information in NeuronDB
5. Response returned to client

### Workflow 3: MCP Tool Execution

1. MCP client (e.g., Claude Desktop) connects to NeuronMCP
2. NeuronMCP provides tools for vector operations
3. Client calls tool through NeuronMCP
4. NeuronMCP executes operation on NeuronDB
5. Results returned to MCP client

### Workflow 4: RAG Pipeline

1. Documents ingested into NeuronDB
2. Embeddings generated using NeuronDB functions
3. NeuronAgent retrieves relevant context using vector search
4. Context passed to LLM through NeuronAgent
5. Response generated and returned

### Workflow 5: Multi-Agent Collaboration

1. Primary agent receives complex task requiring multiple capabilities
2. Primary agent delegates subtasks to specialized agents (research agent, analysis agent, etc.)
3. Agents communicate via NeuronAgent collaboration API
4. Each agent uses hierarchical memory to retrieve relevant context
5. Results aggregated and returned to primary agent
6. Final response generated with combined insights

### Workflow 6: DAG-Based Workflow Engine with HITL

1. User creates workflow definition with DAG structure (agent steps, tool steps, approval steps)
2. Workflow engine executes steps based on dependencies
3. Human-in-the-loop (HITL) approval step triggers notification (email/webhook)
4. Human reviews and approves/rejects via approval API
5. Workflow continues based on approval decision
6. All steps logged with execution history and state snapshots
7. Workflow completion triggers webhook notifications

### Workflow 7: Agent with Budget Management

1. Agent created with budget constraints (per-agent or per-session)
2. Agent processes requests with real-time cost tracking
3. Budget alerts triggered when approaching thresholds
4. Agent operations paused or limited when budget exceeded
5. Cost reports generated for analysis and optimization

## Docker Integration

### Unified Docker Compose

Use the unified Docker Compose setup for easy integration:

```bash
# From repository root
docker compose up -d
```

This starts all services with automatic networking.

### Start components individually

From repository root:

```bash
# Database only (CPU profile)
docker compose up -d neurondb

# Agent (depends on neurondb)
docker compose up -d neuronagent

# MCP server (depends on neurondb)
docker compose up -d neuronmcp

# Desktop UI + API
docker compose up -d neurondesk-api neurondesk-frontend
```

### Docker Network Configuration

Services communicate via Docker network using container names:

- **NeuronAgent → NeuronDB**: `neurondb-cpu:5432`
- **NeuronMCP → NeuronDB**: `neurondb-cpu:5432`
- **NeuronDesktop → NeuronDB**: `localhost:5433` (external)

## Verification

### Verify Integration

```bash
# Test NeuronDB
psql -d neurondb -c "SELECT neurondb.version();"

# Test NeuronAgent
curl http://localhost:8080/health

# Test NeuronMCP
which neurondb-mcp

# Test NeuronDesktop
curl http://localhost:8081/health
```

### Integration Test Script

Use the verification script:

```bash
./scripts/neurondb-healthcheck.sh integration
```

This tests:
- NeuronDB extension functionality
- NeuronMCP schema and functions
- NeuronAgent schema and tables
- Cross-module integration
- Vector operations

## Common Integration Patterns

### Pattern 1: Web Application

- **Frontend**: NeuronDesktop web interface
- **Backend**: NeuronDesktop API
- **Database**: NeuronDB
- **Agents**: NeuronAgent for AI capabilities
- **MCP**: NeuronMCP for tool integration

### Pattern 2: API-First Application

- **API**: NeuronAgent REST API
- **Database**: NeuronDB
- **Clients**: Custom applications using NeuronAgent API

### Pattern 3: MCP Client Integration

- **MCP Client**: Claude Desktop or custom MCP client
- **MCP Server**: NeuronMCP
- **Database**: NeuronDB

### Pattern 4: Hybrid Architecture

- **Web UI**: NeuronDesktop for management
- **API**: NeuronAgent for programmatic access
- **MCP**: NeuronMCP for MCP client integration
- **Database**: NeuronDB as shared data layer

## Troubleshooting

### Connection Issues

1. Verify all services are running
2. Check database connection parameters
3. Verify network connectivity
4. Check firewall rules
5. Review service logs

### Data Consistency

- All components use the same database instance
- Ensure migrations are run in correct order
- Verify extension is installed before other components

### Performance

- Monitor database connections
- Check connection pool settings
- Review query performance
- Monitor resource utilization

## Best Practices

1. **Use same database instance**: All components should connect to the same NeuronDB instance
2. **Run migrations in order**: Install NeuronDB extension first, then component-specific schemas
3. **Configure connection pooling**: Set appropriate pool sizes for each component
4. **Monitor integration**: Use health checks and monitoring to verify integration
5. **Test integration**: Use verification scripts to test component integration

## Official Documentation

For comprehensive integration guides:
** [https://www.neurondb.ai/docs/ecosystem](https://www.neurondb.ai/docs/ecosystem)**

