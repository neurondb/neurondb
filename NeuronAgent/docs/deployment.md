# NeuronAgent Deployment Guide

## Prerequisites

- PostgreSQL 16+ with NeuronDB extension
- Go 1.21+
- Docker (optional)

## Configuration

### Configuration File

Create a `config.yaml` file:

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  read_timeout: 30s
  write_timeout: 30s

database:
  host: "localhost"
  port: 5432
  database: "neurondb"
  user: "postgres"
  password: "postgres"
  max_open_conns: 25
  max_idle_conns: 5
  conn_max_lifetime: 5m
  conn_max_idle_time: 10m

auth:
  api_key_header: "Authorization"

logging:
  level: "info"
  format: "json"

workflow:
  base_url: "http://localhost:8080"
```

### Environment Variables

All configuration can be set via environment variables (takes precedence over config file):

**Server Configuration**:
- `SERVER_HOST` - Server bind address (default: "0.0.0.0")
- `SERVER_PORT` - Server port (default: 8080)
- `SERVER_READ_TIMEOUT` - Read timeout (default: 30s)
- `SERVER_WRITE_TIMEOUT` - Write timeout (default: 30s)

**Database Configuration**:
- `DB_HOST` - Database host (default: "localhost")
- `DB_PORT` - Database port (default: 5432)
- `DB_NAME` - Database name (default: "neurondb")
- `DB_USER` - Database user (default: "postgres")
- `DB_PASSWORD` - Database password (default: "postgres")
- `DB_MAX_OPEN_CONNS` - Max open connections (default: 25)
- `DB_MAX_IDLE_CONNS` - Max idle connections (default: 5)
- `DB_CONN_MAX_LIFETIME` - Connection max lifetime (default: 5m)
- `DB_CONN_MAX_IDLE_TIME` - Connection max idle time (default: 10m)

**Authentication**:
- `AUTH_API_KEY_HEADER` - API key header name (default: "Authorization")

**Logging**:
- `LOG_LEVEL` - Log level: debug, info, warn, error (default: "info")
- `LOG_FORMAT` - Log format: json, text, console (default: "json")

**Configuration File**:
- `CONFIG_PATH` - Path to config.yaml file

### Configuration Priority

1. Environment variables (highest priority)
2. Config file (`config.yaml`)
3. Default values (lowest priority)

## Database Setup

### Prerequisites

1. **PostgreSQL 16+** installed and running
2. **NeuronDB extension** installed in PostgreSQL

### Setup Steps

1. **Create database**:
```sql
CREATE DATABASE neurondb;
```

2. **Install NeuronDB extension**:
```sql
\c neurondb
CREATE EXTENSION IF NOT EXISTS neurondb;
```

3. **Verify NeuronDB extension**:
```sql
SELECT * FROM pg_extension WHERE extname = 'neurondb';
```

4. **Run NeuronAgent migrations**:
```bash
# From NeuronAgent directory
psql -d neurondb -f sql/neuronagent_initial_schema.sql
psql -d neurondb -f sql/neuronagent_add_indexes.sql
psql -d neurondb -f sql/neuronagent_add_triggers.sql

# Optional: Additional features
psql -d neurondb -f sql/neuronagent_advanced_features.sql
psql -d neurondb -f sql/neuronagent_workflow_engine.sql
psql -d neurondb -f sql/neuronagent_budget_schema.sql
psql -d neurondb -f sql/neuronagent_collaboration_workspace.sql
psql -d neurondb -f sql/neuronagent_evaluation_framework.sql
psql -d neurondb -f sql/neuronagent_hierarchical_memory.sql
psql -d neurondb -f sql/neuronagent_virtual_filesystem.sql
```

5. **Verify setup**:
```sql
-- Check tables exist
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'neurondb_agent';

-- Check NeuronDB functions are available
SELECT proname FROM pg_proc WHERE proname LIKE 'neurondb_%';
```

## Running

### Local Development

```bash
go mod download
go run cmd/agent-server/main.go
```

### Docker

```bash
docker-compose up -d
```

### Production

Build binary:
```bash
go build -o agent-server ./cmd/agent-server
```

Run:
```bash
./agent-server
```

## API Key Generation

### Generate API Key

**Option 1: Using Script**:
```bash
./scripts/neuronagent_generate_keys.sh
```

**Option 2: Programmatically**:
```go
keyManager := auth.NewAPIKeyManager(queries)
apiKey, err := keyManager.GenerateKey(ctx, "my-key-name")
```

**Option 3: Direct Database**:
```sql
INSERT INTO neurondb_agent.api_keys (name, key_hash, created_at)
VALUES ('my-key', crypt('your-secret-key', gen_salt('bf')), NOW());
```

### Using API Key

All API requests require authentication:
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8080/api/v1/agents
```

## Health Check

```
GET /health
```

Returns 200 if healthy, 503 if database connection fails.

**Response**:
```json
{
  "status": "ok"
}
```

## Production Deployment

### Best Practices

1. **Database**:
   - Use connection pooling (configure `max_open_conns`, `max_idle_conns`)
   - Set appropriate timeouts
   - Enable SSL/TLS for database connections
   - Regular backups

2. **Security**:
   - Change default passwords
   - Use strong API keys
   - Enable rate limiting
   - Use HTTPS in production
   - Configure CORS appropriately

3. **Performance**:
   - Tune connection pool settings
   - Monitor database performance
   - Use HNSW indexes for vector search
   - Enable query logging for optimization

4. **Monitoring**:
   - Enable Prometheus metrics
   - Set up structured logging
   - Monitor API response times
   - Track error rates

5. **Scaling**:
   - Use load balancer for multiple instances
   - Configure database read replicas
   - Use connection pooling
   - Monitor resource usage

### Docker Deployment

```bash
# Build image
docker build -t neuronagent:latest -f docker/Dockerfile .

# Run container
docker run -d \
  -p 8080:8080 \
  -e DB_HOST=postgres \
  -e DB_PORT=5432 \
  -e DB_NAME=neurondb \
  -e DB_USER=neurondb \
  -e DB_PASSWORD=neurondb \
  -e NEURONAGENT_API_KEY=your_api_key \
  neuronagent:latest
```

### Kubernetes Deployment

See `helm/neurondb/` for Helm charts and Kubernetes deployment configurations.

## Troubleshooting

### Database Connection Issues

```bash
# Test connection
psql -h localhost -p 5432 -U neurondb -d neurondb -c "SELECT 1;"

# Check NeuronDB extension
psql -d neurondb -c "SELECT * FROM pg_extension WHERE extname = 'neurondb';"
```

### Service Won't Start

1. Check database connection
2. Verify environment variables
3. Check logs for errors
4. Verify NeuronDB extension is installed

### Performance Issues

1. Check database connection pool usage
2. Monitor query performance
3. Verify indexes exist (especially HNSW for vectors)
4. Review connection pool settings

