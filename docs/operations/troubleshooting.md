# Troubleshooting Guide

Common issues and fixes for the NeuronDB extension and PostgreSQL.

---

## Table of contents

1. [Common issues](#common-issues)
2. [Database and extension](#database-and-extension)
3. [Performance](#performance)
4. [Deployment](#deployment)

---

## Common issues

### Services not starting (Docker)

**Symptoms:** Container fails to start or exits immediately.

**Diagnosis:**
```bash
docker compose -f docker/docker-compose.yml logs neurondb
docker compose -f docker/docker-compose.yml ps
```

**Solutions:**
1. Check environment variables (e.g. `POSTGRES_PASSWORD` in `.env`).
2. Check port conflicts: `lsof -i :5433` or `ss -tuln | grep 5433`.
3. Check disk space and memory.

### Connection refused

**Symptoms:** Cannot connect to PostgreSQL.

**Diagnosis:**
```bash
docker compose -f docker/docker-compose.yml ps neurondb
docker compose -f docker/docker-compose.yml exec neurondb pg_isready -U neurondb -d neurondb
```

**Solutions:**
1. Ensure the neurondb service is running.
2. Use correct host/port: default host port is 5433 when using Compose from repo root.
3. Check firewall if connecting remotely: allow TCP on the PostgreSQL port.

---

## Database and extension

### PostgreSQL connection failed

**Symptoms:** `psql` or application cannot connect.

**Diagnosis:**
```bash
# With Docker Compose
docker compose -f docker/docker-compose.yml exec neurondb psql -U neurondb -d neurondb -c "SELECT 1;"

# Or from host
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c "SELECT 1;"
```

**Solutions:**
1. Use correct credentials (default dev: user `neurondb`, password `neurondb`, database `neurondb`). Change in production.
2. Confirm PostgreSQL is running and `pg_isready` succeeds.
3. Check `max_connections` and current connections: `SELECT count(*) FROM pg_stat_activity;`

### NeuronDB extension not loading

**Symptoms:** `function neurondb.version() does not exist` or similar.

**Diagnosis:**
```sql
SELECT * FROM pg_extension WHERE extname = 'neurondb';
\dx neurondb
```

**Solutions:**
1. Create the extension in the database: `CREATE EXTENSION IF NOT EXISTS neurondb;`
2. Confirm extension files are installed (e.g. `ls /usr/lib/postgresql/*/lib/neurondb.*` or equivalent for your install).
3. Check PostgreSQL version matches the built extension (e.g. PG 16/17/18).

---

## Performance

### Slow queries

**Diagnosis:**
```sql
-- If pg_stat_statements is available
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Index usage
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
WHERE indexname LIKE '%hnsw%' OR indexname LIKE '%ivf%';
```

**Solutions:**
1. Add or adjust indexes (e.g. HNSW/IVF for vector columns).
2. Run `ANALYZE` on relevant tables.
3. Use connection pooling (e.g. PgBouncer) if connection count is high.

### High memory or CPU

- Set resource limits in Docker/Kubernetes if needed.
- Tune PostgreSQL parameters (e.g. `work_mem`, `shared_buffers`) per [configuration](configuration.md) and [operational playbook](playbook.md).

---

## Deployment

### Docker Compose not starting

- Run `docker compose -f docker/docker-compose.yml config` to validate.
- Ensure Docker and Docker Compose versions are sufficient (see [Docker deployment](../deployment/docker.md)).
- Use the correct Compose file path: `docker/docker-compose.yml` when running from repo root.

### Helm / Kubernetes

- Run `helm lint` and `helm install --dry-run --debug` to validate chart and values.
- Check node resources and pod events: `kubectl describe pod <pod-name>`.

### Backup / restore

- Ensure sufficient disk space and permissions for backup/restore commands.
- Use `pg_dump` / `pg_restore` or filesystem backup as in [Backup and restore](../deployment/backup-restore.md).

---

## Getting help

- **Logs:** Docker: `docker compose -f docker/docker-compose.yml logs neurondb`. System: `journalctl -u postgresql` or equivalent.
- **Documentation:** [Getting started troubleshooting](../getting-started/troubleshooting.md), [Operational playbook](playbook.md), [Documentation index](../readme.md).

---

## Related documentation

| Document | Description |
|----------|-------------|
| [Getting started troubleshooting](../getting-started/troubleshooting.md) | Setup and first-run issues |
| [Observability setup](observability-setup.md) | Monitoring and diagnostics |
| [Deployment](../deployment/README.md) | Deployment options |

---

[Operations](README.md) · [Documentation](../readme.md)
