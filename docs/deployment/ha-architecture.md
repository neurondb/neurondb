# High Availability Architecture

High availability (HA) for the NeuronDB extension in production: PostgreSQL HA and connection pooling.

---

## Overview

NeuronDB runs inside PostgreSQL. HA is achieved by making PostgreSQL highly available. Recommended options:

- **Kubernetes:** CloudNativePG (CNPG) for PostgreSQL with automatic failover
- **Non-Kubernetes:** Patroni + PgBouncer (or similar)

## Database layer

### Kubernetes: CloudNativePG (recommended)

- **Cluster:** Primary + N standbys (e.g. `neurondb.cnpg.instances: 3`)
- **Services:** `-rw` (primary), `-ro` (read-only replicas), `-r` (any replica)
- **Failover:** CNPG promotes a standby when the primary is lost (~30s)
- **Pooler:** Optional CNPG Pooler (PgBouncer) for connection pooling
- **Synchronous replication:** Use `minSyncReplicas` / `maxSyncReplicas` for RPO=0

See [Kubernetes/Helm](kubernetes-helm.md) for chart configuration.

### Outside Kubernetes: Patroni + PgBouncer

- Run PostgreSQL with Patroni for automatic failover and replication.
- Put PgBouncer in front of PostgreSQL for connection pooling.
- Applications connect to PgBouncer; Patroni manages primary/replica roles.

## Connection pooling

Use PgBouncer (or the CNPG Pooler) in front of PostgreSQL to limit connections and improve stability. Configure pool mode (e.g. transaction) and pool size to match your workload.

## Failover

- **Database primary failure:** CNPG or Patroni promotes a standby; clients using the primary endpoint reconnect to the new primary.
- **Application tier:** If you run application services that use NeuronDB, run multiple replicas behind a load balancer so one node failure does not cut access.

## Monitoring

- **PostgreSQL:** `pg_isready`, replication lag, connection counts
- **Backups:** WAL archiving and full backups (see [Backup and restore](backup-restore.md))

## Best practices

1. Use connection pooling (PgBouncer or CNPG Pooler).
2. Monitor replication lag (keep it low).
3. Test failover regularly.
4. Use health checks for database and any app tier.

---

[Deployment](README.md) · [Documentation](../readme.md)
