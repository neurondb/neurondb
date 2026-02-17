# Disaster Recovery

## Objectives

- **RTO (Recovery Time Objective)**: Target time to restore service (e.g. 4 hours).
- **RPO (Recovery Point Objective)**: Target maximum data loss (e.g. 1 hour; drive backup frequency accordingly).

## Backups

- **PostgreSQL (NeuronDB)**: Run `pg_dump` or use a continuous backup solution (e.g. WAL archiving, managed DB backups). Schedule daily full backups and retain per policy (e.g. 30 days). Test restore to a staging instance regularly.
- **Application state**: NeuronDesktop/NeuronAgent store minimal state in DB; backup the DB. Config and secrets are in env or Vault; document restore steps.

## Failover

- **Database**: If using a managed Postgres with replicas, promote a replica to primary and point applications to the new primary. Update connection strings and restart app pods.
- **Application services**: Deploy multiple replicas behind a load balancer; if a region or AZ fails, traffic can be shifted (DNS or LB) to another region.

## Recovery procedure

1. Declare incident and notify stakeholders.
2. Restore DB from latest backup to a new instance (or promote replica).
3. Point NeuronAgent, NeuronDesktop, NeuronMCP to the restored DB; restart services.
4. Run smoke tests (extension load, one train, one predict).
5. Cut over traffic (or keep read-only until validated).
6. Post-mortem and update runbooks.

## Testing

- Quarterly: Restore backup to staging and run full test suite.
- **Backup and restore verification**: Use `scripts/backup_restore_verify.sh` to create a `pg_dump` and optionally restore to a temporary database to verify the dump and that the NeuronDB extension is present. Example:
  - `PGDATABASE=neurondb ./scripts/backup_restore_verify.sh` — dump and verify restore.
  - `./scripts/backup_restore_verify.sh --restore-only ./backups/neurondb_YYYYMMDD_HHMMSS.dump` — verify an existing dump.
