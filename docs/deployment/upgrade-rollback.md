# Upgrade and Rollback Guide

<div align="center">

**Complete procedures for upgrading and rolling back NeuronDB**

[![Upgrade](https://img.shields.io/badge/upgrade-guide-blue)](.)
[![Status](https://img.shields.io/badge/status-stable-brightgreen)](.)

</div>

---

> [!WARNING]
> Always back up your database before upgrading. Test upgrades in a development environment first.

---

## Pre-Upgrade Checklist

<details>
<summary><strong>✅ Pre-Upgrade Checklist</strong></summary>

| Task | Command | Required |
|------|---------|----------|
| **Review release notes** | Check CHANGELOG.md | ⚠️ Critical |
| **Backup database** | See [Backup Guide](./backup-restore.md) | ⚠️ Critical |
| **Verify current version** | `helm list -n neurondb` | ✅ Yes |
| **Check migration status** | `kubectl get jobs -n neurondb \| grep migration` | ✅ Yes |
| **Ensure resources** | Check cluster capacity | ✅ Yes |
| **Review values.yaml** | Check for deprecated options | ⭐ High |

</details>

## Upgrade Procedure

### Step 1: Backup

```bash
# Ensure backup is recent
kubectl get cronjob neurondb-backup -n neurondb
kubectl create job --from=cronjob/neurondb-backup manual-backup-$(date +%s) -n neurondb
```

### Step 2: Review Changes

```bash
# Dry-run upgrade
helm upgrade neurondb ./helm/neurondb \
  --version <new-version> \
  --dry-run --debug \
  -f values-production-external-postgres.yaml \
  -n neurondb
```

### Step 3: Run Migrations

Migrations run automatically via pre-upgrade hooks. Monitor:

```bash
# Watch migration job
kubectl get jobs -n neurondb -w | grep migration
kubectl logs -f job/neurondb-migration-<revision> -n neurondb
```

### Step 4: Upgrade Chart

```bash
helm upgrade neurondb ./helm/neurondb \
  --version <new-version> \
  -f values-production-external-postgres.yaml \
  -n neurondb
```

### Step 5: Monitor Upgrade

```bash
# Watch pods
kubectl get pods -n neurondb -w

# Check rollout status
kubectl rollout status statefulset/neurondb-neurondb -n neurondb
kubectl rollout status deployment/neurondb-neuronagent -n neurondb

# Check logs
kubectl logs -f deployment/neurondb-neuronagent -n neurondb
```

### Step 6: Verify Upgrade

```bash
# Check versions
helm list -n neurondb
kubectl get pods -n neurondb -o jsonpath='{.items[*].spec.containers[*].image}'

# Test functionality
kubectl port-forward svc/neurondb-neuronagent 8080:8080 -n neurondb
curl http://localhost:8080/health
```

## Rollback Procedure

### Step 1: Identify Previous Revision

```bash
# List release history
helm history neurondb -n neurondb

# Note the revision number to rollback to
```

### Step 2: Rollback Helm Release

```bash
# Rollback to previous revision
helm rollback neurondb <revision> -n neurondb

# Or rollback to previous version
helm rollback neurondb -n neurondb
```

### Step 3: Database Rollback (if needed)

If database migrations were applied, rollback them:

```bash
# Connect to database
kubectl exec -it statefulset/neurondb-neurondb -n neurondb -- \
  psql -U neurondb -d neurondb

# Run rollback script (if provided)
\i /path/to/rollback.sql
```

### Step 4: Verify Rollback

```bash
# Check pod versions
kubectl get pods -n neurondb -o jsonpath='{.items[*].spec.containers[*].image}'

# Verify functionality
kubectl port-forward svc/neurondb-neuronagent 8080:8080 -n neurondb
curl http://localhost:8080/health
```

## Zero-Downtime Upgrades

### StatefulSet Rolling Update

Configure update strategy:

```yaml
neurondb:
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      partition: null  # Update all pods
      maxUnavailable: 1
```

### Canary Deployment

Gradual rollout using partition:

```yaml
neurondb:
  replicas: 3
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      partition: 2  # Only update pod 2 and 3
```

After verification, reduce partition:
```bash
helm upgrade neurondb ./helm/neurondb \
  --set neurondb.updateStrategy.rollingUpdate.partition=1 \
  -n neurondb
```

## Migration Management

### Check Migration Status

```bash
# View migration history
kubectl get jobs -n neurondb | grep migration

# Check database migration version
kubectl exec -it statefulset/neurondb-neurondb -n neurondb -- \
  psql -U neurondb -d neurondb -c \
  "SELECT version, name, applied_at FROM neurondb_agent.schema_migrations ORDER BY version DESC;"
```

### Manual Migration

If automatic migration fails:

```bash
# Create migration job manually
kubectl create job --from=cronjob/neurondb-migration manual-migration -n neurondb

# Or run migration script directly
kubectl exec -it statefulset/neurondb-neurondb -n neurondb -- \
  psql -U neurondb -d neurondb -f /path/to/migration.sql
```

## Troubleshooting

### Upgrade Fails

1. Check migration job:
```bash
kubectl describe job neurondb-migration-<revision> -n neurondb
kubectl logs job/neurondb-migration-<revision> -n neurondb
```

2. Check pod status:
```bash
kubectl describe pod <pod-name> -n neurondb
kubectl logs <pod-name> -n neurondb
```

3. Rollback if needed:
```bash
helm rollback neurondb -n neurondb
```

### Migration Fails

1. Check migration logs:
```bash
kubectl logs job/neurondb-migration-<revision> -n neurondb
```

2. Fix migration issues and retry:
```bash
# Delete failed job
kubectl delete job neurondb-migration-<revision> -n neurondb

# Retry upgrade
helm upgrade neurondb ./helm/neurondb --version <version> -n neurondb
```

### Rollback Fails

1. Check release history:
```bash
helm history neurondb -n neurondb
```

2. Manual rollback:
```bash
# Delete current resources
helm uninstall neurondb -n neurondb

# Reinstall previous version
helm install neurondb ./helm/neurondb \
  --version <previous-version> \
  -f values-production-external-postgres.yaml \
  -n neurondb
```

## Best Practices

> [!IMPORTANT]
> Follow these practices for safe upgrades.

<details>
<summary><strong>✅ Upgrade Best Practices</strong></summary>

| Practice | Description | Priority |
|----------|-------------|----------|
| **Always backup** | Backup before upgrade | ⚠️ Critical |
| **Test in staging** | Test upgrades in staging first | ⚠️ Critical |
| **Review changes** | Review breaking changes in release notes | ⚠️ Critical |
| **Monitor** | Monitor during and after upgrade | ⚠️ Critical |
| **Rollback plan** | Keep rollback plan ready | ⚠️ Critical |
| **Document migrations** | Document custom migrations | ⭐ High |
| **Canary deployments** | Use canary deployments for major upgrades | ⭐ High |

</details>

---

## 🔗 Related Documentation

| Document | Description |
|----------|-------------|
| **[Production Installation](production-install.md)** | Production setup guide |
| **[Backup and Restore](backup-restore.md)** | Backup procedures |
| **[Troubleshooting](../operations/troubleshooting.md)** | Common issues |
| **[CHANGELOG](../../CHANGELOG.md)** | Version history |

---

<div align="center">

[⬆ Back to Top](#upgrade-and-rollback-guide) · [📚 Deployment Index](README.md) · [📚 Main Documentation](../../README.md)

</div>

