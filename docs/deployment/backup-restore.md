# Backup and Restore Guide

Complete guide for backing up and restoring NeuronDB. On Kubernetes with **CloudNativePG (CNPG)**, use the chart’s CNPG backup and ScheduledBackup; otherwise use the legacy CronJob or manual procedures below.

## CloudNativePG (CNPG) Backup (Kubernetes, recommended)

When the Helm chart deploys PostgreSQL via CNPG, backups are configured through the Cluster and optional ScheduledBackup resources.

### Enable CNPG Backup (Barman object store)

Configure WAL archiving and full backups to S3, GCS, or Azure:

```yaml
neurondb:
  cnpg:
    backup:
      enabled: true
      barmanObjectStore:
        destinationPath: "s3://my-bucket/neurondb-backups/"
        s3Credentials:
          accessKeyId:    { name: "backup-credentials", key: "ACCESS_KEY_ID" }
          secretAccessKey: { name: "backup-credentials", key: "ACCESS_SECRET_KEY" }
          region: "us-east-1"
        wal:
          compression: "gzip"
        data:
          compression: "gzip"
          jobs: 2
      retentionPolicy: "30d"
    scheduledBackup:
      enabled: true
      schedule: "0 0 2 * * *"   # Daily at 02:00
      target: "prefer-standby"
```

Create the secret referenced above:

```bash
kubectl create secret generic backup-credentials -n neurondb \
  --from-literal=ACCESS_KEY_ID="<key>" \
  --from-literal=ACCESS_SECRET_KEY="<secret>"
```

For GCS or Azure, use `googleCredentials` or `azureCredentials` in `barmanObjectStore` (see chart `values.yaml`).

### On-demand backup (CNPG)

```bash
kubectl cnpg backup neurondb-neurondb -n neurondb --backup-name manual-$(date +%Y%m%d-%H%M)
kubectl get backup -n neurondb
```

### Point-in-time recovery (PITR) with CNPG

Create a new Cluster that recovers from a backup or another cluster. Set `neurondb.cnpg.bootstrap.method: recovery`, `neurondb.cnpg.bootstrap.recovery.enabled: true`, `bootstrap.recovery.source` to the backup name (or external cluster), and optionally `recoveryTarget.targetTime` (e.g. `"2025-02-26T12:00:00Z"`). Deploy the chart (or a second release) with these values; after recovery, point applications to the new Cluster’s `-rw` service.

See [Kubernetes/Helm](kubernetes-helm.md#cloudnativepg-cnpg) and the chart’s `values.yaml` for all CNPG backup options (volume snapshot, retention, etc.).

---

## Quick Start (legacy / non-CNPG)

### Backup

When using **CNPG**, enable backup as in [CloudNativePG (CNPG) Backup](#cloudnativepg-cnpg-backup-kubernetes-recommended) above.

For **legacy** (CronJob-style) backup with S3:

```bash
helm upgrade neurondb ./helm/neurondb \
  --set backup.enabled=true \
  --set backup.s3.enabled=true \
  --set backup.s3.bucket=neurondb-backups \
  --set backup.s3.region=us-east-1 \
  -n neurondb
```

### Restore

```bash
# Restore from S3 backup
helm upgrade neurondb ./helm/neurondb \
  --set backup.restore.enabled=true \
  --set backup.restore.backupFile=neurondb-backup-20240101-020000.dump \
  --set backup.restore.fromS3=true \
  -n neurondb
```

## Backup Configuration

### S3 Backup

1. Create IAM user with S3 permissions
2. Create secret:
```bash
kubectl create secret generic neurondb-backup-credentials \
  --from-literal=aws-access-key-id=<key> \
  --from-literal=aws-secret-access-key=<secret> \
  -n neurondb
```

3. Configure values:
```yaml
backup:
  enabled: true
  schedule: "0 2 * * *"  # Daily at 2 AM
  retention: 30  # days
  s3:
    enabled: true
    bucket: neurondb-backups
    region: us-east-1
    prefix: backups
```

### GCS Backup

1. Create service account and download JSON key
2. Create secret:
```bash
kubectl create secret generic neurondb-backup-credentials \
  --from-file=gcs-credentials.json=service-account.json \
  -n neurondb
```

3. Configure values:
```yaml
backup:
  enabled: true
  gcs:
    enabled: true
    bucket: neurondb-backups
    prefix: backups
```

### Azure Blob Storage Backup

1. Create secret:
```bash
kubectl create secret generic neurondb-backup-credentials \
  --from-literal=azure-client-id=<client-id> \
  --from-literal=azure-client-secret=<secret> \
  --from-literal=azure-tenant-id=<tenant-id> \
  -n neurondb
```

2. Configure values:
```yaml
backup:
  enabled: true
  azure:
    enabled: true
    accountName: neurondbstorage
    containerName: backups
    prefix: backups
```

## Manual Backup

### Using pg_dump (CNPG or any deployment)

```bash
# Get primary pod name (CNPG)
POD=$(kubectl get pod -n neurondb -l cnpg.io/instanceRole=primary -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n neurondb "$POD" -c postgres -- \
  pg_dump -U neurondb -d neurondb -Fc -f /tmp/backup.dump
kubectl cp neurondb/"$POD":/tmp/backup.dump ./backup.dump -c postgres
```

For non-CNPG (StatefulSet) deployment:

```bash
kubectl exec -it statefulset/neurondb-neurondb -n neurondb -- \
  pg_dump -U neurondb -d neurondb -Fc -f /tmp/backup.dump
kubectl cp neurondb/neurondb-neurondb-0:/tmp/backup.dump ./backup.dump
```

### Using kubectl

```bash
# Backup PVC
kubectl get pvc -n neurondb
# Use your storage provider's backup tool (e.g., Velero, Restic)
```

## Restore Procedure

### Step 1: Stop Services

```bash
# Scale down any application deployments that write to the database
# (e.g. application pods that use NeuronDB). Skip if only PostgreSQL/NeuronDB is deployed.
kubectl scale deployment <your-app> --replicas=0 -n neurondb
```

### Step 2: Restore Database

**CNPG:** Use a new Cluster with `bootstrap.recovery` as described in [Point-in-time recovery (PITR) with CNPG](#point-in-time-recovery-pitr-with-cnpg) above.

**Legacy:** Enable restore in values:

```bash
helm upgrade neurondb ./helm/neurondb \
  --set backup.restore.enabled=true \
  --set backup.restore.backupFile=neurondb-backup-20240101-020000.dump \
  --set backup.restore.fromS3=true \
  -n neurondb
```

### Step 3: Monitor Restore

```bash
# Watch restore job
kubectl get jobs -n neurondb -w
kubectl logs -f job/neurondb-restore-<hash> -n neurondb
```

### Step 4: Verify Restore

```bash
# CNPG: use primary pod
POD=$(kubectl get pod -n neurondb -l cnpg.io/instanceRole=primary -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n neurondb "$POD" -c postgres -- \
  psql -U neurondb -d neurondb -c "SELECT neurondb.version();"

# Legacy StatefulSet:
# kubectl exec -it statefulset/neurondb-neurondb -n neurondb -- \
#   psql -U neurondb -d neurondb -c "SELECT neurondb.version();"
```

### Step 5: Restart Services

```bash
# Scale back up any application deployments that were scaled down in Step 1
kubectl scale deployment <your-app> --replicas=2 -n neurondb
```

## Point-in-Time Recovery

**CNPG:** See [Point-in-time recovery (PITR) with CNPG](#point-in-time-recovery-pitr-with-cnpg) above.

**Legacy / self-managed:** For point-in-time recovery, ensure WAL archiving is enabled:

```yaml
neurondb:
  postgresql:
    config: |
      wal_level = replica
      archive_mode = on
      archive_command = 'aws s3 cp %p s3://neurondb-wal/%f'
```

Restore to specific point:
```bash
# Use pg_basebackup and WAL replay
# See PostgreSQL documentation for PITR procedures
```

## Backup Verification

### Test Restore

Regularly test restore to verify backups:

```bash
# Create test namespace
kubectl create namespace neurondb-test

# Restore to test namespace
helm install neurondb-test ./helm/neurondb \
  --set backup.restore.enabled=true \
  --set backup.restore.backupFile=neurondb-backup-20240101-020000.dump \
  --set backup.restore.fromS3=true \
  -n neurondb-test
```

### Backup Integrity

```bash
# List backups
aws s3 ls s3://neurondb-backups/backups/

# Verify backup file
aws s3 cp s3://neurondb-backups/backups/neurondb-backup-20240101-020000.dump - | \
  pg_restore --list - | head -20
```

## Retention Policy

Backups older than retention period are automatically deleted:

```yaml
backup:
  retention: 30  # days
```

Manual cleanup:
```bash
# List old backups
aws s3 ls s3://neurondb-backups/backups/ | \
  awk '$1 < "'$(date -d '30 days ago' -u +%Y-%m-%d)'" {print $4}'

# Delete old backups
aws s3 rm s3://neurondb-backups/backups/ --recursive --exclude "*" \
  --include "neurondb-backup-2023*"
```

## Troubleshooting

### Backup Job Fails

```bash
# Check job logs
kubectl logs job/neurondb-backup-<hash> -n neurondb

# Common issues:
# - S3 credentials incorrect
# - Network connectivity
# - Database not accessible
```

### Restore Job Fails

```bash
# Check restore logs
kubectl logs job/neurondb-restore-<hash> -n neurondb

# Common issues:
# - Backup file not found
# - Database connection failed
# - Insufficient permissions
```

### Backup Too Large

```bash
# Compress backup
pg_dump -Fc -Z 9 -f backup.dump database

# Split large backups
split -b 1G backup.dump backup-part-
```

