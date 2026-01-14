# Production Installation Guide

<div align="center">

**Production-ready installation of NeuronDB on Kubernetes**

[![Production](https://img.shields.io/badge/production-ready-brightgreen)](.)
[![Kubernetes](https://img.shields.io/badge/kubernetes-1.24+-blue)](.)
[![Difficulty](https://img.shields.io/badge/difficulty-advanced-orange)](.)

</div>

---

> [!WARNING]
> This guide is for production deployments. Use strong passwords, enable TLS, and configure security properly. Do not use default credentials.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [External PostgreSQL Setup](#external-postgresql-setup)
3. [TLS Configuration](#tls-configuration)
4. [NetworkPolicy Configuration](#networkpolicy-configuration)
5. [Backup and Restore](#backup-and-restore)
6. [Upgrade and Rollback](#upgrade-and-rollback)
7. [Sizing Guidance](#sizing-guidance)

## Prerequisites

<details>
<summary><strong>📋 Prerequisites Checklist</strong></summary>

| Requirement | Minimum Version | Required |
|-------------|----------------|----------|
| **Kubernetes** | 1.24+ | ✅ Yes |
| **Helm** | 3.8+ | ✅ Yes |
| **kubectl** | Latest | ✅ Yes |
| **StorageClass** | - | ✅ Yes |
| **Prometheus Operator** | - | ❌ Optional |
| **External Secrets Operator** | - | ❌ Optional |

</details>

## External PostgreSQL Setup

> [!NOTE]
> Use external PostgreSQL for production. It provides better reliability, backups, and management.

### Option A: AWS RDS

<details>
<summary><strong>☁️ AWS RDS Setup</strong></summary>

1. Create RDS PostgreSQL instance:
```bash
aws rds create-db-instance \
  --db-instance-identifier neurondb-prod \
  --db-instance-class db.r5.xlarge \
  --engine postgres \
  --engine-version 17 \
  --master-username neurondb \
  --master-user-password <secure-password> \
  --allocated-storage 500 \
  --storage-type gp3 \
  --backup-retention-period 30
```

2. Create secret:
```bash
kubectl create secret generic neurondb-external-postgres-secret \
  --from-literal=host=neurondb-prod.xxxxx.us-east-1.rds.amazonaws.com \
  --from-literal=port=5432 \
  --from-literal=database=neurondb \
  --from-literal=username=neurondb \
  --from-literal=password=<secure-password> \
  -n neurondb
```

3. Install with external PostgreSQL:
```bash
helm install neurondb ./helm/neurondb \
  -f values-production-external-postgres.yaml \
  -n neurondb --create-namespace
```

### Option B: Google Cloud SQL

1. Create Cloud SQL instance via console or gcloud:
```bash
gcloud sql instances create neurondb-prod \
  --database-version=POSTGRES_17 \
  --tier=db-custom-4-16384 \
  --region=us-central1 \
  --backup-start-time=02:00
```

2. Create secret with Cloud SQL proxy connection:
```bash
kubectl create secret generic neurondb-external-postgres-secret \
  --from-literal=host=127.0.0.1 \
  --from-literal=port=5432 \
  --from-literal=database=neurondb \
  --from-literal=username=neurondb \
  --from-literal=password=<secure-password> \
  -n neurondb
```

</details>

### Option C: Azure Database for PostgreSQL

<details>
<summary><strong>☁️ Azure Database Setup</strong></summary>

1. Create Azure PostgreSQL Flexible Server:
```bash
az postgres flexible-server create \
  --resource-group neurondb-rg \
  --name neurondb-prod \
  --location eastus \
  --admin-user neurondb \
  --admin-password <secure-password> \
  --sku-name Standard_D4s_v3 \
  --storage-size 512
```

2. Create secret:
```bash
kubectl create secret generic neurondb-external-postgres-secret \
  --from-literal=host=neurondb-prod.postgres.database.azure.com \
  --from-literal=port=5432 \
  --from-literal=database=neurondb \
  --from-literal=username=neurondb \
  --from-literal=password=<secure-password> \
  -n neurondb
```

## TLS Configuration

### Ingress TLS

1. Create TLS secret:
```bash
kubectl create secret tls neurondb-tls \
  --cert=tls.crt \
  --key=tls.key \
  -n neurondb
```

2. Configure ingress in values:
```yaml
ingress:
  enabled: true
  className: "nginx"
  tls:
    - secretName: neurondb-tls
      hosts:
        - neurondb.example.com
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
```

### Database TLS

For external PostgreSQL, configure TLS in connection string:
```yaml
neurondb:
  postgresql:
    external:
      connectionString: "postgresql://user:pass@host:port/db?sslmode=require"
```

## NetworkPolicy Configuration

Enable NetworkPolicies for production:

```yaml
networkPolicy:
  enabled: true
  allowMonitoring: true
  monitoringNamespace: "monitoring"
  ingressNamespace: "ingress-nginx"
  allowedNamespaces: []  # Empty = same namespace only
```

Apply:
```bash
helm upgrade neurondb ./helm/neurondb \
  --set networkPolicy.enabled=true \
  -n neurondb
```

## Backup and Restore

### Configure Backup

1. Create backup credentials secret:
```bash
kubectl create secret generic neurondb-backup-credentials \
  --from-literal=aws-access-key-id=<key> \
  --from-literal=aws-secret-access-key=<secret> \
  -n neurondb
```

2. Enable backup in values:
```yaml
backup:
  enabled: true
  schedule: "0 2 * * *"  # Daily at 2 AM
  retention: 30
  s3:
    enabled: true
    bucket: neurondb-backups
    region: us-east-1
```

### Restore from Backup

1. Set restore configuration:
```yaml
backup:
  restore:
    enabled: true
    backupFile: "neurondb-backup-20240101-020000.dump"
    fromS3: true
```

2. Apply restore job:
```bash
helm upgrade neurondb ./helm/neurondb \
  --set backup.restore.enabled=true \
  --set backup.restore.backupFile=neurondb-backup-20240101-020000.dump \
  --set backup.restore.fromS3=true \
  -n neurondb
```

3. Monitor restore:
```bash
kubectl logs -f job/neurondb-restore-<hash> -n neurondb
```

## Upgrade and Rollback

### Upgrade Procedure

1. **Backup first:**
```bash
# Ensure backup is recent
kubectl get cronjob neurondb-backup -n neurondb
```

2. **Check migration status:**
```bash
kubectl get jobs -n neurondb | grep migration
```

3. **Upgrade chart:**
```bash
helm upgrade neurondb ./helm/neurondb \
  --version <new-version> \
  -f values-production-external-postgres.yaml \
  -n neurondb
```

4. **Monitor upgrade:**
```bash
kubectl get pods -n neurondb -w
kubectl logs -f deployment/neurondb-neuronagent -n neurondb
```

### Rollback Procedure

1. **Rollback Helm release:**
```bash
helm rollback neurondb -n neurondb
```

2. **If database migrations were applied, rollback them:**
```bash
# Connect to database and run rollback script
psql -h <db-host> -U neurondb -d neurondb -f rollback.sql
```

3. **Verify rollback:**
```bash
kubectl get pods -n neurondb
kubectl get svc -n neurondb
```

## Sizing Guidance

### Small Deployment (Development/Testing)

- **NeuronDB**: 2 CPU, 4Gi memory, 50Gi storage
- **NeuronAgent**: 1 replica, 500m CPU, 512Mi memory
- **NeuronMCP**: 1 replica, 250m CPU, 256Mi memory
- **Total**: ~6 CPU, ~6Gi memory

### Medium Deployment (Production - Small)

- **NeuronDB**: 4 CPU, 8Gi memory, 200Gi storage
- **NeuronAgent**: 2 replicas, 1 CPU, 1Gi memory each
- **NeuronMCP**: 1 replica, 500m CPU, 512Mi memory
- **Total**: ~7 CPU, ~11Gi memory

### Large Deployment (Production - Medium)

- **NeuronDB**: 8 CPU, 16Gi memory, 500Gi storage
- **NeuronAgent**: 3 replicas, 2 CPU, 2Gi memory each
- **NeuronMCP**: 2 replicas, 1 CPU, 1Gi memory each
- **Total**: ~20 CPU, ~26Gi memory

### Enterprise Deployment (Production - Large)

- **NeuronDB**: 16 CPU, 32Gi memory, 1Ti storage
- **NeuronAgent**: 5 replicas, 4 CPU, 4Gi memory each
- **NeuronMCP**: 3 replicas, 2 CPU, 2Gi memory each
- **Total**: ~46 CPU, ~58Gi memory

### Storage Sizing

- **Small**: 50-100Gi (development)
- **Medium**: 200-500Gi (production)
- **Large**: 500Gi-1Ti (enterprise)
- **Growth rate**: Plan for 2-3x current size for 1 year

### Network Bandwidth

- **Small**: 100 Mbps
- **Medium**: 1 Gbps
- **Large**: 10 Gbps

## Post-Installation Verification

1. Check all pods are running:
```bash
kubectl get pods -n neurondb
```

2. Verify services:
```bash
kubectl get svc -n neurondb
```

3. Test database connectivity:
```bash
kubectl exec -it deployment/neurondb-neuronagent -n neurondb -- \
  psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT version();"
```

4. Check health endpoints:
```bash
kubectl port-forward svc/neurondb-neuronagent 8080:8080 -n neurondb
curl http://localhost:8080/health
```

5. Verify NetworkPolicies:
```bash
kubectl get networkpolicies -n neurondb
```

6. Check backups:
```bash
kubectl get cronjob -n neurondb
kubectl get jobs -n neurondb | grep backup
```

---

## Troubleshooting

> [!TIP]
> Most production issues relate to configuration or resource limits. Check logs and resource usage first.

See [Troubleshooting Guide](../operations/troubleshooting.md) for common issues.

---

## 🔗 Related Documentation

| Document | Description |
|----------|-------------|
| **[Kubernetes/Helm Guide](kubernetes-helm.md)** | Kubernetes deployment |
| **[HA Architecture](ha-architecture.md)** | High availability setup |
| **[Backup and Restore](backup-restore.md)** | Backup procedures |
| **[Sizing Guide](sizing-guide.md)** | Resource sizing |

---

<div align="center">

[⬆ Back to Top](#production-installation-guide) · [📚 Deployment Index](README.md) · [📚 Main Documentation](../../README.md)

</div>

