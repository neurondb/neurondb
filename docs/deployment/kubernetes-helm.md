# Kubernetes Helm Deployment Guide

This guide covers deploying the **NeuronDB** (PostgreSQL + extension) on Kubernetes with Helm. The chart may include only the NeuronDB/PostgreSQL component; other services are not documented in this repository.

---

## 📑 Table of Contents

| Section | Description |
|---------|-------------|
| [Prerequisites](#prerequisites) | Required tools and cluster setup |
| [Quick Start](#quick-start) | Fast deployment steps |
| [CloudNativePG (CNPG)](#cloudnativepg-cnpg) | PostgreSQL via CNPG Operator (recommended) |
| [Local Testing (kind)](#local-testing-with-kind) | Full setup and test with kind |
| [Installation](#installation) | Detailed installation guide |
| [Configuration](#configuration) | Configuration options |
| [Upgrading](#upgrading) | Upgrade procedures |
| [Accessing Services](#accessing-services) | Service access methods |
| [Monitoring](#monitoring) | Monitoring setup |
| [Troubleshooting](#troubleshooting) | Common issues and solutions |

## Prerequisites

<details>
<summary><strong>Prerequisites Checklist</strong></summary>

| Requirement | Minimum Version | Description |
|-------------|----------------|-------------|
| **Kubernetes** | 1.24+ | Kubernetes cluster |
| **Helm** | 3.8+ | Helm package manager |
| **kubectl** | Latest | Kubernetes CLI tool |
| **CloudNativePG Operator** | 0.27+ | Required for in-cluster PostgreSQL (see [CNPG section](#cloudnativepg-cnpg)) |
| **Storage Class** | - | For persistent volumes |
| **Ingress Controller** | - | Optional, for external access |

</details>

## Features

The Helm chart can deploy NeuronDB (PostgreSQL with the extension) using **CloudNativePG (CNPG)** when not using an external database.

<details>
<summary><strong>Helm chart features (NeuronDB / PostgreSQL)</strong></summary>

| Feature | Description | Status |
|---------|-------------|--------|
| **CloudNativePG Cluster** | PostgreSQL managed by CNPG (primary + standbys) | Included |
| **CNPG Pooler** | PgBouncer connection pooling (Pooler CRD) | Optional |
| **CNPG ScheduledBackup** | Automated backups (Barman object store or volume snapshot) | Optional |
| **CNPG Monitoring** | Prometheus queries ConfigMap, PodMonitor | Included |
| **Resource limits** | CPU and memory limits | Included |
| **Health checks** | Liveness and readiness probes | Included |
| **ConfigMaps / Secrets** | Configuration and credentials | Included |
| **Network Policies** | Optional network security | Optional |

</details>

- **CloudNativePG Cluster** for NeuronDB PostgreSQL: automatic failover, replication, `-rw` / `-ro` / `-r` services
- **CNPG Pooler** (PgBouncer) for connection pooling when enabled
- **CNPG ScheduledBackup** for WAL archiving and full backups
- **Health checks** and **resource limits** on the database pods
- **ConfigMaps** and **Secrets** for configuration and PostgreSQL credentials
- **Network Policies** (optional) for network security

### Verify Prerequisites

```bash
# Check Kubernetes version
kubectl version --client --short

# Check Helm version
helm version

# Check cluster access
kubectl cluster-info

# List available storage classes
kubectl get storageclass
```

## Quick Start

### 1. Create Namespace

```bash
kubectl create namespace neurondb
```

### 2. Set PostgreSQL Password

Create a secret with your PostgreSQL password:

```bash
# Generate a secure password
POSTGRES_PASSWORD=$(openssl rand -base64 32)

# Create secret
kubectl create secret generic neurondb-secrets \
  --from-literal=postgres-password="$POSTGRES_PASSWORD" \
  --namespace=neurondb
```

Or set it in values.yaml:

```yaml
secrets:
  create: true
  postgresPassword: "your-secure-password-here"
```

### 3. Install with Helm

```bash
# Install from local chart
helm install neurondb ./helm/neurondb \
  --namespace neurondb \
  --create-namespace \
  --set secrets.postgresPassword="$POSTGRES_PASSWORD"
```

### 4. Verify Installation

```bash
# Check all pods are running
kubectl get pods -n neurondb

# Check services (CNPG creates -rw, -ro, -r)
kubectl get svc -n neurondb

# If using CNPG: check cluster status (requires kubectl-cnpg plugin)
kubectl cnpg status neurondb-neurondb -n neurondb

# Check persistent volumes
kubectl get pvc -n neurondb

# Run Helm test (if available)
helm test neurondb -n neurondb

# Validate chart (comprehensive validation)
./scripts/neurondb-helm.sh validate

# Test chart installation end-to-end (requires Kubernetes cluster or kind)
./scripts/neurondb-helm.sh test
```

## CloudNativePG (CNPG)

When not using external PostgreSQL, the chart deploys PostgreSQL via **CloudNativePG (CNPG)**. The CNPG Operator must be installed in the cluster first.

### Install CNPG Operator

```bash
helm repo add cloudnative-pg https://cloudnative-pg.github.io/charts
helm repo update
helm install cnpg-operator cloudnative-pg/cloudnative-pg \
  --namespace cnpg-system \
  --create-namespace
```

### What the Chart Deploys (CNPG)

| Resource | Description |
|----------|-------------|
| **Cluster** | CNPG Cluster CRD: primary + optional standbys, replication slots, WAL, storage |
| **Services** | `<release>-neurondb-rw` (primary), `-ro` (read-only), `-r` (any replica) |
| **Pooler** | Optional PgBouncer Pooler CRD for connection pooling |
| **ScheduledBackup** | Optional cron-based backups to S3/GCS/Azure or volume snapshot |
| **ConfigMap** | Custom Prometheus queries for Postgres metrics |

### Key CNPG Values

```yaml
neurondb:
  enabled: true
  cnpg:
    instances: 2                    # 1 = primary only; 2+ = HA
    storage:
      size: "50Gi"
      storageClass: ""
    walStorage:
      enabled: false
      size: "10Gi"
    pooler:
      enabled: true
      type: "rw"
      instances: 2
      pgbouncer:
        poolMode: "session"
        defaultPoolSize: "25"
        maxClientConn: "1000"
    backup:
      enabled: false
      barmanObjectStore:
        destinationPath: "s3://bucket/path"
        # s3Credentials, googleCredentials, or azureCredentials
      retentionPolicy: "7d"
    scheduledBackup:
      enabled: false
      schedule: "0 0 2 * * *"
    monitoring:
      enabled: true
    primaryUpdateStrategy: "unsupervised"
    minSyncReplicas: 0
    maxSyncReplicas: 0
```

PostgreSQL parameters (e.g. `shared_buffers`, `max_connections`) are set under `neurondb.cnpg.postgresql.parameters`. Do not set `shared_preload_libraries` in parameters; the CNPG operator manages it. Use a NeuronDB image that already loads the extension.

### External PostgreSQL

To use an existing PostgreSQL instance instead of CNPG:

```yaml
neurondb:
  enabled: true
  postgresql:
    external:
      enabled: true
      host: "my-postgres.example.com"
      port: 5432
      database: "neurondb"
      username: "neurondb"
      # Or use connectionString / secretName
```

When `external.enabled` is true, no Cluster, Pooler, or ScheduledBackup resources are created.

## Local Testing with kind

A full end-to-end test (kind cluster + CNPG operator + NeuronDB chart) can be run locally.

**Prerequisites:** [kind](https://kind.sigs.k8s.io/), kubectl, Helm, Docker

```bash
# From the repository root
./scripts/test-cnpg-local.sh
```

This script creates a 3-node kind cluster, installs the CNPG operator, deploys the chart with `helm/neurondb/examples/values-cnpg-test.yaml`, waits for Cluster and Pooler to be ready, verifies `-rw`/`-ro`/`-r` services, then deletes the cluster. Use `--keep` to leave the cluster for inspection; use `--destroy` to remove an existing test cluster only.

## Installation

### Basic Installation

```bash
helm install neurondb ./helm/neurondb \
  --namespace neurondb \
  --create-namespace
```

### Installation with Custom Values

Create a custom values file:

```yaml
# my-values.yaml (NeuronDB / PostgreSQL only)
neurondb:
  cnpg:
    instances: 2
    storage:
      size: 100Gi
      storageClass: "fast-ssd"

monitoring:
  enabled: true
  grafana:
    adminPassword: "secure-password"
```

Install with custom values:

```bash
helm install neurondb ./helm/neurondb \
  --namespace neurondb \
  --create-namespace \
  --values my-values.yaml
```

### Installation with Ingress

```yaml
# ingress-values.yaml
ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: neurondb.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: neurondb-tls
      hosts:
        - neurondb.yourdomain.com
```

```bash
helm install neurondb ./helm/neurondb \
  --namespace neurondb \
  --create-namespace \
  --values ingress-values.yaml
```

## Configuration

### Key Configuration Options

#### NeuronDB (PostgreSQL)

When using CNPG (default, in-cluster PostgreSQL):

```yaml
neurondb:
  enabled: true
  image:
    registry: ghcr.io
    repository: neurondb/neurondb-cuda
    tag: "latest"  # or e.g. "3.1.0" / "pg18-3.1.0" — match your release; requires GPU nodes for CUDA
  postgresql:
    database: "neurondb"
    username: "neurondb"
    port: 5432
  cnpg:
    instances: 2
    storage:
      size: "50Gi"
      storageClass: ""
    pooler:
      enabled: true
  resources:
    requests:
      memory: "4Gi"
      cpu: "2"
    limits:
      memory: "8Gi"
      cpu: "4"
```

When using external PostgreSQL, set `neurondb.postgresql.external.enabled: true` and configure host, database, and credentials.

#### Monitoring

```yaml
monitoring:
  enabled: true
  prometheus:
    enabled: true
    retention: "30d"
    persistence:
      enabled: true
      size: "20Gi"
  grafana:
    enabled: true
    adminPassword: "admin"  # Change in production!
    persistence:
      enabled: true
      size: "10Gi"
  jaeger:
    enabled: true
```

## Upgrading

### Upgrade to New Version

```bash
# Update values if needed
helm upgrade neurondb ./helm/neurondb \
  --namespace neurondb \
  --values my-values.yaml \
  --set neurondb.image.tag="3.1.0"  # or pg18-3.1.0 / latest — see container-images.md
```

### Rolling Back

```bash
# List releases
helm list -n neurondb

# Rollback to previous version
helm rollback neurondb -n neurondb

# Rollback to specific revision
helm rollback neurondb 2 -n neurondb
```

## Accessing Services

### Port Forwarding

#### PostgreSQL (NeuronDB)

```bash
# Forward the read-write service (primary)
kubectl port-forward -n neurondb svc/neurondb-neurondb-rw 5432:5432
# Or the generic service name if your chart exposes it:
# kubectl port-forward -n neurondb svc/neurondb-neurondb 5432:5432
```

Connect with `psql -h localhost -p 5432 -U neurondb -d neurondb` and run `SELECT neurondb.version();`.

#### Grafana

```bash
kubectl port-forward -n neurondb \
  svc/neurondb-grafana 3001:3000
```

Access at: http://localhost:3001 (admin/admin)

#### Prometheus

```bash
kubectl port-forward -n neurondb \
  svc/neurondb-prometheus 9090:9090
```

Access at: http://localhost:9090

#### Jaeger

```bash
kubectl port-forward -n neurondb \
  svc/neurondb-jaeger 16686:16686
```

Access at: http://localhost:16686

### Using Ingress

If ingress is enabled, access services via the configured host:

```bash
# Get ingress address
kubectl get ingress -n neurondb

# Access via hostname
curl https://neurondb.yourdomain.com
```

## Monitoring

### Prometheus Metrics

PostgreSQL and the NeuronDB extension can be monitored via:
- **Postgres Exporter** (if enabled): standard PostgreSQL metrics
- **CNPG**: cluster and instance metrics when using CloudNativePG
- Extension version and GPU status via SQL: `SELECT neurondb.version();`, `SELECT neurondb.gpu_enabled();`

### Grafana Dashboards

Grafana is pre-configured with:

- Prometheus datasource
- Default dashboard provisioning

Access Grafana and create custom dashboards for:
- Service health
- Request rates and latencies
- Database connection metrics
- Resource utilization

### Jaeger Tracing

Jaeger is available for distributed tracing:

- UI: Port 16686
- OTLP gRPC: Port 4317
- OTLP HTTP: Port 4318

## Troubleshooting

### Pods Not Starting

#### Check Pod Status

```bash
kubectl get pods -n neurondb
kubectl describe pod <pod-name> -n neurondb
kubectl logs <pod-name> -n neurondb
```

#### Common Issues

**Pending Pods (Storage Issues)**

```bash
# Check PVC status
kubectl get pvc -n neurondb

# Check storage class
kubectl get storageclass

# If PVC is pending, check events
kubectl describe pvc <pvc-name> -n neurondb
```

**CrashLoopBackOff**

```bash
# Check logs
kubectl logs <pod-name> -n neurondb --previous

# Common causes:
# - Database connection failures
# - Missing secrets
# - Resource limits too low
```

### Database Connection Issues

#### Verify Database is Ready (CNPG)

```bash
# Check CNPG cluster status (install kubectl-cnpg plugin for full status)
kubectl get cluster -n neurondb
kubectl get pods -n neurondb -l cnpg.io/cluster=neurondb-neurondb

# Check primary pod logs
kubectl logs -n neurondb -l cnpg.io/instanceRole=primary -c postgres

# Test connection via -rw service
kubectl run -it --rm debug --image=postgres:17 --restart=Never -n neurondb -- \
  psql -h neurondb-neurondb-rw -U neurondb -d neurondb -c "SELECT version();"
```

#### Verify Service Connectivity

```bash
# Check service endpoints
kubectl get endpoints -n neurondb

# -rw = primary (read-write), -ro = read-only replicas, -r = any replica
kubectl get svc -n neurondb | grep neurondb
```

### Health Check Failures

#### Check Probe Configuration

```bash
# View pod spec
kubectl get pod <pod-name> -n neurondb -o yaml | grep -A 10 probes
```

#### Common Fixes

- Increase `initialDelaySeconds` if service takes time to start
- Verify health endpoint is accessible: `/health`
- Check resource limits aren't causing OOM kills

### Resource Issues

#### Check Resource Usage

```bash
# View resource requests/limits
kubectl describe pod <pod-name> -n neurondb | grep -A 5 "Limits\|Requests"

# Check node resources
kubectl top nodes
kubectl top pods -n neurondb
```

#### Adjust Resources

Update values and upgrade. For the NeuronDB (PostgreSQL) pods:

```yaml
neurondb:
  resources:
    requests:
      memory: "4Gi"
      cpu: "2"
    limits:
      memory: "8Gi"
      cpu: "4"
```

### Uninstalling

```bash
# Uninstall release
helm uninstall neurondb -n neurondb

# Delete namespace (removes all resources)
kubectl delete namespace neurondb

# WARNING: This deletes all data in PVCs!
# Backup data before uninstalling if needed
```

### Backup and Restore

When using CNPG, backups are managed by the chart via **CNPG backup** (Barman object store or volume snapshot) and optional **ScheduledBackup**. See **[Backup and Restore](backup-restore.md)** for:

- Enabling `neurondb.cnpg.backup` (S3/GCS/Azure) and `neurondb.cnpg.scheduledBackup`
- Point-in-time recovery (PITR) with `bootstrap.recovery`
- On-demand backup with `kubectl cnpg backup`

#### Manual Backup (any deployment)

```bash
# Get primary pod name (CNPG)
POD=$(kubectl get pod -n neurondb -l cnpg.io/instanceRole=primary -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n neurondb "$POD" -c postgres -- \
  pg_dump -U neurondb -d neurondb -Fc -f /tmp/backup.dump
kubectl cp neurondb/"$POD":/tmp/backup.dump ./backup.dump -c postgres
```

#### Restore

See [Backup and Restore](backup-restore.md) for restore procedures (CNPG recovery cluster or manual restore).

## Advanced Configuration

### Custom PostgreSQL Configuration

When using CNPG, configure PostgreSQL via `neurondb.cnpg.postgresql.parameters`:

```yaml
neurondb:
  cnpg:
    postgresql:
      parameters:
        shared_buffers: "256MB"
        max_connections: "200"
        effective_cache_size: "768MB"
        work_mem: "4MB"
# Do not set shared_preload_libraries; CNPG operator manages it.
```

### Service Account and RBAC

```yaml
serviceAccount:
  create: true
  name: "neurondb-sa"
  annotations:
    eks.amazonaws.com/role-arn: "arn:aws:iam::ACCOUNT:role/neurondb-role"
```

### Node Selectors and Affinity

Add to the NeuronDB (PostgreSQL) deployment or use chart values:

```yaml
neurondb:
  nodeSelector:
    workload-type: "database"
  affinity:
    podAntiAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchExpressions:
            - key: app.kubernetes.io/component
              operator: In
              values:
              - neurondb
          topologyKey: kubernetes.io/hostname
```

## Production Recommendations

> [!IMPORTANT]
> Follow these recommendations for production deployments.

<details>
<summary><strong>Production Checklist</strong></summary>

| Recommendation | Description | Priority |
|----------------|-------------|----------|
| **Secrets Management** | Use external secret management (AWS Secrets Manager, HashiCorp Vault) | Critical |
| **Backup Strategy** | Use CNPG backup (Barman/ScheduledBackup) or external backup tools | Critical |
| **Monitoring** | Enable full observability stack and set up alerting | High |
| **Resource Limits** | Set appropriate requests and limits based on workload | High |
| **High Availability** | Use multiple replicas and pod anti-affinity | High |
| **Storage** | Use fast, reliable storage class for production | High |
| **Ingress** | Enable TLS/SSL for external access | Critical |
| **Network Policies** | Implement network policies for security | Critical |

</details>

---

## Related Documentation

| Document | Description |
|----------|-------------|
| **[Production Installation](production-install.md)** | Production setup guide |
| **[HA Architecture](ha-architecture.md)** | High availability (CNPG + Patroni) |
| **[Backup and Restore](backup-restore.md)** | CNPG and legacy backup procedures |
| **[Troubleshooting](../operations/troubleshooting.md)** | Common issues |

---

<div align="center">

[Back to Top](#kubernetes-helm-deployment-guide) · [Deployment Index](README.md) · [Main Documentation](../../README.md)

</div>

