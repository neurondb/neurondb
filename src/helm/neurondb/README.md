# NeuronDB Helm Chart

Complete cloud-native Helm chart for deploying the NeuronDB ecosystem on Kubernetes.

## Quick Start

```bash
# Create namespace
kubectl create namespace neurondb

# Install with default values
helm install neurondb ./helm/neurondb \
  --namespace neurondb \
  --create-namespace \
  --set secrets.postgresPassword="$(openssl rand -base64 32)"
```

## Components

This chart deploys:

- **NeuronDB**: PostgreSQL with NeuronDB extension via **CloudNativePG (CNPG)** Cluster (primary + optional standbys), with optional Pooler and ScheduledBackup
- **NeuronAgent**: AI agent service (Deployment with HPA)
- **NeuronMCP**: Model Context Protocol server (Deployment)
- **NeuronDesktop**: Web UI (API + Frontend Deployments)
- **Prometheus**: Metrics collection (optional)
- **Grafana**: Dashboards and visualization (optional)
- **Jaeger**: Distributed tracing (optional)

**Requirements:** Kubernetes 1.24+, Helm 3.8+, and the [CloudNativePG Operator](https://cloudnative-pg.io/) installed (e.g. in `cnpg-system`) when using in-cluster PostgreSQL. For local testing, use `./scripts/test-cnpg-local.sh` (creates a kind cluster, installs CNPG, deploys the chart).

## Configuration

See `values.yaml` for all configuration options.

### Key Values

```yaml
# Database (CNPG)
neurondb:
  enabled: true
  cnpg:
    instances: 2
    storage:
      size: 50Gi
      storageClass: ""
    pooler:
      enabled: true

# Services
neuronagent:
  replicas: 2
  autoscaling:
    enabled: true
    maxReplicas: 10

# Monitoring
monitoring:
  enabled: true
  prometheus:
    retention: "30d"
  grafana:
    adminPassword: "change-me"

# Ingress
ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: neurondb.example.com
```

## Requirements

- Kubernetes 1.24+
- Helm 3.8+
- **CloudNativePG Operator** (when using in-cluster PostgreSQL): install in `cnpg-system` before deploying this chart
- Storage class for persistent volumes
- Ingress controller (optional)

## Installation

See [Docs/deployment/kubernetes-helm.md](../../Docs/deployment/kubernetes-helm.md) for complete installation guide.

## Upgrading

```bash
helm upgrade neurondb ./helm/neurondb \
  --namespace neurondb \
  --values my-values.yaml
```

## Uninstalling

```bash
helm uninstall neurondb -n neurondb
```

**Warning**: This will delete all data in persistent volumes!

## Support

- Documentation: [Docs/deployment/kubernetes-helm.md](../../Docs/deployment/kubernetes-helm.md)
- Issues: https://github.com/neurondb/neurondb/issues

