# NeuronDB Cloud-Native Deployment

Complete cloud-native deployment guide for the NeuronDB ecosystem on Kubernetes.

## Overview

The NeuronDB Helm chart provides a production-ready, cloud-native deployment of the entire NeuronDB ecosystem including:

- **NeuronDB**: PostgreSQL with NeuronDB extension (StatefulSet)
- **NeuronAgent**: AI agent service (Deployment with HPA)
- **NeuronMCP**: Model Context Protocol server (Deployment)
- **NeuronDesktop**: Web UI (API + Frontend Deployments)
- **Observability**: Prometheus, Grafana, and Jaeger

## Quick Start

```bash
# 1. Create namespace
kubectl create namespace neurondb

# 2. Generate secure password
POSTGRES_PASSWORD=$(openssl rand -base64 32)

# 3. Install chart
helm install neurondb ./helm/neurondb \
  --namespace neurondb \
  --create-namespace \
  --set secrets.postgresPassword="$POSTGRES_PASSWORD"

# 4. Wait for pods
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=neurondb -n neurondb --timeout=300s

# 5. Access services
kubectl port-forward svc/neurondb-neurondesktop-frontend 3000:3000 -n neurondb
```

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────┐
│                    Ingress (Optional)                    │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
│   Frontend   │  │     API      │  │    Agent     │
│  (Next.js)   │  │   (Go API)   │  │   (Go API)   │
└───────┬──────┘  └───────┬──────┘  └───────┬──────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                  ┌───────▼──────┐
                  │   NeuronDB    │
                  │ (PostgreSQL)  │
                  └───────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
│  Prometheus  │  │   Grafana    │  │    Jaeger    │
│  (Metrics)   │  │ (Dashboards)  │  │  (Tracing)   │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Resource Types

- **StatefulSet**: NeuronDB (with persistent storage)
- **Deployments**: NeuronAgent, NeuronMCP, NeuronDesktop API/Frontend
- **Services**: ClusterIP for internal communication
- **Ingress**: Optional external access
- **ConfigMaps**: Configuration files
- **Secrets**: Passwords and API keys
- **PVCs**: Persistent storage for database and monitoring
- **HPA**: Auto-scaling for NeuronAgent
- **PDB**: Pod Disruption Budgets for HA
- **NetworkPolicy**: Optional network security

## Features

### High Availability

- **Pod Disruption Budgets**: Ensure minimum availability during updates
- **Multiple Replicas**: Configurable for all services
- **Health Checks**: Liveness and readiness probes
- **Init Containers**: Wait for database before starting services

### Scalability

- **Horizontal Pod Autoscaling**: Automatic scaling for NeuronAgent
- **Resource Limits**: CPU and memory constraints
- **Configurable Replicas**: Scale services independently

### Security

- **ServiceAccounts**: RBAC support
- **Network Policies**: Optional network isolation
- **Secrets Management**: Secure credential storage
- **Non-root Containers**: Security contexts
- **TLS Support**: Ingress with certificate management

### Observability

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Pre-configured dashboards
- **Jaeger**: Distributed tracing
- **Health Endpoints**: Built-in health checks

### Storage

- **Persistent Volumes**: Database and monitoring data
- **Storage Classes**: Configurable storage backends
- **Volume Claims**: Dynamic provisioning

## Configuration

### Key Values

```yaml
# Database
neurondb:
  persistence:
    size: 50Gi
    storageClass: "standard"

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
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: neurondb-tls
      hosts:
        - neurondb.example.com
```

### Environment Variables

All services are configured via environment variables that match the docker-compose setup:

- **NeuronAgent**: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `SERVER_PORT`, `LOG_LEVEL`
- **NeuronMCP**: `NEURONDB_HOST`, `NEURONDB_PORT`, `NEURONDB_DATABASE`, `NEURONDB_USER`, `NEURONDB_PASSWORD`
- **NeuronDesktop API**: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `SERVER_PORT`
- **NeuronDesktop Frontend**: `NEXT_PUBLIC_API_URL`

## Deployment

### Production Deployment

```bash
# 1. Create values file
cat > production-values.yaml <<EOF
neurondb:
  persistence:
    size: 100Gi
    storageClass: "fast-ssd"
  resources:
    limits:
      memory: "16Gi"
      cpu: "8"

neuronagent:
  replicas: 3
  autoscaling:
    enabled: true
    maxReplicas: 20

monitoring:
  enabled: true
  prometheus:
    retention: "90d"

ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: neurondb.example.com
  tls:
    - secretName: neurondb-tls
      hosts:
        - neurondb.example.com
EOF

# 2. Install
helm install neurondb ./helm/neurondb \
  --namespace neurondb \
  --create-namespace \
  --values production-values.yaml \
  --set secrets.postgresPassword="$(openssl rand -base64 32)"
```

### Development Deployment

```bash
helm install neurondb ./helm/neurondb \
  --namespace neurondb-dev \
  --create-namespace \
  --set neurondb.persistence.size=10Gi \
  --set neuronagent.replicas=1 \
  --set monitoring.enabled=false \
  --set secrets.postgresPassword="dev-password"
```

## Operations

### Upgrading

```bash
# Upgrade to new version
helm upgrade neurondb ./helm/neurondb \
  --namespace neurondb \
  --values my-values.yaml

# Rollback if needed
helm rollback neurondb -n neurondb
```

### Scaling

```bash
# Scale NeuronAgent
kubectl scale deployment neurondb-neuronagent --replicas=5 -n neurondb

# Or update via Helm
helm upgrade neurondb ./helm/neurondb \
  --namespace neurondb \
  --set neuronagent.replicas=5
```

### Monitoring

```bash
# Port-forward to Prometheus
kubectl port-forward svc/neurondb-prometheus 9090:9090 -n neurondb

# Port-forward to Grafana
kubectl port-forward svc/neurondb-grafana 3000:3000 -n neurondb

# View logs
kubectl logs -f deployment/neurondb-neuronagent -n neurondb
```

### Troubleshooting

```bash
# Check pod status
kubectl get pods -n neurondb

# Describe pod
kubectl describe pod <pod-name> -n neurondb

# Check events
kubectl get events -n neurondb --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> -n neurondb

# Validate chart (comprehensive validation)
./scripts/neurondb-helm.sh validate

# Test chart installation (requires Kubernetes cluster)
./scripts/neurondb-helm.sh test
```

## Validation and Testing

### Chart Validation

The Helm chart includes comprehensive validation and testing tools:

```bash
# Comprehensive validation (schema, templates, references, security)
./scripts/neurondb-helm.sh validate

# End-to-end installation testing (requires Kubernetes cluster)
./scripts/neurondb-helm.sh test

# Release automation
./scripts/neurondb-helm.sh release --version-type [patch|minor|major]
```

### Example Configurations

Pre-configured example values files are available in `helm/neurondb/examples/`:

- `values-dev.yaml` - Development configuration with minimal resources
- `values-prod.yaml` - Production configuration with HA and monitoring
- `values-ha.yaml` - High availability multi-zone configuration
- `values-external-postgres.yaml` - External PostgreSQL configuration
- `values-external-monitoring.yaml` - External monitoring stack

Example usage:

```bash
# Development
helm install neurondb ./helm/neurondb \
  --values ./helm/neurondb/examples/values-dev.yaml \
  --namespace neurondb \
  --create-namespace

# Production
helm install neurondb ./helm/neurondb \
  --values ./helm/neurondb/examples/values-prod.yaml \
  --namespace neurondb \
  --create-namespace \
  --set secrets.postgresPassword="$(openssl rand -base64 32)"
```

## Best Practices

1. **Use Secrets**: Never hardcode passwords in values.yaml. Use Kubernetes secrets or external secret management.
2. **Enable Monitoring**: Always enable monitoring in production. Use Prometheus Operator integration for advanced monitoring.
3. **Set Resource Limits**: Configure appropriate CPU/memory limits based on your workload. See example configurations.
4. **Use Storage Classes**: Configure appropriate storage for your cluster (fast-ssd for production).
5. **Enable Network Policies**: For production security, enable network policies to restrict traffic.
6. **Use Ingress with TLS**: For external access, use Ingress with TLS certificates (cert-manager recommended).
7. **Backup Regularly**: Configure automated backups to S3/GCS/Azure with appropriate retention policies.
8. **Monitor Resources**: Watch CPU, memory, and disk usage. Set up alerting via Prometheus.
9. **Use Priority Classes**: Enable resource governance priority classes for critical workloads.
10. **Validate Before Deploy**: Always run validation scripts before deploying to production.
11. **Test Upgrades**: Test chart upgrades in a staging environment before production.
12. **Enable Pod Disruption Budgets**: Configure PDBs to ensure availability during updates.

## Documentation

- [Kubernetes Helm Guide](docs/deployment/kubernetes-helm.md) - Detailed deployment guide
- [Helm Chart README](helm/neurondb/README.md) - Chart-specific documentation

## Support

For issues and questions:
- Check [Troubleshooting Guide](docs/deployment/kubernetes-helm.md#troubleshooting)
- Review [Helm Chart README](helm/neurondb/README.md)
- Open an issue on GitHub

