# 🚢 NeuronDB Production Deployment Documentation

<div align="center">

**Complete production deployment guide for NeuronDB on Kubernetes**

[![Kubernetes](https://img.shields.io/badge/kubernetes-ready-blue)](kubernetes-helm.md)
[![Production](https://img.shields.io/badge/production-ready-brightgreen)](production-install.md)

</div>

---

## 📋 Quick Links

| Guide | Description | Difficulty |
|-------|-------------|------------|
| [Production Installation Guide](./production-install.md) | Complete production setup | ⭐⭐ Medium |
| [Backup and Restore Guide](./backup-restore.md) | Backup/restore procedures | ⭐ Easy |
| [Upgrade and Rollback Guide](./upgrade-rollback.md) | Upgrade procedures | ⭐⭐ Medium |
| [Sizing Guide](./sizing-guide.md) | Resource sizing recommendations | ⭐ Easy |
| [Kubernetes/Helm Guide](./kubernetes-helm.md) | Kubernetes deployment | ⭐⭐⭐ Advanced |
| [Container Images](./container-images.md) | Container image information | ⭐ Easy |
| [HA Architecture](./ha-architecture.md) | High availability setup | ⭐⭐⭐ Advanced |

---

## ✨ Features

### 🔒 Security

- ✅ Per-component RBAC with minimal permissions
- ✅ NetworkPolicies with default deny
- ✅ Pod Security Standards enforcement
- ✅ External Secrets Operator integration
- ✅ CSI Secrets Store support

### 🔄 High Availability

- ✅ Zero-downtime upgrades with migration hooks
- ✅ StatefulSet rolling updates
- ✅ Pod Disruption Budgets
- ✅ PriorityClasses for critical components
- ✅ Health checks with SLO focus

### 📊 Observability

- ✅ ServiceMonitor for Prometheus Operator
- ✅ PrometheusRule alerts
- ✅ OpenTelemetry exporter config
- ✅ Structured logging

### ⚙️ Operations

- ✅ Automated backups (S3/GCS/Azure)
- ✅ Restore procedures
- ✅ Migration management
- ✅ External PostgreSQL support
- ✅ Advanced autoscaling (HPA/KEDA)

### 🔀 GitOps

- ✅ Argo CD examples
- ✅ Flux examples
- ✅ Declarative configuration

---

## 🚀 Quick Start

### Production Installation

<details>
<summary><strong>📦 Kubernetes Installation</strong></summary>

```bash
# 1. Create namespace
kubectl create namespace neurondb

# 2. Create external PostgreSQL secret (if using external DB)
kubectl create secret generic neurondb-external-postgres-secret \
  --from-literal=host=postgres.example.com \
  --from-literal=port=5432 \
  --from-literal=database=neurondb \
  --from-literal=username=neurondb \
  --from-literal=password=<secure-password> \
  -n neurondb

# 3. Install with production values
helm install neurondb ./helm/neurondb \
  -f helm/neurondb/values-production-external-postgres.yaml \
  -n neurondb
```

</details>

### Enable Production Features

<details>
<summary><strong>⚙️ Production Configuration</strong></summary>

```bash
helm upgrade neurondb ./helm/neurondb \
  --set rbac.enabled=true \
  --set networkPolicy.enabled=true \
  --set backup.enabled=true \
  --set backup.s3.enabled=true \
  --set backup.s3.bucket=neurondb-backups \
  --set backup.s3.region=us-east-1 \
  --set observability.prometheusOperator.enabled=true \
  -n neurondb
```

</details>

---

## 📁 Example Values Files

| File | Description | Use Case |
|------|-------------|----------|
| `values-minimal.yaml` | Minimal configuration | Development |
| `values-production-external-postgres.yaml` | Production with external PostgreSQL | Production |
| `values-observability-external.yaml` | With external observability stack | Production |
| `values-external-postgres.yaml` | External PostgreSQL example | Production |

---

## 🔄 CI/CD Integration

<details>
<summary><strong>🔄 CI/CD Features</strong></summary>

All CI workflows are configured:

- ✅ Image signing (cosign)
- ✅ SBOM generation (Syft)
- ✅ SLSA provenance
- ✅ Trivy security scanning
- ✅ Helm lint and unittest
- ✅ Chart testing
- ✅ OCI registry publishing

</details>

---

## 💬 Support

<details>
<summary><strong>📞 Get Help</strong></summary>

| Resource | Link |
|---------|------|
| **GitHub Issues** | [Report Issues](https://github.com/neurondb/neurondb2/issues) |
| **Documentation** | [https://docs.neurondb.ai](https://docs.neurondb.ai) |
| **Email Support** | support@neurondb.ai |

</details>

---

## 📚 Related Documentation

- **[Docker Deployment](./docker.md)** - Docker-based deployment
- **[Getting Started](../getting-started/README.md)** - Setup guides
- **[Components](../components/README.md)** - Component overviews

---

<div align="center">

[⬆ Back to Top](#-neurondb-production-deployment-documentation) · [📚 Main Documentation](../../documentation.md)

</div>
