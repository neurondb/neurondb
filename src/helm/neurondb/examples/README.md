# NeuronDB Helm Chart Examples

This directory contains example configuration files for various deployment scenarios.

## Available Examples

### values-dev.yaml
Development configuration optimized for local development and testing.

**Features:**
- Minimal resource requirements
- Single replica deployments
- Monitoring disabled
- Backup disabled
- Network policies disabled

**Usage:**
```bash
helm install neurondb ./helm/neurondb \
  --namespace neurondb-dev \
  --create-namespace \
  --values ./helm/neurondb/examples/values-dev.yaml \
  --set secrets.postgresPassword="dev-password"
```

### values-prod.yaml
Production-ready configuration with high availability and comprehensive monitoring.

**Features:**
- Multiple replicas for HA
- Full monitoring stack (Prometheus, Grafana, Jaeger)
- Automated backups to S3
- Network policies enabled
- Resource limits configured
- Ingress with TLS
- Advanced autoscaling

**Usage:**
```bash
helm install neurondb ./helm/neurondb \
  --namespace neurondb \
  --create-namespace \
  --values ./helm/neurondb/examples/values-prod.yaml \
  --set secrets.postgresPassword="$(openssl rand -base64 32)" \
  --set monitoring.grafana.adminPassword="secure-password"
```

### values-ha.yaml
High availability configuration optimized for multi-zone deployments.

**Features:**
- 5+ replicas for maximum availability
- Pod anti-affinity across zones
- Multi-zone storage
- Advanced backup retention
- High priority classes
- Comprehensive resource governance

**Usage:**
```bash
helm install neurondb ./helm/neurondb \
  --namespace neurondb \
  --create-namespace \
  --values ./helm/neurondb/examples/values-ha.yaml \
  --set secrets.postgresPassword="$(openssl rand -base64 32)"
```

### values-external-monitoring.yaml
Configuration for environments with existing monitoring infrastructure.

**Features:**
- Disables built-in monitoring stack
- Enables Prometheus Operator integration
- Configures ServiceMonitors for external Prometheus
- Network policies allow external monitoring access

**Usage:**
```bash
helm install neurondb ./helm/neurondb \
  --namespace neurondb \
  --create-namespace \
  --values ./helm/neurondb/examples/values-external-monitoring.yaml \
  --set observability.prometheusOperator.serviceMonitor.additionalLabels.release=prometheus
```

### values-multi-cloud.yaml
Multi-cloud deployment with cross-cloud backups.

**Features:**
- Backups to S3 (AWS), GCS (GCP), and Azure Blob Storage
- Multi-region pod anti-affinity
- Cross-cloud redundancy
- High availability across cloud providers

**Usage:**
```bash
helm install neurondb ./helm/neurondb \
  --namespace neurondb \
  --create-namespace \
  --values ./helm/neurondb/examples/values-multi-cloud.yaml \
  --set backup.s3.bucket="your-s3-bucket" \
  --set backup.gcs.bucket="your-gcs-bucket" \
  --set backup.azure.accountName="your-azure-account"
```

## Customizing Examples

All examples can be further customized by:

1. **Overriding specific values:**
   ```bash
   helm install neurondb ./helm/neurondb \
     --values ./helm/neurondb/examples/values-prod.yaml \
     --set neurondb.replicas=5 \
     --set neuronagent.autoscaling.maxReplicas=20
   ```

2. **Combining multiple value files:**
   ```bash
   helm install neurondb ./helm/neurondb \
     --values ./helm/neurondb/examples/values-prod.yaml \
     --values ./custom-overrides.yaml
   ```

3. **Environment-specific values:**
   ```bash
   # Create environment-specific override
   cat > staging-overrides.yaml <<EOF
   neurondb:
     persistence:
       size: 100Gi
   neuronagent:
     replicas: 3
   EOF
   
   helm install neurondb ./helm/neurondb \
     --values ./helm/neurondb/examples/values-prod.yaml \
     --values staging-overrides.yaml
   ```

## Best Practices

1. **Never commit secrets** - Use `--set` or external secret management
2. **Validate before deploying** - Run `./scripts/neurondb-helm.sh validate`
3. **Test in staging** - Always test configurations in non-production first
4. **Review resource limits** - Adjust based on actual workload
5. **Enable monitoring** - Always enable monitoring in production
6. **Use network policies** - Enable for production security
7. **Configure backups** - Set up automated backups before going live
8. **Review storage classes** - Ensure appropriate storage for your cluster

## Advanced Configuration

For advanced configurations, see:
- [values-external-postgres.yaml](../values-external-postgres.yaml) - External PostgreSQL
- [values-minimal.yaml](../values-minimal.yaml) - Minimal configuration
- [values-production-external-postgres.yaml](../values-production-external-postgres.yaml) - Production with external PostgreSQL

