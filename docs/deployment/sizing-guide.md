# Sizing Guide

Comprehensive sizing guidance for NeuronDB deployments.

## Resource Profiles

### Small (Development/Testing)

**Use Case**: Development, testing, small teams

**Components**:
- NeuronDB: 1 replica, 2 CPU, 4Gi memory, 50Gi storage
- NeuronAgent: 1 replica, 500m CPU, 512Mi memory
- NeuronMCP: 1 replica, 250m CPU, 256Mi memory
- NeuronDesktop: Disabled

**Total Resources**:
- CPU: ~3 cores
- Memory: ~5Gi
- Storage: 50Gi

**Workload Capacity**:
- Concurrent users: 10-50
- Requests/second: 100-500
- Vector operations: 1K-10K/day

### Medium (Production - Small)

**Use Case**: Small production deployments, <100 users

**Components**:
- NeuronDB: 1 replica, 4 CPU, 8Gi memory, 200Gi storage
- NeuronAgent: 2 replicas, 1 CPU, 1Gi memory each
- NeuronMCP: 1 replica, 500m CPU, 512Mi memory
- NeuronDesktop: Enabled (2 replicas API, 2 replicas frontend)

**Total Resources**:
- CPU: ~7 cores
- Memory: ~11Gi
- Storage: 200Gi

**Workload Capacity**:
- Concurrent users: 50-200
- Requests/second: 500-2K
- Vector operations: 10K-100K/day

### Large (Production - Medium)

**Use Case**: Medium production deployments, 100-500 users

**Components**:
- NeuronDB: 1 replica, 8 CPU, 16Gi memory, 500Gi storage
- NeuronAgent: 3 replicas, 2 CPU, 2Gi memory each
- NeuronMCP: 2 replicas, 1 CPU, 1Gi memory each
- NeuronDesktop: Enabled (3 replicas API, 3 replicas frontend)

**Total Resources**:
- CPU: ~20 cores
- Memory: ~26Gi
- Storage: 500Gi

**Workload Capacity**:
- Concurrent users: 200-1000
- Requests/second: 2K-10K
- Vector operations: 100K-1M/day

### Enterprise (Production - Large)

**Use Case**: Large production deployments, 500+ users

**Components**:
- NeuronDB: 1 replica, 16 CPU, 32Gi memory, 1Ti storage
- NeuronAgent: 5 replicas, 4 CPU, 4Gi memory each
- NeuronMCP: 3 replicas, 2 CPU, 2Gi memory each
- NeuronDesktop: Enabled (5 replicas API, 5 replicas frontend)

**Total Resources**:
- CPU: ~46 cores
- Memory: ~58Gi
- Storage: 1Ti

**Workload Capacity**:
- Concurrent users: 1000+
- Requests/second: 10K+
- Vector operations: 1M+/day

## Storage Sizing

### Base Storage Requirements

- **OS and binaries**: ~10Gi
- **PostgreSQL data**: Variable
- **WAL files**: ~10% of data size
- **Logs**: ~5Gi per month
- **Backups**: 2-3x data size (if local)

### Growth Projections

Plan storage for 1 year growth:

- **Small**: 50Gi → 100-150Gi
- **Medium**: 200Gi → 400-600Gi
- **Large**: 500Gi → 1-1.5Ti
- **Enterprise**: 1Ti → 2-3Ti

### Storage Class Recommendations

- **Development**: Standard SSD
- **Production**: Premium SSD or GP3
- **Enterprise**: Premium SSD with IOPS optimization

## Network Bandwidth

### Estimated Requirements

- **Small**: 100 Mbps
- **Medium**: 1 Gbps
- **Large**: 10 Gbps
- **Enterprise**: 25 Gbps+

### Factors Affecting Bandwidth

- Vector embedding size
- Query frequency
- Replication (if enabled)
- Backup operations

## CPU Sizing

### NeuronDB CPU

- **Base**: 2 CPU for PostgreSQL
- **Vector operations**: +1 CPU per 10K vectors/second
- **Concurrent queries**: +0.5 CPU per 100 concurrent

### NeuronAgent CPU

- **Base**: 500m per replica
- **Request handling**: +100m per 100 req/s
- **Background workers**: +200m per worker

### NeuronMCP CPU

- **Base**: 250m per replica
- **MCP requests**: +50m per 50 req/s

## Memory Sizing

### NeuronDB Memory

- **Base**: 4Gi for PostgreSQL
- **Shared buffers**: 25% of total memory
- **Work memory**: 256Mi per connection
- **Vector cache**: 1Gi per 1M vectors

### NeuronAgent Memory

- **Base**: 512Mi per replica
- **Request buffer**: 100Mi per 100 req/s
- **Cache**: 200Mi per replica

### NeuronMCP Memory

- **Base**: 256Mi per replica
- **Request buffer**: 50Mi per 50 req/s

## Autoscaling Recommendations

### HPA Configuration

```yaml
neuronagent:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
```

### KEDA Configuration

```yaml
neuronagent:
  autoscaling:
    keda:
      enabled: true
      minReplicas: 2
      maxReplicas: 20
      triggers:
        http:
          enabled: true
          threshold: "100"
        queueDepth:
          enabled: true
          targetValue: "100"
```

## Node Requirements

### Minimum Node Specs

- **Small**: 4 CPU, 8Gi memory nodes
- **Medium**: 8 CPU, 16Gi memory nodes
- **Large**: 16 CPU, 32Gi memory nodes
- **Enterprise**: 32+ CPU, 64Gi+ memory nodes

### Node Affinity

Use node selectors for production:

```yaml
neurondb:
  nodeSelector:
    node-type: database
  affinity:
    podAntiAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          podAffinityTerm:
            labelSelector:
              matchExpressions:
                - key: app.kubernetes.io/component
                  operator: In
                  values: [neurondb]
            topologyKey: kubernetes.io/hostname
```

## Monitoring Resource Usage

### Key Metrics

- CPU utilization: <70% average
- Memory utilization: <80% average
- Disk I/O: Monitor p95 latency
- Network: Monitor bandwidth usage

### Alerts

Configure alerts for:
- CPU > 80% for 5 minutes
- Memory > 90% for 5 minutes
- Disk usage > 80%
- Pod restarts > 3 in 10 minutes

## Cost Optimization

### Right-Sizing Tips

1. Start with medium profile, scale based on metrics
2. Use HPA for dynamic scaling
3. Enable resource quotas
4. Use spot instances for non-critical workloads
5. Optimize storage class based on access patterns

### Resource Quotas

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: neurondb-quota
  namespace: neurondb
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 30Gi
    limits.cpu: "40"
    limits.memory: 60Gi
    persistentvolumeclaims: "5"
```

## Example Configurations

### values-small.yaml

```yaml
neurondb:
  resources:
    requests: {cpu: "2", memory: "4Gi"}
    limits: {cpu: "4", memory: "8Gi"}
  persistence:
    size: 50Gi

neuronagent:
  replicas: 1
  resources:
    requests: {cpu: "500m", memory: "512Mi"}
    limits: {cpu: "1", memory: "1Gi"}
```

### values-production.yaml

```yaml
neurondb:
  resources:
    requests: {cpu: "8", memory: "16Gi"}
    limits: {cpu: "16", memory: "32Gi"}
  persistence:
    size: 500Gi

neuronagent:
  replicas: 3
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10
  resources:
    requests: {cpu: "2", memory: "2Gi"}
    limits: {cpu: "4", memory: "4Gi"}
```

