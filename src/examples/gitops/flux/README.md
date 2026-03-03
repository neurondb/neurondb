# Flux Example

This directory contains an example Flux Kustomization for deploying NeuronDB.

## Prerequisites

- Flux v2 installed in your cluster
- Helm Controller enabled
- Source Controller enabled

## Installation

1. Create the HelmRepository:
```bash
kubectl apply -f helmrepository.yaml
```

2. Wait for repository to be ready:
```bash
flux get sources helm
```

3. Create the namespace:
```bash
kubectl apply -f namespace.yaml
```

4. Create any required secrets:
```bash
kubectl create secret generic neurondb-external-postgres-secret \
  --from-literal=host=postgres.example.com \
  --from-literal=port=5432 \
  --from-literal=database=neurondb \
  --from-literal=username=neurondb \
  --from-literal=password=your-password \
  -n neurondb
```

5. Apply the Kustomization:
```bash
kubectl apply -k .
```

## Configuration

Edit `helmrelease.yaml` to customize:
- Chart version constraints
- Values for the Helm chart
- Install/upgrade/rollback remediation

Edit `kustomization.yaml` to:
- Add patches for environment-specific overrides
- Include additional resources

## Monitoring

Check HelmRelease status:
```bash
flux get helmrelease neurondb -n neurondb
```

View reconciliation status:
```bash
flux get kustomizations
```

Suspend reconciliation:
```bash
flux suspend helmrelease neurondb -n neurondb
```

Resume reconciliation:
```bash
flux resume helmrelease neurondb -n neurondb
```

