# Argo CD Example

This directory contains an example Argo CD Application manifest for deploying NeuronDB.

## Prerequisites

- Argo CD installed in your cluster
- Access to the Helm chart repository

## Installation

1. Create the namespace:
```bash
kubectl create namespace neurondb
```

2. Create any required secrets (e.g., external PostgreSQL credentials):
```bash
kubectl create secret generic neurondb-external-postgres-secret \
  --from-literal=host=postgres.example.com \
  --from-literal=port=5432 \
  --from-literal=database=neurondb \
  --from-literal=username=neurondb \
  --from-literal=password=your-password \
  -n neurondb
```

3. Apply the Argo CD Application:
```bash
kubectl apply -f application.yaml
```

## Configuration

Edit `application.yaml` to customize:
- Source repository and path
- Values files to use
- Helm parameters
- Sync policy (automated vs manual)
- Destination namespace

## Sync Options

- `CreateNamespace=true`: Automatically create the namespace if it doesn't exist
- `PrunePropagationPolicy=foreground`: Wait for resources to be deleted
- `PruneLast=true`: Prune resources after sync

## Monitoring

Check application status:
```bash
argocd app get neurondb
```

View sync history:
```bash
argocd app history neurondb
```

