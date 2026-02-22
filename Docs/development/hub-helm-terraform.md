# Hub Helm Chart and Terraform

## Helm chart (in-repo)

A minimal Helm chart for **neurondb-hub** is provided in this repo under `helm/neurondb-hub/`. It includes:

- **Backend**: Deployment and Service for the Hub Go API (port 8081).
- **Values**: `backend`, `gateway`, `frontend`, and `hubDb` sections in `values.yaml`.

When building images from the neurondb-hub repo, push them to your registry and set in values:

```yaml
backend:
  image:
    repository: your-registry/neurondb-hub-backend
    tag: "1.0.0"
```

Secrets (JWT_SECRET, DATABASE_URL, NEURONAGENT_ENDPOINT, etc.) should be provided via a Kubernetes Secret and wired into the backend/gateway deployments (e.g. `envFrom` or individual `valueFrom.secretKeyRef`).

### Install

From the neurondb repo root:

```bash
helm install neurondb-hub ./helm/neurondb-hub --namespace neurondb-hub --create-namespace
```

To add gateway and frontend deployments, extend the chart using the same pattern as `deployment-backend.yaml` and `service-backend.yaml`.

## Terraform

To provision Hub with Terraform using the Helm chart:

1. Use the [Helm provider](https://registry.terraform.io/providers/hashicorp/helm/latest/docs):

```hcl
terraform {
  required_providers {
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
}

resource "helm_release" "neurondb_hub" {
  name       = "neurondb-hub"
  repository = "oci://your-registry/charts"  # or path to chart
  chart      = "neurondb-hub"
  version    = "1.0.0-devel"
  namespace  = "neurondb-hub"

  set_sensitive {
    name  = "hubDb.auth.password"
    value = var.hub_db_password
  }
  # Add more set/set_sensitive for secrets
}
```

2. If the chart is local, use `chart = "./helm/neurondb-hub"` (path relative to the Terraform module).

3. For production, add a Kubernetes namespace resource and wire Hub to the same cluster as NeuronAgent so `NEURONAGENT_ENDPOINT` points to the NeuronAgent service.

This gives a single place (Terraform) to manage Hub deployment and upgrades when the neurondb-hub chart is used from this repo or copied into the Hub repo.
