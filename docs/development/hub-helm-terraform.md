# Hub Helm Chart and Terraform

## Helm chart (neuron-hub repo)

A minimal Helm chart for **neuron-hub** is provided in the **neuron-hub** repo under `helm/`. It includes:

- **Backend**: Deployment and Service for the Hub Go API (port 8081).
- **Values**: `backend`, `gateway`, `frontend`, and `hubDb` sections in `values.yaml`.

When building images from the neuron-hub repo, push them to your registry and set in values:

```yaml
backend:
  image:
    repository: your-registry/neuron-hub-backend
    tag: "1.0.0"
```

Secrets (JWT_SECRET, DATABASE_URL, NEURONAGENT_ENDPOINT, etc.) should be provided via a Kubernetes Secret and wired into the backend/gateway deployments (e.g. `envFrom` or individual `valueFrom.secretKeyRef`).

### Install

From the **neuron-hub** repo root (or use the chart from that repo):

```bash
helm install neuron-hub ./helm --namespace neuron-hub --create-namespace
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
  name       = "neuron-hub"
  repository = "oci://your-registry/charts"  # or path to chart
  chart      = "neuron-hub"
  version    = "1.0.0-devel"
  namespace  = "neuron-hub"

  set_sensitive {
    name  = "hubDb.auth.password"
    value = var.hub_db_password
  }
  # Add more set/set_sensitive for secrets
}
```

2. If the chart is local (from neuron-hub repo), use `chart = "./helm"` (path relative to the Terraform module or neuron-hub clone).

3. For production, add a Kubernetes namespace resource and wire Hub to the same cluster as NeuronAgent so `NEURONAGENT_ENDPOINT` points to the NeuronAgent service.

This gives a single place (Terraform) to manage Hub deployment and upgrades when using the neuron-hub chart from the neuron-hub repo.
