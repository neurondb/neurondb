# Secrets Management

NeuronDB (PostgreSQL extension) and the database it runs in can use external secrets for production.

## Environment-based secrets (default)

PostgreSQL and any clients read credentials from environment variables (e.g. `POSTGRES_PASSWORD`, `PGPASSWORD`). Do not commit real secrets. Use a `.env` file (gitignored) or your orchestrator’s secret injection.

## HashiCorp Vault

1. Store database credentials in Vault (e.g. KV v2), for example:
   - `neurondb/data/postgres` → `password`, `username`
2. Expose them as env vars via:
   - **Vault Agent** sidecar or init container that writes env files, or
   - **Kubernetes External Secrets Operator**: create `ExternalSecret` resources that sync Vault KV to Kubernetes `Secret`s; mount or `envFrom` into the PostgreSQL/NeuronDB pod.
3. Use those env vars when starting PostgreSQL or when running `psql`/application clients (e.g. `POSTGRES_PASSWORD` from Vault).

## Kubernetes External Secrets

1. Install [External Secrets Operator](https://external-secrets.io/).
2. Create a `SecretStore` or `ClusterSecretStore` pointing to your backend (Vault, AWS Secrets Manager, etc.).
3. Create `ExternalSecret` resources that map backend keys to Kubernetes `Secret` keys.
4. Reference those secrets in your deployment (e.g. `envFrom` or volume mount) so the PostgreSQL/NeuronDB pod receives the required env vars.

## Encryption at rest

Use PostgreSQL TDE (transparent data encryption) or encrypted storage at the infrastructure layer for data at rest.

## DSN sanitization

Avoid logging connection strings that contain passwords. Do not log `POSTGRES_PASSWORD`, `PGPASSWORD`, or raw DSNs.

---

[Operations](README.md) · [Documentation](../readme.md)
