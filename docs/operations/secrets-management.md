# Secrets Management

NeuronDB supports external secrets for production deployments.

## Environment-based secrets (default)

All components read credentials from environment variables (e.g. `POSTGRES_PASSWORD`, `DB_PASSWORD`, `JWT_SECRET`, `NEURONDB_LLM_API_KEY`). **Never commit real secrets.** Use a `.env` file (gitignored) or your orchestrator's secret injection.

## HashiCorp Vault

To use Vault for secrets:

1. Store secrets in Vault (e.g. KV v2) and expose them as env vars via:
   - **Vault Agent** sidecar or init container that writes env files
   - **Kubernetes External Secrets Operator**: create `ExternalSecret` resources that sync Vault KV to Kubernetes `Secret`s; mount or envFrom into NeuronDB/NeuronAgent/NeuronDesktop/NeuronMCP pods
2. Example Vault path layout:
   - `neurondb/data/postgres` → `password`, `username`
   - `neurondb/data/neurondesktop` → `db_password`, `jwt_secret`
   - `neurondb/data/neuronagent` → `db_password`, `llm_api_key`
3. Set env vars from Vault (e.g. `POSTGRES_PASSWORD` from `vault kv get -field=password neurondb/data/postgres` in your entrypoint or use External Secrets to populate K8s Secrets and `envFrom`).

## Kubernetes External Secrets

1. Install [External Secrets Operator](https://external-secrets.io/).
2. Create a `SecretStore` or `ClusterSecretStore` pointing to your backend (Vault, AWS Secrets Manager, etc.).
3. Create `ExternalSecret` resources that map backend keys to Kubernetes `Secret` keys.
4. In Helm values, reference those secrets for `neurondb.existingSecret`, `neuronagent.envFrom`, etc., so the pods receive env vars from the generated Secrets.

## Encryption at rest for API keys

NeuronDesktop stores hashed API keys in the database. For additional encryption at rest:

- Use PostgreSQL TDE (transparent data encryption) or encrypted storage at the infrastructure layer.
- Optional: add an application-level encryption layer for sensitive columns using a key from Vault or KMS; this requires code changes to encrypt before insert and decrypt after read.

## DSN sanitization

All components MUST avoid logging connection strings that contain passwords. Use:

- **NeuronDesktop**: `cfg.Database.SanitizedDSN()` for any log message that might include DSN.
- **NeuronAgent**: `BuildMaskedConnectionString()` for logs.
- Never log `POSTGRES_PASSWORD`, `DB_PASSWORD`, or raw DSN.

## Secure session cookies

In production, set `SESSION_SECURE=true` (default) so session cookies are sent only over HTTPS. Set `SESSION_SAMESITE=Lax` or `Strict` to reduce CSRF risk.
