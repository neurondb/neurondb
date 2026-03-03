# Identity Model Mapping

<div align="center">

**Component-to-unified identity model mapping**

[![Internals](https://img.shields.io/badge/internals-advanced-orange)](.)
[![Status](https://img.shields.io/badge/status-stable-brightgreen)](.)

</div>

---

> [!NOTE]
> This document maps existing component-specific identity models to the unified identity model. Use this for migration planning.

## NeuronDesktop → Unified Model

| NeuronDesktop | Unified Model | Notes |
|--------------|---------------|-------|
| `users.id` (UUID) | `identity.User.ID` | Direct mapping |
| `users.username` | `identity.User.Username` | Direct mapping |
| `users.is_admin` | `identity.User.IsAdmin` | Direct mapping |
| `profiles.id` | `identity.Project.ID` | Profile = Project |
| `profiles.user_id` (TEXT→UUID) | `identity.Project.UserID` | Needs migration |
| `profiles.name` | `identity.Project.Name` | Direct mapping |
| `profiles.mcp_config` | `identity.Project.MCPConfig` | Direct mapping |
| `profiles.neurondb_dsn` | `identity.Project.NeuronDBDSN` | Direct mapping |
| `api_keys.id` | `identity.APIKey.ID` | Direct mapping |
| `api_keys.key_prefix` | `identity.APIKey.KeyPrefix` | Direct mapping |
| `api_keys.user_id` (TEXT) | `identity.APIKey.PrincipalID` | Via principal lookup |
| `audit_log.user_id` | `identity.AuditLogEntry.PrincipalID` | Via principal lookup |

## NeuronAgent → Unified Model

| NeuronAgent | Unified Model | Notes |
|-------------|---------------|-------|
| `principals.id` | `identity.Principal.ID` | Direct mapping |
| `principals.type` | `identity.Principal.Type` | Direct mapping |
| `principals.name` | `identity.Principal.Name` | Direct mapping |
| `api_keys.principal_id` | `identity.APIKey.PrincipalID` | Direct mapping |
| `api_keys.organization_id` | `identity.APIKey.PrincipalID` | Via org principal |
| `api_keys.user_id` | `identity.APIKey.PrincipalID` | Via user principal |
| `policies.principal_id` | `identity.Policy.PrincipalID` | Direct mapping |
| `policies.resource_type` | `identity.Policy.ResourceType` | Direct mapping |
| `policies.permissions` | `identity.Policy.Permissions` | Direct mapping |
| `audit_log.principal_id` | `identity.AuditLogEntry.PrincipalID` | Direct mapping |

## NeuronMCP → Unified Model

| NeuronMCP | Unified Model | Notes |
|-----------|---------------|-------|
| `TenantContext.UserID` | `identity.TenantContext.UserID` | Direct mapping |
| `TenantContext.OrgID` | `identity.TenantContext.OrgID` | Direct mapping |
| `TenantContext.ProjectID` | `identity.TenantContext.ProjectID` | Direct mapping |
| OIDC subject | `identity.User.ID` | Via OIDC identity mapping |

## Migration Path

> [!WARNING]
> Test migrations in a development environment first. Back up databases before running migration scripts.

### Phase 1: Schema Alignment

<details>
<summary><strong>📋 Schema Migration Steps</strong></summary>

1. Run `008_unified_identity_model.sql` on NeuronDesktop database
2. Run `014_unified_identity_model.sql` on NeuronAgent database
3. Create principals for existing users and organizations
4. Link API keys to principals

</details>

### Phase 2: Code Updates

<details>
<summary><strong>💻 Code Migration Steps</strong></summary>

1. Update imports to use `pkg/identity`
2. Replace component-specific types with unified types
3. Update API key resolution logic
4. Update permission checks

</details>

### Phase 3: Service Integration

<details>
<summary><strong>🔗 Service Integration Steps</strong></summary>

1. Implement service account tokens
2. Update service-to-service authentication
3. Add cross-component identity resolution

</details>

## Data Migration Examples

### Migrate Desktop User to Principal
```sql
-- Create principal for existing user
INSERT INTO principals (type, name, metadata, created_at)
SELECT 
    'user',
    username,
    jsonb_build_object('user_id', id::text),
    created_at
FROM users
ON CONFLICT (type, name) DO NOTHING;

-- Link API keys to principals
UPDATE api_keys ak
SET 
    principal_id = p.id,
    principal_type = 'user'
FROM principals p
WHERE p.type = 'user' AND p.name = ak.user_id;
```

### Migrate Agent Organization to Principal
```sql
-- Create principal for existing organization
INSERT INTO principals (type, name, created_at)
SELECT 
    'org',
    organization_id,
    NOW()
FROM api_keys
WHERE organization_id IS NOT NULL
GROUP BY organization_id
ON CONFLICT (type, name) DO NOTHING;

-- Link API keys to principals
UPDATE neurondb_agent.api_keys ak
SET 
    principal_id = p.id,
    principal_type = 'org'
FROM neurondb_agent.principals p
WHERE p.type = 'org' AND p.name = ak.organization_id;
```

## Backward Compatibility

### NeuronDesktop
- Keep `profiles` table name (maps to Project internally)
- Keep `user_id` TEXT field during transition (add UUID column)
- Support both old and new API key formats during migration

### NeuronAgent
- Keep `principals` table (already aligned)
- Keep existing policy structure
- Support both old and new API key lookups

### NeuronMCP
- Keep `TenantContext` structure
- Add principal resolution layer
- Support both tenant-based and principal-based auth

## Validation Queries

### Check Migration Status
```sql
-- Desktop: Check principals created
SELECT COUNT(*) FROM principals WHERE type = 'user';

-- Desktop: Check API keys linked
SELECT COUNT(*) FROM api_keys WHERE principal_id IS NOT NULL;

-- Agent: Check principals aligned
SELECT type, COUNT(*) FROM neurondb_agent.principals GROUP BY type;

-- Agent: Check API keys with principal_type
SELECT principal_type, COUNT(*) 
FROM neurondb_agent.api_keys 
WHERE principal_type IS NOT NULL
GROUP BY principal_type;
```

### Verify Data Integrity

> [!TIP]
> Run these queries after migration to verify data integrity.

```sql
-- Check orphaned API keys
SELECT COUNT(*) 
FROM api_keys 
WHERE principal_id IS NOT NULL 
AND principal_id NOT IN (SELECT id FROM principals);

-- Check orphaned audit log entries
SELECT COUNT(*) 
FROM audit_log 
WHERE principal_id IS NOT NULL 
AND principal_id NOT IN (SELECT id FROM principals);
```

---

## 🔗 Related Documentation

| Document | Description |
|----------|-------------|
| **[Unified Identity Model](unified-identity-model.md)** | Unified identity architecture |
| **[Identity Integration Guide](identity-integration-guide.md)** | Integration procedures |
| **[OIDC Session Security](oidc-session-security.md)** | Security implementation |

---

<div align="center">

[⬆ Back to Top](#identity-model-mapping) · [📚 Internals Index](README.md) · [📚 Main Documentation](../../README.md)

</div>







