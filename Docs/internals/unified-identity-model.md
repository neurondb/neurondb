# Unified Identity Model Specification

## Overview

This document defines the unified identity, authentication, and authorization model used across all NeuronDB ecosystem components (NeuronDesktop, NeuronAgent, NeuronMCP).

## Core Concepts

### 1. Identity Entities

#### User
- **Type**: `user`
- **ID**: UUID (primary key)
- **Attributes**:
  - `username`: Unique username
  - `email`: Email address (optional, for OIDC)
  - `is_admin`: Boolean flag for admin privileges
  - `metadata`: JSONB for additional attributes
  - `created_at`, `updated_at`: Timestamps

#### Organization
- **Type**: `org`
- **ID**: UUID (primary key)
- **Attributes**:
  - `name`: Organization name
  - `slug`: URL-friendly identifier
  - `metadata`: JSONB for additional attributes
  - `created_at`, `updated_at`: Timestamps

#### Project (Profile)
- **Type**: `project` (maps to Desktop "Profile")
- **ID**: UUID (primary key)
- **Attributes**:
  - `name`: Project name
  - `org_id`: UUID reference to organization (nullable for user-level projects)
  - `user_id`: UUID reference to user (nullable for org-level projects)
  - `is_default`: Boolean flag
  - `neurondb_dsn`: Connection string
  - `mcp_config`: JSONB configuration
  - `agent_endpoint`: Optional agent endpoint
  - `agent_api_key`: Optional agent API key
  - `metadata`: JSONB for additional attributes
  - `created_at`, `updated_at`: Timestamps

#### Service Account
- **Type**: `service_account`
- **ID**: UUID (primary key)
- **Attributes**:
  - `name`: Service account name
  - `org_id`: UUID reference to organization (nullable)
  - `project_id`: UUID reference to project (nullable)
  - `metadata`: JSONB for additional attributes
  - `created_at`, `updated_at`: Timestamps

### 2. Authentication

#### API Keys
- **ID**: UUID (primary key)
- **Attributes**:
  - `key_hash`: Bcrypt hash of the API key
  - `key_prefix`: First 8 characters for identification
  - `principal_id`: UUID reference to principal (user, org, service_account)
  - `principal_type`: Type of principal ('user', 'org', 'service_account')
  - `project_id`: Optional UUID reference to project
  - `rate_limit_per_minute`: Integer rate limit
  - `roles`: Array of role strings
  - `expires_at`: Optional expiration timestamp
  - `last_used_at`: Timestamp of last use
  - `created_at`: Creation timestamp

#### OIDC Identities
- **ID**: UUID (primary key)
- **Attributes**:
  - `issuer`: OIDC issuer URL
  - `subject`: OIDC subject identifier
  - `user_id`: UUID reference to user
  - `email`: Email from OIDC claims
  - `name`: Name from OIDC claims
  - `created_at`, `updated_at`: Timestamps

#### Sessions
- **ID**: UUID (primary key)
- **Attributes**:
  - `user_id`: UUID reference to user
  - `created_at`: Creation timestamp
  - `last_seen_at`: Last activity timestamp
  - `revoked_at`: Optional revocation timestamp
  - `user_agent_hash`: Hashed user agent
  - `ip_hash`: Hashed IP address

### 3. Authorization

#### Principals
- **ID**: UUID (primary key)
- **Type**: Enum ('user', 'org', 'agent', 'tool', 'dataset', 'service_account')
- **Attributes**:
  - `name`: Principal name
  - `metadata`: JSONB for additional attributes
  - `created_at`, `updated_at`: Timestamps

#### Policies
- **ID**: UUID (primary key)
- **Attributes**:
  - `principal_id`: UUID reference to principal
  - `resource_type`: Type of resource ('agent', 'tool', 'dataset', 'schema', 'table', 'project')
  - `resource_id`: Optional specific resource ID (NULL for wildcard)
  - `permissions`: Array of permission strings (e.g., 'read', 'write', 'execute', 'admin')
  - `conditions`: JSONB for ABAC conditions
  - `created_at`, `updated_at`: Timestamps

#### Tool Permissions
- **ID**: UUID (primary key)
- **Attributes**:
  - `agent_id`: UUID reference to agent
  - `tool_name`: Tool identifier
  - `allowed`: Boolean flag
  - `conditions`: JSONB for additional conditions
  - `created_at`, `updated_at`: Timestamps

#### Data Permissions
- **ID**: UUID (primary key)
- **Attributes**:
  - `principal_id`: UUID reference to principal
  - `schema_name`: Optional schema name
  - `table_name`: Optional table name
  - `row_filter`: Optional SQL WHERE clause for row-level filtering
  - `column_mask`: JSONB map of column names to masking rules
  - `permissions`: Array of permission strings ('select', 'insert', 'update', 'delete')
  - `created_at`, `updated_at`: Timestamps

### 4. Audit Logging

#### Audit Log
- **ID**: BIGSERIAL (primary key)
- **Attributes**:
  - `timestamp`: Event timestamp
  - `principal_id`: UUID reference to principal
  - `api_key_id`: UUID reference to API key
  - `agent_id`: Optional UUID reference to agent
  - `session_id`: Optional UUID reference to session
  - `project_id`: Optional UUID reference to project
  - `action`: Action type (e.g., 'tool_call', 'sql_execute', 'agent_execute', 'login', 'logout')
  - `resource_type`: Type of resource
  - `resource_id`: Resource identifier
  - `inputs_hash`: SHA-256 hash of inputs
  - `outputs_hash`: SHA-256 hash of outputs
  - `inputs`: Optional JSONB inputs (may be truncated)
  - `outputs`: Optional JSONB outputs (may be truncated)
  - `metadata`: JSONB for additional context
  - `created_at`: Creation timestamp

## Standard Roles

### System Roles
- `admin`: Full system access
- `user`: Standard user access (create, read, update)
- `read-only`: Read-only access
- `service`: Service account access (limited scope)

### Project Roles
- `project_admin`: Full project access
- `project_editor`: Edit project resources
- `project_viewer`: View-only access

## Permission Model

### Resource Types
- `agent`: Agent resources
- `tool`: Tool resources
- `dataset`: Dataset resources
- `schema`: Database schema
- `table`: Database table
- `project`: Project/profile resources
- `org`: Organization resources

### Permission Actions
- `read`: Read access
- `write`: Write access
- `execute`: Execute access
- `admin`: Administrative access
- `select`: SQL SELECT permission
- `insert`: SQL INSERT permission
- `update`: SQL UPDATE permission
- `delete`: SQL DELETE permission

## Implementation Strategy

### Phase 1: Schema Alignment
1. Create unified schema migration scripts
2. Migrate existing data to unified model
3. Update foreign key relationships

### Phase 2: Code Integration
1. Create shared identity types/interfaces
2. Update component code to use unified model
3. Implement cross-component identity resolution

### Phase 3: Service-to-Service Auth
1. Implement service account tokens
2. Add token validation middleware
3. Update service communication to use tokens

## Migration Notes

### NeuronDesktop
- `profiles` table maps to `projects` in unified model
- `user_id` TEXT fields need to be migrated to UUID references
- OIDC identities already exist, need to link to unified principals

### NeuronAgent
- `principals` table already exists, needs minor updates
- `api_keys` table needs `principal_type` field
- Policies system is already well-defined

### NeuronMCP
- Tenant context needs to map to unified principals
- Add principal resolution from MCP requests

## Security Considerations

1. **API Key Security**: All API keys must be hashed using bcrypt
2. **Token Rotation**: Service account tokens should rotate periodically
3. **Audit Logging**: All authentication and authorization events must be logged
4. **Row-Level Security**: Use PostgreSQL RLS for data isolation
5. **Rate Limiting**: Enforce rate limits per API key/principal

## Future Enhancements

1. **Multi-Factor Authentication**: Add MFA support
2. **OAuth2 Scopes**: Extend OIDC with custom scopes
3. **Attribute-Based Access Control**: Enhanced ABAC conditions
4. **Federation**: Support for external identity providers
5. **Role Templates**: Predefined role templates for common use cases







