# Identity Integration Guide

## Overview

This guide explains how to integrate the unified identity model into NeuronDesktop, NeuronAgent, and NeuronMCP components.

## Migration Steps

### 1. Run Database Migrations

#### NeuronDesktop
```bash
psql -d neurondesk -f NeuronDesktop/api/migrations/008_unified_identity_model.sql
```

#### NeuronAgent
```bash
psql -d neurondb -f NeuronAgent/sql/014_unified_identity_model.sql
```

### 2. Update Code to Use Unified Types

#### Import Shared Types
```go
import "github.com/neurondb/neurondb/pkg/identity"
```

#### Use Unified Principal Types
```go
// Instead of component-specific types
principal := &identity.Principal{
    ID:   principalID,
    Type: identity.PrincipalTypeUser,
    Name: username,
}
```

### 3. Update API Key Resolution

#### Before (Component-Specific)
```go
// NeuronAgent-specific
apiKey, err := queries.GetAPIKeyByHash(ctx, keyHash)
```

#### After (Unified)
```go
// Use shared identity resolver
principal, err := identityResolver.ResolvePrincipalForAPIKey(ctx, keyPrefix)
```

### 4. Update Permission Checks

#### Before
```go
// Component-specific permission check
if !auth.HasRole(apiKey, "admin") {
    return errors.New("insufficient permissions")
}
```

#### After
```go
// Unified permission check
hasPermission, err := permissionChecker.HasPermission(
    ctx,
    principalID,
    identity.ResourceTypeAgent,
    &agentID,
    identity.PermissionWrite,
)
```

### 5. Update Audit Logging

#### Before
```go
// Component-specific audit log
auditLog := &db.AuditLog{
    PrincipalID: userID,
    Action:      "tool_call",
    // ...
}
```

#### After
```go
// Unified audit log
entry, err := identity.NewAuditLogEntry(
    identity.ActionToolCall,
    identity.ResourceTypeTool,
    &toolName,
    &principalID,
    &apiKeyID,
    inputs,
    outputs,
)
auditLogger.Log(ctx, entry)
```

## Component-Specific Integration

### NeuronDesktop

#### Update User Model
```go
// Use unified User type
user := &identity.User{
    ID:       userID,
    Username: username,
    Email:    &email,
    IsAdmin:  isAdmin,
}
```

#### Update Profile to Project
```go
// Map Profile to Project
project := &identity.Project{
    ID:          profileID,
    Name:        profileName,
    UserID:      &userID,
    NeuronDBDSN: neurondbDSN,
    MCPConfig:   mcpConfig,
}
```

### NeuronAgent

#### Update Principal Resolution
```go
// Use unified principal resolution
principal, err := identityResolver.GetPrincipal(ctx, principalID)
if err != nil {
    return err
}

// Check permissions using unified model
hasPermission, err := permissionChecker.HasPermission(
    ctx,
    principalID,
    identity.ResourceTypeAgent,
    &agentID,
    identity.PermissionExecute,
)
```

#### Update API Key Model
```go
// Use unified APIKey type
apiKey := &identity.APIKey{
    ID:              keyID,
    KeyPrefix:       keyPrefix,
    PrincipalID:     &principalID,
    PrincipalType:   &identity.PrincipalTypeUser,
    RateLimitPerMin: rateLimit,
    Roles:           roles,
}
```

### NeuronMCP

#### Update Tenant Context
```go
// Use unified TenantContext
tenantCtx := &identity.TenantContext{
    UserID:    userID,
    OrgID:     orgID,
    ProjectID: projectID,
    Scopes:    scopes,
}

// Resolve principal
principal, err := identityResolver.GetPrincipal(ctx, principalID)
```

## Service-to-Service Authentication

### Create Service Account Token
```go
// Create service account
serviceAccount := &identity.ServiceAccount{
    ID:   serviceAccountID,
    Name: "desktop-api",
    OrgID: &orgID,
}

// Create API key for service account
apiKey := &identity.APIKey{
    PrincipalID:     &serviceAccountID,
    PrincipalType:    &identity.PrincipalTypeServiceAccount,
    RateLimitPerMin: 1000,
    Roles:           []string{string(identity.RoleService)},
}
```

### Use Service Account Token
```go
// In service-to-service calls
req.Header.Set("Authorization", "Bearer "+serviceAccountToken)

// Validate service account token
principal, err := identityResolver.ResolvePrincipalForAPIKey(ctx, keyPrefix)
if principal.Type != identity.PrincipalTypeServiceAccount {
    return errors.New("invalid service account")
}
```

## Testing

### Unit Tests
```go
func TestUnifiedIdentity(t *testing.T) {
    // Create test principal
    principal := &identity.Principal{
        ID:   uuid.New(),
        Type: identity.PrincipalTypeUser,
        Name: "testuser",
    }
    
    // Test permission check
    hasPermission, err := permissionChecker.HasPermission(
        ctx,
        principal.ID,
        identity.ResourceTypeAgent,
        nil,
        identity.PermissionRead,
    )
    assert.NoError(t, err)
    assert.True(t, hasPermission)
}
```

### Integration Tests
```go
func TestIdentityIntegration(t *testing.T) {
    // Run migrations
    runMigrations(t)
    
    // Create test user
    user := createTestUser(t)
    
    // Create API key
    apiKey := createTestAPIKey(t, user.ID)
    
    // Resolve principal
    principal, err := identityResolver.ResolvePrincipalForAPIKey(ctx, apiKey.KeyPrefix)
    assert.NoError(t, err)
    assert.Equal(t, identity.PrincipalTypeUser, principal.Type)
}
```

## Best Practices

1. **Always use unified types** - Don't create component-specific identity types
2. **Use identity resolver** - Don't query database directly for principals
3. **Log all actions** - Use unified audit logging for all operations
4. **Check permissions** - Always check permissions before operations
5. **Use service accounts** - Use service accounts for service-to-service auth
6. **Validate tokens** - Always validate API keys and tokens
7. **Audit everything** - Log all authentication and authorization events

## Troubleshooting

### Migration Issues
- Ensure all migrations run in order
- Check for foreign key constraints
- Verify UUID types are correct

### Permission Issues
- Check principal type matches expected type
- Verify policy exists for resource
- Check role assignments

### Audit Logging Issues
- Verify audit log table exists
- Check principal ID is valid
- Ensure timestamps are set correctly








