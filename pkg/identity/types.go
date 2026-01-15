/*-------------------------------------------------------------------------
 *
 * types.go
 *    Unified identity types for NeuronDB ecosystem
 *
 * Provides shared types for identity, authentication, and authorization
 * across NeuronDesktop, NeuronAgent, and NeuronMCP.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    pkg/identity/types.go
 *
 *-------------------------------------------------------------------------
 */

package identity

import (
	"time"

	"github.com/google/uuid"
)

/* PrincipalType represents the type of a principal */
type PrincipalType string

const (
	PrincipalTypeUser          PrincipalType = "user"
	PrincipalTypeOrg           PrincipalType = "org"
	PrincipalTypeAgent         PrincipalType = "agent"
	PrincipalTypeTool          PrincipalType = "tool"
	PrincipalTypeDataset       PrincipalType = "dataset"
	PrincipalTypeServiceAccount PrincipalType = "service_account"
)

/* Principal represents a unified principal entity */
type Principal struct {
	ID        uuid.UUID              `json:"id"`
	Type      PrincipalType          `json:"type"`
	Name      string                 `json:"name"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt time.Time              `json:"created_at"`
	UpdatedAt time.Time              `json:"updated_at"`
}

/* User represents a user in the system */
type User struct {
	ID        uuid.UUID              `json:"id"`
	Username  string                 `json:"username"`
	Email     *string                `json:"email,omitempty"`
	IsAdmin   bool                   `json:"is_admin"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt time.Time              `json:"created_at"`
	UpdatedAt time.Time              `json:"updated_at"`
}

/* Organization represents an organization */
type Organization struct {
	ID        uuid.UUID              `json:"id"`
	Name      string                 `json:"name"`
	Slug      string                 `json:"slug"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt time.Time              `json:"created_at"`
	UpdatedAt time.Time              `json:"updated_at"`
}

/* Project represents a project/profile */
type Project struct {
	ID              uuid.UUID              `json:"id"`
	Name            string                 `json:"name"`
	OrgID           *uuid.UUID             `json:"org_id,omitempty"`
	UserID          *uuid.UUID             `json:"user_id,omitempty"`
	IsDefault       bool                   `json:"is_default"`
	NeuronDBDSN     string                 `json:"neurondb_dsn"`
	MCPConfig       map[string]interface{} `json:"mcp_config,omitempty"`
	AgentEndpoint   *string                `json:"agent_endpoint,omitempty"`
	AgentAPIKey     *string                `json:"agent_api_key,omitempty"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
}

/* ServiceAccount represents a service account */
type ServiceAccount struct {
	ID        uuid.UUID              `json:"id"`
	Name      string                 `json:"name"`
	OrgID     *uuid.UUID             `json:"org_id,omitempty"`
	ProjectID *uuid.UUID             `json:"project_id,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt time.Time              `json:"created_at"`
	UpdatedAt time.Time              `json:"updated_at"`
}

/* APIKey represents an API key */
type APIKey struct {
	ID                uuid.UUID              `json:"id"`
	KeyPrefix         string                 `json:"key_prefix"`
	PrincipalID       *uuid.UUID             `json:"principal_id,omitempty"`
	PrincipalType     *PrincipalType         `json:"principal_type,omitempty"`
	ProjectID         *uuid.UUID             `json:"project_id,omitempty"`
	RateLimitPerMin   int                    `json:"rate_limit_per_minute"`
	Roles             []string               `json:"roles"`
	ExpiresAt         *time.Time             `json:"expires_at,omitempty"`
	LastUsedAt        *time.Time            `json:"last_used_at,omitempty"`
	Metadata          map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt         time.Time              `json:"created_at"`
}

/* Policy represents a permission policy */
type Policy struct {
	ID           uuid.UUID              `json:"id"`
	PrincipalID uuid.UUID              `json:"principal_id"`
	ResourceType string                `json:"resource_type"`
	ResourceID   *string               `json:"resource_id,omitempty"`
	Permissions  []string               `json:"permissions"`
	Conditions   map[string]interface{} `json:"conditions,omitempty"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
}

/* ResourceType represents the type of a resource */
type ResourceType string

const (
	ResourceTypeAgent   ResourceType = "agent"
	ResourceTypeTool    ResourceType = "tool"
	ResourceTypeDataset ResourceType = "dataset"
	ResourceTypeSchema  ResourceType = "schema"
	ResourceTypeTable   ResourceType = "table"
	ResourceTypeProject ResourceType = "project"
	ResourceTypeOrg     ResourceType = "org"
)

/* Permission represents a permission action */
type Permission string

const (
	PermissionRead   Permission = "read"
	PermissionWrite  Permission = "write"
	PermissionExecute Permission = "execute"
	PermissionAdmin  Permission = "admin"
	PermissionSelect Permission = "select"
	PermissionInsert Permission = "insert"
	PermissionUpdate Permission = "update"
	PermissionDelete Permission = "delete"
)

/* StandardRole represents a standard role */
type StandardRole string

const (
	RoleAdmin     StandardRole = "admin"
	RoleUser      StandardRole = "user"
	RoleReadOnly  StandardRole = "read-only"
	RoleService   StandardRole = "service"
	RoleProjectAdmin StandardRole = "project_admin"
	RoleProjectEditor StandardRole = "project_editor"
	RoleProjectViewer StandardRole = "project_viewer"
)

/* AuditLogEntry represents an audit log entry */
type AuditLogEntry struct {
	ID           int64                  `json:"id"`
	Timestamp    time.Time              `json:"timestamp"`
	PrincipalID  *uuid.UUID             `json:"principal_id,omitempty"`
	APIKeyID     *uuid.UUID             `json:"api_key_id,omitempty"`
	AgentID      *uuid.UUID             `json:"agent_id,omitempty"`
	SessionID    *uuid.UUID             `json:"session_id,omitempty"`
	ProjectID    *uuid.UUID             `json:"project_id,omitempty"`
	Action       string                 `json:"action"`
	ResourceType string                 `json:"resource_type"`
	ResourceID   *string                `json:"resource_id,omitempty"`
	InputsHash   *string                `json:"inputs_hash,omitempty"`
	OutputsHash  *string                `json:"outputs_hash,omitempty"`
	Inputs       map[string]interface{} `json:"inputs,omitempty"`
	Outputs      map[string]interface{} `json:"outputs,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt    time.Time              `json:"created_at"`
}

/* TenantContext represents tenant isolation context */
type TenantContext struct {
	UserID    string
	OrgID     string
	ProjectID string
	Scopes    []string
}

/* IdentityResolver provides methods to resolve identity information */
type IdentityResolver interface {
	GetPrincipal(ctx interface{}, principalID uuid.UUID) (*Principal, error)
	GetUser(ctx interface{}, userID uuid.UUID) (*User, error)
	GetOrganization(ctx interface{}, orgID uuid.UUID) (*Organization, error)
	GetServiceAccount(ctx interface{}, serviceAccountID uuid.UUID) (*ServiceAccount, error)
	ResolvePrincipalForAPIKey(ctx interface{}, apiKeyPrefix string) (*Principal, error)
}

/* PermissionChecker provides methods to check permissions */
type PermissionChecker interface {
	HasPermission(ctx interface{}, principalID uuid.UUID, resourceType ResourceType, resourceID *string, permission Permission) (bool, error)
	HasRole(ctx interface{}, principalID uuid.UUID, role StandardRole) (bool, error)
	CheckPolicy(ctx interface{}, principalID uuid.UUID, resourceType ResourceType, resourceID *string, permission Permission) (bool, error)
}








