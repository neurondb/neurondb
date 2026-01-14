/*-------------------------------------------------------------------------
 *
 * rbac.go
 *    Role-Based Access Control (RBAC) for NeuronMCP
 *
 * Implements fine-grained permissions per tool with read/write/execute
 * permissions as specified in Phase 2.1.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/security/rbac.go
 *
 *-------------------------------------------------------------------------
 */

package security

import (
	"context"
	"fmt"
	"strings"
)

/* Permission represents a permission level */
type Permission string

const (
	PermissionRead    Permission = "read"
	PermissionWrite  Permission = "write"
	PermissionExecute Permission = "execute"
	PermissionAdmin   Permission = "admin"
)

/* Role represents a user role */
type Role struct {
	Name        string
	Permissions map[string][]Permission /* tool_name -> permissions */
}

/* RBACManager manages role-based access control */
type RBACManager struct {
	roles map[string]*Role
	users map[string]string /* user_id -> role_name */
}

/* NewRBACManager creates a new RBAC manager */
func NewRBACManager() *RBACManager {
	return &RBACManager{
		roles: make(map[string]*Role),
		users: make(map[string]string),
	}
}

/* AddRole adds a role with permissions */
func (r *RBACManager) AddRole(roleName string, permissions map[string][]Permission) {
	r.roles[roleName] = &Role{
		Name:        roleName,
		Permissions: permissions,
	}
}

/* AssignRole assigns a role to a user */
func (r *RBACManager) AssignRole(userID, roleName string) error {
	if _, exists := r.roles[roleName]; !exists {
		return fmt.Errorf("role %s does not exist", roleName)
	}
	r.users[userID] = roleName
	return nil
}

/* HasPermission checks if a user has permission for a tool */
func (r *RBACManager) HasPermission(ctx context.Context, userID, toolName string, permission Permission) bool {
	/* Get user's role */
	roleName, exists := r.users[userID]
	if !exists {
		return false
	}

	/* Get role */
	role, exists := r.roles[roleName]
	if !exists {
		return false
	}

	/* Check if role has admin permission */
	if perms, exists := role.Permissions["*"]; exists {
		for _, p := range perms {
			if p == PermissionAdmin {
				return true
			}
		}
	}

	/* Check tool-specific permissions */
	if perms, exists := role.Permissions[toolName]; exists {
		for _, p := range perms {
			if p == permission || p == PermissionAdmin {
				return true
			}
		}
	}

	/* Check wildcard permissions */
	if perms, exists := role.Permissions["*"]; exists {
		for _, p := range perms {
			if p == permission || p == PermissionAdmin {
				return true
			}
		}
	}

	return false
}

/* GetUserPermissions returns all permissions for a user */
func (r *RBACManager) GetUserPermissions(ctx context.Context, userID string) map[string][]Permission {
	roleName, exists := r.users[userID]
	if !exists {
		return make(map[string][]Permission)
	}

	role, exists := r.roles[roleName]
	if !exists {
		return make(map[string][]Permission)
	}

	return role.Permissions
}

/* GetRequiredPermission returns the required permission for a tool operation */
func GetRequiredPermission(toolName, operation string) Permission {
	/* Determine permission based on tool name and operation */
	lowerTool := strings.ToLower(toolName)
	lowerOp := strings.ToLower(operation)

	/* Read operations */
	if strings.Contains(lowerOp, "read") || strings.Contains(lowerOp, "get") ||
		strings.Contains(lowerOp, "list") || strings.Contains(lowerOp, "search") ||
		strings.Contains(lowerTool, "read") || strings.Contains(lowerTool, "get") ||
		strings.Contains(lowerTool, "list") || strings.Contains(lowerTool, "search") {
		return PermissionRead
	}

	/* Write operations */
	if strings.Contains(lowerOp, "write") || strings.Contains(lowerOp, "create") ||
		strings.Contains(lowerOp, "update") || strings.Contains(lowerOp, "delete") ||
		strings.Contains(lowerOp, "insert") || strings.Contains(lowerOp, "modify") ||
		strings.Contains(lowerTool, "create") || strings.Contains(lowerTool, "update") ||
		strings.Contains(lowerTool, "delete") || strings.Contains(lowerTool, "insert") {
		return PermissionWrite
	}

	/* Execute operations (default for most tools) */
	return PermissionExecute
}



