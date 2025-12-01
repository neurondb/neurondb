package auth

import (
	"fmt"

	"github.com/neurondb/NeuronAgent/internal/db"
)

const (
	RoleAdmin    = "admin"
	RoleUser     = "user"
	RoleReadOnly = "read-only"
)

// HasRole checks if an API key has a specific role
func HasRole(apiKey *db.APIKey, role string) bool {
	for _, r := range apiKey.Roles {
		if r == role {
			return true
		}
	}
	return false
}

// RequireRole checks if an API key has required role, returns error if not
func RequireRole(apiKey *db.APIKey, role string) error {
	if !HasRole(apiKey, role) {
		return fmt.Errorf("insufficient permissions: role %s required", role)
	}
	return nil
}

// RequireAnyRole checks if an API key has any of the required roles
func RequireAnyRole(apiKey *db.APIKey, roles ...string) error {
	for _, role := range roles {
		if HasRole(apiKey, role) {
			return nil
		}
	}
	return fmt.Errorf("insufficient permissions: one of roles %v required", roles)
}

// RequireAllRoles checks if an API key has all of the required roles
func RequireAllRoles(apiKey *db.APIKey, roles ...string) error {
	for _, role := range roles {
		if !HasRole(apiKey, role) {
			return fmt.Errorf("insufficient permissions: role %s required", role)
		}
	}
	return nil
}

// CanCreate checks if API key can create resources
func CanCreate(apiKey *db.APIKey) bool {
	return HasRole(apiKey, RoleAdmin) || HasRole(apiKey, RoleUser)
}

// CanUpdate checks if API key can update resources
func CanUpdate(apiKey *db.APIKey) bool {
	return HasRole(apiKey, RoleAdmin) || HasRole(apiKey, RoleUser)
}

// CanDelete checks if API key can delete resources
func CanDelete(apiKey *db.APIKey) bool {
	return HasRole(apiKey, RoleAdmin)
}

// CanRead checks if API key can read resources
func CanRead(apiKey *db.APIKey) bool {
	return HasRole(apiKey, RoleAdmin) || HasRole(apiKey, RoleUser) || HasRole(apiKey, RoleReadOnly)
}

