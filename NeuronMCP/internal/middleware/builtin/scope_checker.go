/*-------------------------------------------------------------------------
 *
 * scope_checker.go
 *    Scope checker interface and default implementation
 *
 * Provides scope checking for scoped authentication middleware.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/builtin/scope_checker.go
 *
 *-------------------------------------------------------------------------
 */

package builtin

import "context"

/* ScopeChecker defines the interface for checking user scopes */
type ScopeChecker interface {
	HasScope(ctx context.Context, requiredScope string) bool
	GetUserScopes(ctx context.Context) []string
}

/* DefaultScopeChecker provides a default scope checking implementation */
type DefaultScopeChecker struct {
	/* In a real implementation, this would query scopes from a database or cache */
	/* For now, we use a simple hardcoded approach for demonstration */
}

/* NewDefaultScopeChecker creates a new default scope checker */
func NewDefaultScopeChecker() *DefaultScopeChecker {
	return &DefaultScopeChecker{}
}

/* HasScope checks if the user has the required scope */
func (s *DefaultScopeChecker) HasScope(ctx context.Context, requiredScope string) bool {
	scopes := s.GetUserScopes(ctx)
	for _, scope := range scopes {
		if scope == requiredScope || scope == "admin:full" {
			return true
		}
	}
	return false
}

/* GetUserScopes retrieves the user's scopes from context */
func (s *DefaultScopeChecker) GetUserScopes(ctx context.Context) []string {
	/* Try to get scopes from context */
	if scopes, ok := ctx.Value("scopes").([]string); ok {
		return scopes
	}

	/* Default: return empty scopes (no access) */
	/* In a real implementation, this would query from auth system */
	return []string{}
}










