/*-------------------------------------------------------------------------
 *
 * context.go
 *    Context helper functions for API handlers
 *
 * Provides functions to extract API keys, principals, and other context values.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/context.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronAgent/internal/db"
)

/* GetAPIKeyFromContext gets the API key from context */
func GetAPIKeyFromContext(ctx context.Context) (*db.APIKey, bool) {
	apiKey, ok := ctx.Value(apiKeyContextKey).(*db.APIKey)
	return apiKey, ok
}

/* GetPrincipalFromContext gets the principal from context */
func GetPrincipalFromContext(ctx context.Context) (*db.Principal, bool) {
	principal, ok := ctx.Value(principalContextKey).(*db.Principal)
	return principal, ok
}

/* MustGetAPIKeyFromContext gets the API key from context or returns error */
func MustGetAPIKeyFromContext(ctx context.Context) (*db.APIKey, error) {
	apiKey, ok := GetAPIKeyFromContext(ctx)
	if !ok {
		return nil, fmt.Errorf("API key not found in context: authentication required")
	}
	return apiKey, nil
}

/* MustGetPrincipalFromContext gets the principal from context or returns error */
func MustGetPrincipalFromContext(ctx context.Context) (*db.Principal, error) {
	principal, ok := GetPrincipalFromContext(ctx)
	if !ok {
		return nil, fmt.Errorf("Principal not found in context: authentication required")
	}
	return principal, nil
}

/* GetAuthFromContext gets both API key and principal from context */
func GetAuthFromContext(ctx context.Context) (*db.APIKey, *db.Principal) {
	apiKey, _ := GetAPIKeyFromContext(ctx)
	principal, _ := GetPrincipalFromContext(ctx)
	return apiKey, principal
}
