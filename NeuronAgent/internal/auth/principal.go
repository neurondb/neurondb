/*-------------------------------------------------------------------------
 *
 * principal.go
 *    Principal management for NeuronAgent
 *
 * Provides principal creation, lookup, and resolution from API keys.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/auth/principal.go
 *
 *-------------------------------------------------------------------------
 */

package auth

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type PrincipalManager struct {
	queries *db.Queries
}

func NewPrincipalManager(queries *db.Queries) *PrincipalManager {
	return &PrincipalManager{queries: queries}
}

/* GetOrCreatePrincipalForAPIKey gets or creates a principal for an API key */
func (m *PrincipalManager) GetOrCreatePrincipalForAPIKey(ctx context.Context, apiKey *db.APIKey) (*db.Principal, error) {
	/* If principal_id is already set, return that principal */
	if apiKey.PrincipalID != nil {
		principal, err := m.queries.GetPrincipalByID(ctx, *apiKey.PrincipalID)
		if err == nil {
			return principal, nil
		}
		/* If principal not found, continue to create new one */
	}

	/* Determine principal type and name from API key */
	principalType := "user"
	principalName := "api_key_" + apiKey.ID.String()

	if apiKey.UserID != nil && *apiKey.UserID != "" {
		principalType = "user"
		principalName = "user_" + *apiKey.UserID
	} else if apiKey.OrganizationID != nil && *apiKey.OrganizationID != "" {
		principalType = "org"
		principalName = "org_" + *apiKey.OrganizationID
	}

	/* Try to get existing principal */
	principal, err := m.queries.GetPrincipalByTypeAndName(ctx, principalType, principalName)
	if err == nil {
		return principal, nil
	}

	/* Create new principal */
	principal = &db.Principal{
		Type:     principalType,
		Name:     principalName,
		Metadata: make(db.JSONBMap),
	}
	if apiKey.Metadata != nil {
		principal.Metadata = apiKey.Metadata
	}

	if err := m.queries.CreatePrincipal(ctx, principal); err != nil {
		return nil, fmt.Errorf("failed to create principal: %w", err)
	}

	/* Link API key to principal */
	if apiKey.PrincipalID == nil {
		apiKey.PrincipalID = &principal.ID
		/* Note: This would require an UpdateAPIKey method that includes principal_id */
		/* For now, we'll just return the principal and the caller can handle linking */
	}

	return principal, nil
}

/* ResolvePrincipalFromAPIKey resolves the principal for an API key */
func (m *PrincipalManager) ResolvePrincipalFromAPIKey(ctx context.Context, apiKey *db.APIKey) (*db.Principal, error) {
	if apiKey.PrincipalID == nil {
		/* Auto-create principal if not linked */
		return m.GetOrCreatePrincipalForAPIKey(ctx, apiKey)
	}

	principal, err := m.queries.GetPrincipalByID(ctx, *apiKey.PrincipalID)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve principal: %w", err)
	}

	return principal, nil
}

/* CreatePrincipal creates a new principal */
func (m *PrincipalManager) CreatePrincipal(ctx context.Context, principalType, name string, metadata map[string]interface{}) (*db.Principal, error) {
	principal := &db.Principal{
		Type:     principalType,
		Name:     name,
		Metadata: metadata,
	}

	if err := m.queries.CreatePrincipal(ctx, principal); err != nil {
		return nil, fmt.Errorf("failed to create principal: %w", err)
	}

	return principal, nil
}

/* GetPrincipalByID gets a principal by ID */
func (m *PrincipalManager) GetPrincipalByID(ctx context.Context, id uuid.UUID) (*db.Principal, error) {
	return m.queries.GetPrincipalByID(ctx, id)
}

/* GetPrincipalByTypeAndName gets a principal by type and name */
func (m *PrincipalManager) GetPrincipalByTypeAndName(ctx context.Context, principalType, name string) (*db.Principal, error) {
	return m.queries.GetPrincipalByTypeAndName(ctx, principalType, name)
}
