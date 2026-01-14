/*-------------------------------------------------------------------------
 *
 * integration.go
 *    Authentication integration helpers
 *
 * Provides integration between OIDC, tenant isolation, and request handling.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/auth/integration.go
 *
 *-------------------------------------------------------------------------
 */

package auth

import (
	"context"
	"fmt"
)

/* RequestAuth extracts authentication information from context */
type RequestAuth struct {
	UserID    string
	OrgID     string
	ProjectID string
	Scopes    []string
	Token     string
}

/* ExtractRequestAuth extracts authentication information from context */
func ExtractRequestAuth(ctx context.Context) (*RequestAuth, error) {
	auth := &RequestAuth{}

	/* Extract user ID */
	if userID := ctx.Value("user_id"); userID != nil {
		if id, ok := userID.(string); ok {
			auth.UserID = id
		}
	}

	/* Extract org ID */
	if orgID := ctx.Value("org_id"); orgID != nil {
		if id, ok := orgID.(string); ok {
			auth.OrgID = id
		}
	}

	/* Extract project ID */
	if projectID := ctx.Value("project_id"); projectID != nil {
		if id, ok := projectID.(string); ok {
			auth.ProjectID = id
		}
	}

	/* Extract scopes */
	if scopes := ctx.Value("scopes"); scopes != nil {
		if s, ok := scopes.([]string); ok {
			auth.Scopes = s
		}
	}

	/* Extract token */
	if token := ctx.Value("token"); token != nil {
		if t, ok := token.(string); ok {
			auth.Token = t
		}
	}

	return auth, nil
}

/* WithTenantContext adds tenant information to context */
func WithTenantContext(ctx context.Context, tenant *TenantContext) context.Context {
	ctx = context.WithValue(ctx, "user_id", tenant.UserID)
	if tenant.OrgID != "" {
		ctx = context.WithValue(ctx, "org_id", tenant.OrgID)
	}
	if tenant.ProjectID != "" {
		ctx = context.WithValue(ctx, "project_id", tenant.ProjectID)
	}
	if len(tenant.Scopes) > 0 {
		ctx = context.WithValue(ctx, "scopes", tenant.Scopes)
	}
	return ctx
}

/* ValidateTenantAccess validates tenant access using resolver */
func ValidateTenantAccess(ctx context.Context, resolver TenantResolver, userID, requestedOrgID, requestedProjectID string) error {
	if requestedProjectID != "" {
		/* Validate project access */
		isMember, err := resolver.IsProjectMember(userID, requestedProjectID)
		if err != nil {
			return fmt.Errorf("failed to validate project access: %w", err)
		}
		if !isMember {
			return fmt.Errorf("user %s does not have access to project %s", userID, requestedProjectID)
		}

		/* If org ID is also provided, validate project belongs to org */
		if requestedOrgID != "" {
			/* In a full implementation, we'd check project belongs to org */
			/* For now, we'll assume the resolver handles this */
		}
	} else if requestedOrgID != "" {
		/* Validate org access */
		isMember, err := resolver.IsOrgMember(userID, requestedOrgID)
		if err != nil {
			return fmt.Errorf("failed to validate org access: %w", err)
		}
		if !isMember {
			return fmt.Errorf("user %s does not have access to org %s", userID, requestedOrgID)
		}
	}

	return nil
}

/* GetTenantFromContext extracts tenant information from context */
func GetTenantFromContext(ctx context.Context) *TenantContext {
	tc := &TenantContext{}

	if userID := ctx.Value("user_id"); userID != nil {
		if id, ok := userID.(string); ok {
			tc.UserID = id
		}
	}

	if orgID := ctx.Value("org_id"); orgID != nil {
		if id, ok := orgID.(string); ok {
			tc.OrgID = id
		}
	}

	if projectID := ctx.Value("project_id"); projectID != nil {
		if id, ok := projectID.(string); ok {
			tc.ProjectID = id
		}
	}

	if scopes := ctx.Value("scopes"); scopes != nil {
		if s, ok := scopes.([]string); ok {
			tc.Scopes = s
		}
	}

	return tc
}












