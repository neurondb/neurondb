/*-------------------------------------------------------------------------
 *
 * tenant.go
 *    Multi-tenant organization and project isolation
 *
 * Provides org and project isolation for multi-tenant authentication.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/auth/tenant.go
 *
 *-------------------------------------------------------------------------
 */

package auth

import (
	"context"
)

/* TenantContext provides tenant isolation context */
type TenantContext struct {
	UserID    string
	OrgID     string
	ProjectID string
	Scopes    []string
}

/* TenantResolver resolves tenant information from context */
type TenantResolver interface {
	ResolveTenant(ctx context.Context, userID string) (*TenantContext, error)
	IsOrgMember(userID, orgID string) (bool, error)
	IsProjectMember(userID, projectID string) (bool, error)
}

/* DefaultTenantResolver provides a default tenant resolver implementation */
type DefaultTenantResolver struct {
	userOrgs     map[string][]string
	userProjects map[string][]string
	projectOrgs  map[string]string // projectID -> orgID
}

/* NewDefaultTenantResolver creates a default tenant resolver */
func NewDefaultTenantResolver() *DefaultTenantResolver {
	return &DefaultTenantResolver{
		userOrgs:     make(map[string][]string),
		userProjects: make(map[string][]string),
		projectOrgs:  make(map[string]string),
	}
}

/* SetUserOrg sets organization membership for a user */
func (r *DefaultTenantResolver) SetUserOrg(userID, orgID string) {
	if r.userOrgs[userID] == nil {
		r.userOrgs[userID] = make([]string, 0)
	}
	/* Check if already exists */
	for _, existing := range r.userOrgs[userID] {
		if existing == orgID {
			return
		}
	}
	r.userOrgs[userID] = append(r.userOrgs[userID], orgID)
}

/* SetUserProject sets project membership for a user */
func (r *DefaultTenantResolver) SetUserProject(userID, projectID, orgID string) {
	if r.userProjects[userID] == nil {
		r.userProjects[userID] = make([]string, 0)
	}
	/* Check if already exists */
	for _, existing := range r.userProjects[userID] {
		if existing == projectID {
			return
		}
	}
	r.userProjects[userID] = append(r.userProjects[userID], projectID)
	r.projectOrgs[projectID] = orgID
}

/* ResolveTenant resolves tenant information from context */
func (r *DefaultTenantResolver) ResolveTenant(ctx context.Context, userID string) (*TenantContext, error) {
	/* Extract from context if available */
	orgID := ""
	if org := ctx.Value("org_id"); org != nil {
		if id, ok := org.(string); ok {
			orgID = id
		}
	}

	projectID := ""
	if project := ctx.Value("project_id"); project != nil {
		if id, ok := project.(string); ok {
			projectID = id
		}
	}

	scopes := []string{}
	if s := ctx.Value("scopes"); s != nil {
		if sc, ok := s.([]string); ok {
			scopes = sc
		}
	}

	return &TenantContext{
		UserID:    userID,
		OrgID:     orgID,
		ProjectID: projectID,
		Scopes:    scopes,
	}, nil
}

/* IsOrgMember checks if a user is a member of an organization */
func (r *DefaultTenantResolver) IsOrgMember(userID, orgID string) (bool, error) {
	orgs, exists := r.userOrgs[userID]
	if !exists {
		return false, nil
	}

	for _, org := range orgs {
		if org == orgID {
			return true, nil
		}
	}
	return false, nil
}

/* IsProjectMember checks if a user is a member of a project */
func (r *DefaultTenantResolver) IsProjectMember(userID, projectID string) (bool, error) {
	projects, exists := r.userProjects[userID]
	if !exists {
		return false, nil
	}

	for _, project := range projects {
		if project == projectID {
			return true, nil
		}
	}
	return false, nil
}


