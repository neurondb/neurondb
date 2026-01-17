/*-------------------------------------------------------------------------
 *
 * organizations.go
 *    Organization management for NeuronDesktop
 *
 * Implements multi-tenant organizations with user management.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronDesktop/api/internal/auth/organizations.go
 *
 *-------------------------------------------------------------------------
 */

package auth

import (
	"context"
	"database/sql"
	"time"

	"github.com/google/uuid"
)

// Organization represents a NeuronDesktop organization
type Organization struct {
	ID        uuid.UUID `json:"id"`
	Name      string    `json:"name"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	Status    string    `json:"status"`
}

// OrganizationManager manages organizations
type OrganizationManager struct {
	db *sql.DB
}

// NewOrganizationManager creates a new organization manager
func NewOrganizationManager(db *sql.DB) *OrganizationManager {
	return &OrganizationManager{db: db}
}

// CreateOrganization creates a new organization
func (om *OrganizationManager) CreateOrganization(ctx context.Context, name string) (*Organization, error) {
	org := &Organization{
		ID:        uuid.New(),
		Name:      name,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		Status:    "active",
	}

	query := `INSERT INTO organizations (id, name, created_at, updated_at, status)
	          VALUES ($1, $2, $3, $4, $5)`
	_, err := om.db.ExecContext(ctx, query,
		org.ID, org.Name, org.CreatedAt, org.UpdatedAt, org.Status)
	if err != nil {
		return nil, err
	}

	return org, nil
}

// GetOrganization retrieves an organization by ID
func (om *OrganizationManager) GetOrganization(ctx context.Context, id uuid.UUID) (*Organization, error) {
	org := &Organization{}
	query := `SELECT id, name, created_at, updated_at, status
	          FROM organizations WHERE id = $1`
	err := om.db.QueryRowContext(ctx, query, id).Scan(
		&org.ID, &org.Name, &org.CreatedAt, &org.UpdatedAt, &org.Status)
	if err != nil {
		return nil, err
	}
	return org, nil
}

// AddUserToOrganization adds a user to an organization
func (om *OrganizationManager) AddUserToOrganization(ctx context.Context, userID, orgID uuid.UUID, role string) error {
	query := `INSERT INTO organization_members (user_id, organization_id, role, created_at)
	          VALUES ($1, $2, $3, $4)
	          ON CONFLICT (user_id, organization_id) DO UPDATE SET role = $3`
	_, err := om.db.ExecContext(ctx, query, userID, orgID, role, time.Now())
	return err
}



