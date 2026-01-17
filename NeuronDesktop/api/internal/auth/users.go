/*-------------------------------------------------------------------------
 *
 * users.go
 *    User management for NeuronDesktop
 *
 * Implements user accounts, authentication, and user management
 * for multi-user support in NeuronDesktop.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronDesktop/api/internal/auth/users.go
 *
 *-------------------------------------------------------------------------
 */

package auth

import (
	"context"
	"database/sql"
	"time"

	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
)

// User represents a NeuronDesktop user
type User struct {
	ID           uuid.UUID `json:"id"`
	Email        string    `json:"email"`
	PasswordHash string    `json:"-"`
	Name         string    `json:"name"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
	Status       string    `json:"status"`
	OrganizationID *uuid.UUID `json:"organization_id,omitempty"`
}

// UserManager manages users
type UserManager struct {
	db *sql.DB
}

// NewUserManager creates a new user manager
func NewUserManager(db *sql.DB) *UserManager {
	return &UserManager{db: db}
}

// CreateUser creates a new user
func (um *UserManager) CreateUser(ctx context.Context, email, password, name string) (*User, error) {
	// Hash password
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return nil, err
	}

	user := &User{
		ID:           uuid.New(),
		Email:        email,
		PasswordHash: string(hashedPassword),
		Name:         name,
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
		Status:       "active",
	}

	// Insert into database
	query := `INSERT INTO users (id, email, password_hash, name, created_at, updated_at, status)
	          VALUES ($1, $2, $3, $4, $5, $6, $7)`
	_, err = um.db.ExecContext(ctx, query,
		user.ID, user.Email, user.PasswordHash, user.Name,
		user.CreatedAt, user.UpdatedAt, user.Status)
	if err != nil {
		return nil, err
	}

	return user, nil
}

// GetUserByEmail retrieves a user by email
func (um *UserManager) GetUserByEmail(ctx context.Context, email string) (*User, error) {
	user := &User{}
	query := `SELECT id, email, password_hash, name, created_at, updated_at, status, organization_id
	          FROM users WHERE email = $1`
	err := um.db.QueryRowContext(ctx, query, email).Scan(
		&user.ID, &user.Email, &user.PasswordHash, &user.Name,
		&user.CreatedAt, &user.UpdatedAt, &user.Status, &user.OrganizationID)
	if err != nil {
		return nil, err
	}
	return user, nil
}

// VerifyPassword verifies a user's password
func (um *UserManager) VerifyPassword(user *User, password string) bool {
	err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(password))
	return err == nil
}



