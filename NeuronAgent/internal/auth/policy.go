/*-------------------------------------------------------------------------
 *
 * policy.go
 *    Policy evaluation engine for NeuronAgent
 *
 * Provides RBAC + ABAC policy evaluation for permissions checking.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/auth/policy.go
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

type PolicyEngine struct {
	queries *db.Queries
}

func NewPolicyEngine(queries *db.Queries) *PolicyEngine {
	return &PolicyEngine{queries: queries}
}

/* CheckPermission checks if a principal has permission to perform an action on a resource */
func (e *PolicyEngine) CheckPermission(ctx context.Context, principalID uuid.UUID, action string, resourceType string, resourceID *string, resourceTags map[string]string) (bool, error) {
	/* Get all policies for this principal */
	policies, err := e.queries.ListPoliciesByPrincipal(ctx, principalID)
	if err != nil {
		return false, fmt.Errorf("failed to list policies: %w", err)
	}

	/* Check each policy */
	for _, policy := range policies {
		if e.matchesPolicy(policy, action, resourceType, resourceID, resourceTags) {
			return true, nil
		}
	}

	return false, nil
}

/* matchesPolicy checks if a policy matches the requested action and resource */
func (e *PolicyEngine) matchesPolicy(policy db.Policy, action string, resourceType string, resourceID *string, resourceTags map[string]string) bool {
	/* Check resource type */
	if policy.ResourceType != resourceType && policy.ResourceType != "*" {
		return false
	}

	/* Check resource ID (wildcard match if policy resource_id is NULL) */
	if policy.ResourceID != nil {
		if resourceID == nil || *policy.ResourceID != *resourceID {
			return false
		}
	}

	/* Check if action is in permissions list */
	hasPermission := false
	for _, perm := range policy.Permissions {
		if perm == action || perm == "*" {
			hasPermission = true
			break
		}
	}

	if !hasPermission {
		return false
	}

	/* Check ABAC conditions (tags) */
	if policy.Conditions != nil && len(policy.Conditions) > 0 {
		if !e.matchesConditions(policy.Conditions, resourceTags) {
			return false
		}
	}

	return true
}

/* matchesConditions checks if resource tags match policy conditions (ABAC) */
func (e *PolicyEngine) matchesConditions(conditions db.JSONBMap, resourceTags map[string]string) bool {
	/* Extract tags from conditions */
	tagsValue, ok := conditions["tags"]
	if !ok {
		return true /* No tag conditions, allow */
	}

	tagsMap, ok := tagsValue.(map[string]interface{})
	if !ok {
		return true /* Invalid tags format, allow (should log error) */
	}

	/* Check each tag condition */
	for key, expectedValue := range tagsMap {
		expectedStr, ok := expectedValue.(string)
		if !ok {
			continue
		}

		actualValue, exists := resourceTags[key]
		if !exists {
			return false /* Required tag not present */
		}

		if actualValue != expectedStr {
			return false /* Tag value doesn't match */
		}
	}

	return true
}

/* GetAllowedActions returns all actions allowed for a principal on a resource */
func (e *PolicyEngine) GetAllowedActions(ctx context.Context, principalID uuid.UUID, resourceType string, resourceID *string, resourceTags map[string]string) ([]string, error) {
	/* Get all policies for this principal */
	policies, err := e.queries.ListPoliciesByPrincipal(ctx, principalID)
	if err != nil {
		return nil, fmt.Errorf("failed to list policies: %w", err)
	}

	allowedActions := make(map[string]bool)

	/* Collect all allowed actions from matching policies */
	for _, policy := range policies {
		if !e.matchesPolicyBase(policy, resourceType, resourceID, resourceTags) {
			continue
		}

		for _, perm := range policy.Permissions {
			if perm == "*" {
				/* Wildcard permission - return all possible actions */
				return []string{"*"}, nil
			}
			allowedActions[perm] = true
		}
	}

	/* Convert map to slice */
	actions := make([]string, 0, len(allowedActions))
	for action := range allowedActions {
		actions = append(actions, action)
	}

	return actions, nil
}

/* matchesPolicyBase checks if a policy matches resource type, ID, and tags (without checking action) */
func (e *PolicyEngine) matchesPolicyBase(policy db.Policy, resourceType string, resourceID *string, resourceTags map[string]string) bool {
	/* Check resource type */
	if policy.ResourceType != resourceType && policy.ResourceType != "*" {
		return false
	}

	/* Check resource ID */
	if policy.ResourceID != nil {
		if resourceID == nil || *policy.ResourceID != *resourceID {
			return false
		}
	}

	/* Check ABAC conditions */
	if policy.Conditions != nil && len(policy.Conditions) > 0 {
		if !e.matchesConditions(policy.Conditions, resourceTags) {
			return false
		}
	}

	return true
}

/* CreatePolicy creates a new policy */
func (e *PolicyEngine) CreatePolicy(ctx context.Context, principalID uuid.UUID, resourceType string, resourceID *string, permissions []string, conditions map[string]interface{}) (*db.Policy, error) {
	policy := &db.Policy{
		PrincipalID:  principalID,
		ResourceType: resourceType,
		ResourceID:   resourceID,
		Permissions:  permissions,
		Conditions:   conditions,
	}

	if err := e.queries.CreatePolicy(ctx, policy); err != nil {
		return nil, fmt.Errorf("failed to create policy: %w", err)
	}

	return policy, nil
}
