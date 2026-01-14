/*-------------------------------------------------------------------------
 *
 * scoped_auth.go
 *    Scoped authentication middleware for NeuronMCP
 *
 * Provides per-tool scope checking for fine-grained access control.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/builtin/scoped_auth.go
 *
 *-------------------------------------------------------------------------
 */

package builtin

import (
	"context"
	"fmt"
	"strings"

	"github.com/neurondb/NeuronMCP/internal/middleware"
)

/* ScopedAuthMiddleware provides scoped authentication */
type ScopedAuthMiddleware struct {
	scopeChecker ScopeChecker
	enabled      bool
}

/* NewScopedAuthMiddleware creates a new scoped auth middleware */
func NewScopedAuthMiddleware(scopeChecker ScopeChecker) *ScopedAuthMiddleware {
	return &ScopedAuthMiddleware{
		scopeChecker: scopeChecker,
		enabled:      scopeChecker != nil,
	}
}

/* Name returns the middleware name */
func (m *ScopedAuthMiddleware) Name() string {
	return "scoped_auth"
}

/* Order returns the execution order - should run after auth but before execution */
func (m *ScopedAuthMiddleware) Order() int {
	return 1
}

/* Enabled returns whether the middleware is enabled */
func (m *ScopedAuthMiddleware) Enabled() bool {
	return m.enabled
}

/* GetRequiredScope returns the required scope for a tool */
func GetRequiredScope(toolName string) string {
	/* Map tool names to required scopes */
	/* Read operations */
	if strings.HasPrefix(toolName, "list_") || strings.HasPrefix(toolName, "get_") || 
	   strings.HasPrefix(toolName, "vector_search") || strings.HasPrefix(toolName, "retrieve_") {
		return "read"
	}

	/* Write operations */
	if strings.HasPrefix(toolName, "create_") || strings.HasPrefix(toolName, "insert_") ||
	   strings.HasPrefix(toolName, "update_") || strings.HasPrefix(toolName, "process_") {
		return "write"
	}

	/* Training/ML operations */
	if strings.HasPrefix(toolName, "train_") || strings.HasPrefix(toolName, "predict_") {
		return "train"
	}

	/* Delete operations */
	if strings.HasPrefix(toolName, "delete_") || strings.HasPrefix(toolName, "drop_") {
		return "delete"
	}

	/* Default: require read scope */
	return "read"
}

/* Execute executes the middleware */
func (m *ScopedAuthMiddleware) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	if !m.enabled || m.scopeChecker == nil {
		return next(ctx, req)
	}

	/* Only check scopes for tool calls */
	if req.Method != "tools/call" {
		return next(ctx, req)
	}

	/* Extract tool name */
	toolName, ok := req.Params["name"].(string)
	if !ok || toolName == "" {
		return next(ctx, req) // Let other validation handle this
	}

	/* Get user ID from context */
	userID, ok := ctx.Value("user_id").(string)
	if !ok || userID == "" {
		/* No user ID - allow through (might be handled by other auth middleware) */
		return next(ctx, req)
	}

	/* Get required scope for the tool */
	requiredScope := GetRequiredScope(toolName)

	/* Check if user has the required scope */
	/* Use the scope checker from scope_checker.go which uses context */
	if !m.scopeChecker.HasScope(ctx, requiredScope) {
		userScopes := m.scopeChecker.GetUserScopes(ctx)
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{
					Type: "text",
					Text: fmt.Sprintf("Access denied: tool '%s' requires scope '%s', but user only has scopes: %v", toolName, requiredScope, userScopes),
				},
			},
			IsError: true,
			Metadata: map[string]interface{}{
				"error_code": "INSUFFICIENT_SCOPE",
				"tool":       toolName,
				"required":   requiredScope,
				"user_scopes": userScopes,
			},
		}, nil
	}

	/* Add scopes to context for audit logging */
	if scopes := m.scopeChecker.GetUserScopes(ctx); len(scopes) > 0 {
		ctx = context.WithValue(ctx, "scopes", scopes)
	}

	return next(ctx, req)
}




