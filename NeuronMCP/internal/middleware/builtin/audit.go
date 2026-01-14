/*-------------------------------------------------------------------------
 *
 * audit.go
 *    Audit middleware for NeuronMCP
 *
 * Provides request-level audit logging with request_id tracking.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/builtin/audit.go
 *
 *-------------------------------------------------------------------------
 */

package builtin

import (
	"context"
	"time"

	"github.com/neurondb/NeuronMCP/internal/audit"
	"github.com/neurondb/NeuronMCP/internal/middleware"
)

/* AuditMiddleware provides audit logging */
type AuditMiddleware struct {
	auditLogger *audit.Logger
}

/* NewAuditMiddleware creates a new audit middleware */
func NewAuditMiddleware(auditLogger *audit.Logger) *AuditMiddleware {
	return &AuditMiddleware{
		auditLogger: auditLogger,
	}
}

/* Name returns the middleware name */
func (m *AuditMiddleware) Name() string {
	return "audit"
}

/* Order returns the execution order - should run early */
func (m *AuditMiddleware) Order() int {
	return 1
}

/* Enabled returns whether the middleware is enabled */
func (m *AuditMiddleware) Enabled() bool {
	return m.auditLogger != nil
}

/* Execute executes the middleware */
func (m *AuditMiddleware) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	if m.auditLogger == nil {
		return next(ctx, req)
	}

	/* Create request context with audit info */
	reqCtx := audit.NewRequestContext(ctx)
	ctx = context.WithValue(ctx, "request_id", reqCtx.RequestID)
	ctx = context.WithValue(ctx, "audit_context", reqCtx)

	/* Extract tool name from request */
	toolName := ""
	if req.Method == "tools/call" {
		if name, ok := req.Params["name"].(string); ok {
			toolName = name
		}
	}

	/* Extract metadata */
	ipAddress := ""
	if ip, ok := req.Metadata["ip_address"].(string); ok {
		ipAddress = ip
	}

	startTime := time.Now()
	resp, err := next(ctx, req)
	duration := time.Since(startTime)

	/* Build audit entry */
	entry := audit.AuditEntry{
		RequestID: reqCtx.RequestID,
		Timestamp: startTime,
		Method:    req.Method,
		ToolName:  toolName,
		UserID:    reqCtx.UserID,
		OrgID:     reqCtx.OrgID,
		ProjectID: reqCtx.ProjectID,
		IPAddress: ipAddress,
		Scopes:    reqCtx.Scopes,
		Duration:  duration,
		Metadata: map[string]interface{}{
			"params": req.Params,
		},
	}

	if err != nil {
		entry.Status = "error"
		entry.Error = err.Error()
	} else if resp != nil && resp.IsError {
		entry.Status = "error"
		if len(resp.Content) > 0 {
			entry.Error = resp.Content[0].Text
		}
	} else {
		entry.Status = "success"
	}

	m.auditLogger.LogRequest(entry)

	/* Add request_id to response metadata */
	if resp != nil && resp.Metadata == nil {
		resp.Metadata = make(map[string]interface{})
	}
	if resp != nil {
		resp.Metadata["request_id"] = reqCtx.RequestID
	}

	return resp, err
}












