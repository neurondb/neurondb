/*-------------------------------------------------------------------------
 *
 * audit.go
 *    Audit logging for NeuronMCP
 *
 * Provides request-level audit logging with request_id tracking for compliance.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/audit/audit.go
 *
 *-------------------------------------------------------------------------
 */

package audit

import (
	"context"
	"time"

	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* AuditEntry represents a single audit log entry */
type AuditEntry struct {
	RequestID    string                 `json:"request_id"`
	Timestamp    time.Time              `json:"timestamp"`
	Method       string                 `json:"method"`
	ToolName     string                 `json:"tool_name,omitempty"`
	UserID       string                 `json:"user_id,omitempty"`
	OrgID        string                 `json:"org_id,omitempty"`
	ProjectID    string                 `json:"project_id,omitempty"`
	IPAddress    string                 `json:"ip_address,omitempty"`
	Scopes       []string               `json:"scopes,omitempty"`
	Status       string                 `json:"status"` // "success", "error", "denied"
	Error        string                 `json:"error,omitempty"`
	Duration     time.Duration          `json:"duration_ms"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

/* Logger provides audit logging capabilities */
type Logger struct {
	logger *logging.Logger
}

/* NewLogger creates a new audit logger */
func NewLogger(baseLogger *logging.Logger) *Logger {
	return &Logger{
		logger: baseLogger,
	}
}

/* LogRequest logs an audit entry for a request */
func (l *Logger) LogRequest(entry AuditEntry) {
	if l.logger == nil {
		return
	}

	fields := map[string]interface{}{
		"request_id": entry.RequestID,
		"timestamp":  entry.Timestamp,
		"method":     entry.Method,
		"status":     entry.Status,
		"duration_ms": entry.Duration.Milliseconds(),
	}

	if entry.ToolName != "" {
		fields["tool_name"] = entry.ToolName
	}
	if entry.UserID != "" {
		fields["user_id"] = entry.UserID
	}
	if entry.OrgID != "" {
		fields["org_id"] = entry.OrgID
	}
	if entry.ProjectID != "" {
		fields["project_id"] = entry.ProjectID
	}
	if entry.IPAddress != "" {
		fields["ip_address"] = entry.IPAddress
	}
	if len(entry.Scopes) > 0 {
		fields["scopes"] = entry.Scopes
	}
	if entry.Error != "" {
		fields["error"] = entry.Error
	}
	if entry.Metadata != nil {
		for k, v := range entry.Metadata {
			fields[k] = v
		}
	}

	l.logger.Info("Audit log entry", fields)
}

/* GenerateRequestID generates a unique request ID */
func GenerateRequestID() string {
	/* Simple implementation - in production, use UUID */
	return time.Now().Format("20060102150405.000000")
}

/* RequestContext provides request context with audit information */
type RequestContext struct {
	RequestID string
	UserID    string
	OrgID     string
	ProjectID string
	Scopes    []string
	StartTime time.Time
}

/* NewRequestContext creates a new request context */
func NewRequestContext(ctx context.Context) *RequestContext {
	reqCtx := &RequestContext{
		RequestID: GenerateRequestID(),
		StartTime: time.Now(),
	}

	/* Extract from context if available */
	if userID := ctx.Value("user_id"); userID != nil {
		if id, ok := userID.(string); ok {
			reqCtx.UserID = id
		}
	}
	if orgID := ctx.Value("org_id"); orgID != nil {
		if id, ok := orgID.(string); ok {
			reqCtx.OrgID = id
		}
	}
	if projectID := ctx.Value("project_id"); projectID != nil {
		if id, ok := projectID.(string); ok {
			reqCtx.ProjectID = id
		}
	}
	if scopes := ctx.Value("scopes"); scopes != nil {
		if s, ok := scopes.([]string); ok {
			reqCtx.Scopes = s
		}
	}

	return reqCtx
}

/* Duration returns the duration since the request started */
func (r *RequestContext) Duration() time.Duration {
	return time.Since(r.StartTime)
}












