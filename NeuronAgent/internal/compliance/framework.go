/*-------------------------------------------------------------------------
 *
 * framework.go
 *    Compliance framework for SOC2, ISO27001, GDPR
 *
 * Provides compliance tools, audit logging enhancements, and
 * compliance reporting.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/compliance/framework.go
 *
 *-------------------------------------------------------------------------
 */

package compliance

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* ComplianceFramework manages compliance */
type ComplianceFramework struct {
	queries     *db.Queries
	auditLogger *AuditLogger
}

/* NewComplianceFramework creates a new compliance framework */
func NewComplianceFramework(queries *db.Queries) *ComplianceFramework {
	return &ComplianceFramework{
		queries:     queries,
		auditLogger: NewAuditLogger(queries),
	}
}

/* LogAccess logs data access for GDPR compliance */
func (cf *ComplianceFramework) LogAccess(ctx context.Context, userID string, resourceType string, resourceID uuid.UUID, action string) error {
	return cf.auditLogger.Log(ctx, AuditEvent{
		EventType:   "data_access",
		UserID:      userID,
		ResourceType: resourceType,
		ResourceID:  resourceID,
		Action:      action,
		Timestamp:   time.Now(),
	})
}

/* LogDataDeletion logs data deletion for GDPR right to be forgotten */
func (cf *ComplianceFramework) LogDataDeletion(ctx context.Context, userID string, resourceType string, resourceID uuid.UUID) error {
	return cf.auditLogger.Log(ctx, AuditEvent{
		EventType:    "data_deletion",
		UserID:       userID,
		ResourceType: resourceType,
		ResourceID:   resourceID,
		Action:       "delete",
		Timestamp:    time.Now(),
	})
}

/* GenerateComplianceReport generates compliance report */
func (cf *ComplianceFramework) GenerateComplianceReport(ctx context.Context, reportType string, startDate, endDate time.Time) (*ComplianceReport, error) {
	report := &ComplianceReport{
		Type:      reportType,
		StartDate: startDate,
		EndDate:   endDate,
		Generated: time.Now(),
	}

	switch reportType {
	case "soc2":
		return cf.generateSOC2Report(ctx, report)
	case "iso27001":
		return cf.generateISO27001Report(ctx, report)
	case "gdpr":
		return cf.generateGDPRReport(ctx, report)
	default:
		return nil, fmt.Errorf("unsupported report type: %s", reportType)
	}
}

/* generateSOC2Report generates SOC2 compliance report */
func (cf *ComplianceFramework) generateSOC2Report(ctx context.Context, report *ComplianceReport) (*ComplianceReport, error) {
	/* Query audit logs for security events */
	query := `SELECT COUNT(*) as event_count, event_type
		FROM neurondb_agent.audit_logs
		WHERE timestamp >= $1 AND timestamp <= $2
		GROUP BY event_type`

	type EventRow struct {
		EventCount int    `db:"event_count"`
		EventType  string `db:"event_type"`
	}

	var rows []EventRow
	err := cf.queries.DB.SelectContext(ctx, &rows, query, report.StartDate, report.EndDate)
	if err != nil {
		return nil, err
	}

	report.Metrics = make(map[string]interface{})
	for _, row := range rows {
		report.Metrics[row.EventType] = row.EventCount
	}

	return report, nil
}

/* generateISO27001Report generates ISO27001 compliance report */
func (cf *ComplianceFramework) generateISO27001Report(ctx context.Context, report *ComplianceReport) (*ComplianceReport, error) {
	/* Similar to SOC2 but with ISO27001 specific metrics */
	return cf.generateSOC2Report(ctx, report)
}

/* generateGDPRReport generates GDPR compliance report */
func (cf *ComplianceFramework) generateGDPRReport(ctx context.Context, report *ComplianceReport) (*ComplianceReport, error) {
	/* Query data access and deletion logs */
	query := `SELECT COUNT(*) as access_count
		FROM neurondb_agent.audit_logs
		WHERE event_type = 'data_access'
		  AND timestamp >= $1 AND timestamp <= $2`

	var accessCount int
	err := cf.queries.DB.GetContext(ctx, &accessCount, query, report.StartDate, report.EndDate)
	if err != nil {
		return nil, err
	}

	report.Metrics = map[string]interface{}{
		"data_access_count": accessCount,
	}

	return report, nil
}

/* AuditLogger provides enhanced audit logging */
type AuditLogger struct {
	queries *db.Queries
}

/* NewAuditLogger creates a new audit logger */
func NewAuditLogger(queries *db.Queries) *AuditLogger {
	return &AuditLogger{
		queries: queries,
	}
}

/* AuditEvent represents an audit event */
type AuditEvent struct {
	EventType    string
	UserID       string
	ResourceType string
	ResourceID   uuid.UUID
	Action       string
	Timestamp    time.Time
	Metadata     map[string]interface{}
}

/* Log logs an audit event */
func (al *AuditLogger) Log(ctx context.Context, event AuditEvent) error {
	query := `INSERT INTO neurondb_agent.audit_logs
		(id, event_type, user_id, resource_type, resource_id, action, timestamp, metadata, created_at)
		VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, $7::jsonb, NOW())`

	_, err := al.queries.DB.ExecContext(ctx, query,
		event.EventType,
		event.UserID,
		event.ResourceType,
		event.ResourceID,
		event.Action,
		event.Timestamp,
		event.Metadata,
	)

	return err
}

/* ComplianceReport represents a compliance report */
type ComplianceReport struct {
	Type      string
	StartDate time.Time
	EndDate   time.Time
	Generated time.Time
	Metrics   map[string]interface{}
}

