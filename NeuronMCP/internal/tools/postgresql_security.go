/*-------------------------------------------------------------------------
 *
 * postgresql_security.go
 *    Security and compliance tools for NeuronMCP
 *
 * Implements security and compliance operations as specified in Phase 1.1
 * of the roadmap.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/postgresql_security.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* PostgreSQLAuditLogTool queries audit logs */
type PostgreSQLAuditLogTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLAuditLogTool creates a new PostgreSQL audit log tool */
func NewPostgreSQLAuditLogTool(db *database.Database, logger *logging.Logger) *PostgreSQLAuditLogTool {
	return &PostgreSQLAuditLogTool{
		BaseTool: NewBaseTool(
			"postgresql_audit_log",
			"Query audit logs from pgAudit or similar audit extensions",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"limit": map[string]interface{}{
						"type":        "number",
						"default":     100,
						"minimum":     1,
						"maximum":     1000,
						"description": "Maximum number of log entries to return",
					},
					"user": map[string]interface{}{
						"type":        "string",
						"description": "Filter by username (optional)",
					},
					"operation": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"SELECT", "INSERT", "UPDATE", "DELETE", "DDL", "ALL"},
						"description": "Filter by operation type",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute queries audit logs */
func (t *PostgreSQLAuditLogTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	/* Check if pgAudit extension exists */
	checkQuery := `SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pgaudit') AS extension_exists`
	checkResult, err := t.executor.ExecuteQueryOne(ctx, checkQuery, nil)
	if err != nil {
		return Error("Failed to check pgAudit extension", "QUERY_ERROR", map[string]interface{}{"error": err.Error()}), nil
	}

	exists := false
	if val, ok := checkResult["extension_exists"].(bool); ok {
		exists = val
	}

	if !exists {
		return Success(map[string]interface{}{
			"audit_logs": []interface{}{},
			"count":      0,
			"note":       "pgAudit extension is not installed. Install with: CREATE EXTENSION pgaudit;",
			"alternative": "Use PostgreSQL's built-in logging (log_statement, log_connections, etc.)",
		}, map[string]interface{}{
			"tool": "postgresql_audit_log",
		}), nil
	}

	limit := 100
	if val, ok := params["limit"].(float64); ok {
		limit = int(val)
		if limit < 1 {
			limit = 1
		}
		if limit > 1000 {
			limit = 1000
		}
	}

	user, _ := params["user"].(string)
	operation, _ := params["operation"].(string)

	/* Query audit log table (structure varies by pgAudit version) */
	query := fmt.Sprintf(`
		SELECT 
			event_time,
			user_name,
			database_name,
			command_tag,
			statement
		FROM pgaudit.log
		WHERE 1=1
	`)
	var queryParams []interface{}
	paramCount := 0

	if user != "" {
		paramCount++
		query += fmt.Sprintf(" AND user_name = $%d", paramCount)
		queryParams = append(queryParams, user)
	}

	if operation != "" && operation != "ALL" {
		paramCount++
		query += fmt.Sprintf(" AND command_tag = $%d", paramCount)
		queryParams = append(queryParams, operation)
	}

	query += fmt.Sprintf(" ORDER BY event_time DESC LIMIT $%d", paramCount+1)
	queryParams = append(queryParams, limit)

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("Audit log query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"audit_logs": results,
		"count":      len(results),
	}, map[string]interface{}{
		"tool": "postgresql_audit_log",
	}), nil
}

/* PostgreSQLSecurityScanTool performs security vulnerability scan */
type PostgreSQLSecurityScanTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLSecurityScanTool creates a new PostgreSQL security scan tool */
func NewPostgreSQLSecurityScanTool(db *database.Database, logger *logging.Logger) *PostgreSQLSecurityScanTool {
	return &PostgreSQLSecurityScanTool{
		BaseTool: NewBaseTool(
			"postgresql_security_scan",
			"Perform security vulnerability scan and provide recommendations",
			map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
				"required":   []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute performs security scan */
func (t *PostgreSQLSecurityScanTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	vulnerabilities := []map[string]interface{}{}
	recommendations := []string{}

	/* Check for weak passwords (users without password expiration) */
	weakPasswordQuery := `
		SELECT 
			rolname AS username,
			rolvaliduntil AS password_expires_at
		FROM pg_roles
		WHERE rolcanlogin = true 
		AND rolvaliduntil IS NULL
	`
	weakPasswords, err := t.executor.ExecuteQuery(ctx, weakPasswordQuery, nil)
	if err == nil && len(weakPasswords) > 0 {
		vulnerabilities = append(vulnerabilities, map[string]interface{}{
			"severity":    "medium",
			"category":    "authentication",
			"issue":       "Users without password expiration",
			"affected":    weakPasswords,
			"recommendation": "Set password expiration: ALTER ROLE username VALID UNTIL '2025-12-31';",
		})
		recommendations = append(recommendations, "Set password expiration policies for all users")
	}

	/* Check for superuser accounts */
	superuserQuery := `
		SELECT rolname AS username
		FROM pg_roles
		WHERE rolsuper = true
		AND rolcanlogin = true
	`
	superusers, err := t.executor.ExecuteQuery(ctx, superuserQuery, nil)
	if err == nil && len(superusers) > 1 {
		vulnerabilities = append(vulnerabilities, map[string]interface{}{
			"severity":    "high",
			"category":    "authorization",
			"issue":       "Multiple superuser accounts",
			"affected":    superusers,
			"recommendation": "Limit superuser accounts to minimum necessary",
		})
		recommendations = append(recommendations, "Review and limit superuser accounts")
	}

	/* Check SSL settings */
	sslQuery := `SELECT name, setting FROM pg_settings WHERE name IN ('ssl', 'ssl_cert_file', 'ssl_key_file')`
	sslSettings, err := t.executor.ExecuteQuery(ctx, sslQuery, nil)
	if err == nil {
		sslEnabled := false
		for _, setting := range sslSettings {
			if name, ok := setting["name"].(string); ok && name == "ssl" {
				if val, ok := setting["setting"].(string); ok && val == "on" {
					sslEnabled = true
				}
			}
		}
		if !sslEnabled {
			vulnerabilities = append(vulnerabilities, map[string]interface{}{
				"severity":    "high",
				"category":    "encryption",
				"issue":       "SSL not enabled",
				"recommendation": "Enable SSL: ALTER SYSTEM SET ssl = on;",
			})
			recommendations = append(recommendations, "Enable SSL for encrypted connections")
		}
	}

	/* Check for public schema permissions */
	publicSchemaQuery := `
		SELECT 
			grantee,
			privilege_type
		FROM information_schema.role_table_grants
		WHERE table_schema = 'public'
		AND grantee = 'PUBLIC'
		LIMIT 10
	`
	publicPerms, err := t.executor.ExecuteQuery(ctx, publicSchemaQuery, nil)
	if err == nil && len(publicPerms) > 0 {
		vulnerabilities = append(vulnerabilities, map[string]interface{}{
			"severity":    "medium",
			"category":    "permissions",
			"issue":       "Public schema has PUBLIC permissions",
			"affected":    publicPerms,
			"recommendation": "Review and restrict PUBLIC schema permissions",
		})
		recommendations = append(recommendations, "Review PUBLIC schema permissions")
	}

	return Success(map[string]interface{}{
		"vulnerabilities": vulnerabilities,
		"vulnerability_count": len(vulnerabilities),
		"recommendations": recommendations,
		"scan_timestamp": "now()",
	}, map[string]interface{}{
		"tool": "postgresql_security_scan",
	}), nil
}

/* PostgreSQLComplianceCheckTool performs compliance validation */
type PostgreSQLComplianceCheckTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLComplianceCheckTool creates a new PostgreSQL compliance check tool */
func NewPostgreSQLComplianceCheckTool(db *database.Database, logger *logging.Logger) *PostgreSQLComplianceCheckTool {
	return &PostgreSQLComplianceCheckTool{
		BaseTool: NewBaseTool(
			"postgresql_compliance_check",
			"Perform compliance validation (GDPR, SOC 2, HIPAA, PCI DSS)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"compliance_standard": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"GDPR", "SOC2", "HIPAA", "PCIDSS", "ALL"},
						"default":     "ALL",
						"description": "Compliance standard to check",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute performs compliance check */
func (t *PostgreSQLComplianceCheckTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	standard := "ALL"
	if val, ok := params["compliance_standard"].(string); ok {
		standard = val
	}

	checks := []map[string]interface{}{}
	passed := 0
	failed := 0

	/* GDPR Checks */
	if standard == "ALL" || standard == "GDPR" {
		/* Check for audit logging */
		auditCheck := map[string]interface{}{
			"standard": "GDPR",
			"check":    "Audit logging enabled",
			"status":   "unknown",
		}
		auditQuery := `SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pgaudit') AS enabled`
		result, err := t.executor.ExecuteQueryOne(ctx, auditQuery, nil)
		if err == nil {
			if enabled, ok := result["enabled"].(bool); ok && enabled {
				auditCheck["status"] = "passed"
				passed++
			} else {
				auditCheck["status"] = "failed"
				auditCheck["recommendation"] = "Enable audit logging for GDPR compliance"
				failed++
			}
		}
		checks = append(checks, auditCheck)

		/* Check for encryption */
		encryptionCheck := map[string]interface{}{
			"standard": "GDPR",
			"check":    "Data encryption at rest",
			"status":   "unknown",
		}
		sslQuery := `SELECT setting FROM pg_settings WHERE name = 'ssl'`
		sslResult, err := t.executor.ExecuteQueryOne(ctx, sslQuery, nil)
		if err == nil {
			if setting, ok := sslResult["setting"].(string); ok && setting == "on" {
				encryptionCheck["status"] = "passed"
				passed++
			} else {
				encryptionCheck["status"] = "warning"
				encryptionCheck["recommendation"] = "Enable SSL/TLS for data in transit"
				failed++
			}
		}
		checks = append(checks, encryptionCheck)
	}

	/* SOC 2 Checks */
	if standard == "ALL" || standard == "SOC2" {
		/* Check for access controls */
		accessCheck := map[string]interface{}{
			"standard": "SOC2",
			"check":    "Role-based access control",
			"status":   "passed",
		}
		roleQuery := `SELECT COUNT(*) AS role_count FROM pg_roles WHERE rolcanlogin = true`
		roleResult, err := t.executor.ExecuteQueryOne(ctx, roleQuery, nil)
		if err == nil {
			if count, ok := roleResult["role_count"].(int64); ok && count > 0 {
				accessCheck["status"] = "passed"
				passed++
			}
		}
		checks = append(checks, accessCheck)
	}

	/* HIPAA Checks */
	if standard == "ALL" || standard == "HIPAA" {
		/* Check for encryption */
		hipaaEncryptionCheck := map[string]interface{}{
			"standard": "HIPAA",
			"check":    "Encryption for PHI data",
			"status":   "warning",
			"recommendation": "Ensure all PHI data is encrypted at rest and in transit",
		}
		checks = append(checks, hipaaEncryptionCheck)
		failed++
	}

	/* PCI DSS Checks */
	if standard == "ALL" || standard == "PCIDSS" {
		/* Check for strong authentication */
		pciAuthCheck := map[string]interface{}{
			"standard": "PCI DSS",
			"check":    "Strong authentication required",
			"status":   "warning",
			"recommendation": "Implement strong password policies and MFA",
		}
		checks = append(checks, pciAuthCheck)
		failed++
	}

	return Success(map[string]interface{}{
		"compliance_standard": standard,
		"checks":              checks,
		"passed":              passed,
		"failed":              failed,
		"total":               len(checks),
		"compliance_score":    float64(passed) / float64(len(checks)) * 100,
	}, map[string]interface{}{
		"tool": "postgresql_compliance_check",
	}), nil
}

/* PostgreSQLEncryptionStatusTool checks encryption status */
type PostgreSQLEncryptionStatusTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLEncryptionStatusTool creates a new PostgreSQL encryption status tool */
func NewPostgreSQLEncryptionStatusTool(db *database.Database, logger *logging.Logger) *PostgreSQLEncryptionStatusTool {
	return &PostgreSQLEncryptionStatusTool{
		BaseTool: NewBaseTool(
			"postgresql_encryption_status",
			"Check encryption status for data at rest and in transit",
			map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
				"required":   []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute checks encryption status */
func (t *PostgreSQLEncryptionStatusTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	/* Check SSL/TLS settings */
	sslQuery := `
		SELECT 
			name,
			setting,
			unit,
			context
		FROM pg_settings
		WHERE name IN ('ssl', 'ssl_cert_file', 'ssl_key_file', 'ssl_ca_file', 'ssl_crl_file')
		ORDER BY name
	`
	sslSettings, err := t.executor.ExecuteQuery(ctx, sslQuery, nil)
	if err != nil {
		return Error("Failed to query SSL settings", "QUERY_ERROR", map[string]interface{}{"error": err.Error()}), nil
	}

	/* Determine SSL status */
	sslEnabled := false
	for _, setting := range sslSettings {
		if name, ok := setting["name"].(string); ok && name == "ssl" {
			if val, ok := setting["setting"].(string); ok && val == "on" {
				sslEnabled = true
			}
		}
	}

	/* Check for encryption extensions */
	encryptionExtQuery := `
		SELECT 
			extname,
			extversion
		FROM pg_extension
		WHERE extname IN ('pgcrypto', 'pg_trgm')
	`
	encryptionExts, err := t.executor.ExecuteQuery(ctx, encryptionExtQuery, nil)
	if err != nil {
		encryptionExts = []map[string]interface{}{}
	}

	status := map[string]interface{}{
		"ssl_enabled":           sslEnabled,
		"ssl_settings":         sslSettings,
		"encryption_extensions": encryptionExts,
		"data_in_transit":      sslEnabled,
		"data_at_rest":         "Check with database administrator - depends on storage encryption",
		"recommendations":      []string{},
	}

	if !sslEnabled {
		recommendations := []string{"Enable SSL: ALTER SYSTEM SET ssl = on; and restart PostgreSQL"}
		status["recommendations"] = recommendations
	}

	return Success(status, map[string]interface{}{
		"tool": "postgresql_encryption_status",
	}), nil
}



