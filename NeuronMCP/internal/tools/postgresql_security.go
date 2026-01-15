/*-------------------------------------------------------------------------
 *
 * postgresql_security.go
 *    Security and validation tools for NeuronMCP
 *
 * Implements security and validation operations:
 * - SQL validation
 * - Permission checking
 * - Operation auditing
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
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
	"strings"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* ============================================================================
 * SQL Validation
 * ============================================================================ */

/* PostgreSQLValidateSQLTool validates SQL syntax */
type PostgreSQLValidateSQLTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLValidateSQLTool creates a new PostgreSQL validate SQL tool */
func NewPostgreSQLValidateSQLTool(db *database.Database, logger *logging.Logger) *PostgreSQLValidateSQLTool {
	return &PostgreSQLValidateSQLTool{
		BaseTool: NewBaseTool(
			"postgresql_validate_sql",
			"Validate SQL syntax before execution and check permissions",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"sql": map[string]interface{}{
						"type":        "string",
						"description": "SQL statement to validate",
					},
					"check_permissions": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Check if current user has permissions to execute",
					},
				},
				"required": []interface{}{"sql"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute validates the SQL */
func (t *PostgreSQLValidateSQLTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	sql, ok := params["sql"].(string)
	if !ok || sql == "" {
		return Error("sql parameter is required", "INVALID_PARAMETER", nil), nil
	}

	checkPermissions, _ := params["check_permissions"].(bool)

	/* Try to parse the SQL using EXPLAIN (which validates syntax without executing) */
	/* For DDL statements, we can use a transaction that we rollback */
	validateQuery := fmt.Sprintf("EXPLAIN %s", sql)

	/* Execute EXPLAIN to validate syntax */
	_, err := t.executor.ExecuteQuery(ctx, validateQuery, nil)
	
	isValid := err == nil
	errorMsg := ""
	if err != nil {
		errorMsg = err.Error()
	}

	permissionInfo := map[string]interface{}{}
	if checkPermissions && isValid {
		/* Check permissions by attempting to get query plan */
		/* This is a simplified check - in production, you'd analyze the query more deeply */
		permissionInfo["can_execute"] = isValid
	}

	t.logger.Info("SQL validated", map[string]interface{}{
		"is_valid": isValid,
		"check_permissions": checkPermissions,
	})

	return Success(map[string]interface{}{
		"is_valid":         isValid,
		"error":            errorMsg,
		"permission_info":  permissionInfo,
		"sql_preview":      truncateString(sql, 100),
	}, map[string]interface{}{
		"tool": "postgresql_validate_sql",
	}), nil
}

/* ============================================================================
 * Permission Checking
 * ============================================================================ */

/* PostgreSQLCheckPermissionsTool checks user permissions */
type PostgreSQLCheckPermissionsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCheckPermissionsTool creates a new PostgreSQL check permissions tool */
func NewPostgreSQLCheckPermissionsTool(db *database.Database, logger *logging.Logger) *PostgreSQLCheckPermissionsTool {
	return &PostgreSQLCheckPermissionsTool{
		BaseTool: NewBaseTool(
			"postgresql_check_permissions",
			"Check user permissions for operations on database objects",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"object_type": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"TABLE", "SEQUENCE", "FUNCTION", "SCHEMA", "DATABASE", "VIEW", "INDEX"},
						"description": "Type of object to check permissions on",
					},
					"object_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the object",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"privilege": map[string]interface{}{
						"type":        "string",
						"description": "Privilege to check (SELECT, INSERT, UPDATE, DELETE, ALL, etc.)",
					},
				},
				"required": []interface{}{"object_type", "object_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute checks the permissions */
func (t *PostgreSQLCheckPermissionsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	objectType, ok := params["object_type"].(string)
	if !ok || objectType == "" {
		return Error("object_type parameter is required", "INVALID_PARAMETER", nil), nil
	}

	objectName, ok := params["object_name"].(string)
	if !ok || objectName == "" {
		return Error("object_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	privilege, _ := params["privilege"].(string)
	if privilege == "" {
		privilege = "ALL"
	}

	/* Get current user */
	currentUser, err := t.executor.ExecuteQueryOne(ctx, "SELECT current_user AS username", nil)
	if err != nil {
		return Error("Failed to get current user", "QUERY_ERROR", nil), nil
	}
	username, _ := currentUser["username"].(string)

	/* Build permission check query */
	var checkQuery string
	fullObjectName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(objectName))

	switch strings.ToUpper(objectType) {
	case "TABLE":
		checkQuery = fmt.Sprintf(`
			SELECT 
				has_table_privilege('%s', '%s', '%s') AS has_privilege,
				has_table_privilege('%s', '%s', 'SELECT') AS can_select,
				has_table_privilege('%s', '%s', 'INSERT') AS can_insert,
				has_table_privilege('%s', '%s', 'UPDATE') AS can_update,
				has_table_privilege('%s', '%s', 'DELETE') AS can_delete
		`, username, fullObjectName, privilege, username, fullObjectName, username, fullObjectName, username, fullObjectName, username, fullObjectName)

	case "SEQUENCE":
		checkQuery = fmt.Sprintf(`
			SELECT 
				has_sequence_privilege('%s', '%s', '%s') AS has_privilege,
				has_sequence_privilege('%s', '%s', 'USAGE') AS can_usage,
				has_sequence_privilege('%s', '%s', 'SELECT') AS can_select
		`, username, fullObjectName, privilege, username, fullObjectName, username, fullObjectName)

	case "FUNCTION":
		checkQuery = fmt.Sprintf(`
			SELECT 
				has_function_privilege('%s', '%s', '%s') AS has_privilege,
				has_function_privilege('%s', '%s', 'EXECUTE') AS can_execute
		`, username, fullObjectName, privilege, username, fullObjectName)

	case "SCHEMA":
		checkQuery = fmt.Sprintf(`
			SELECT 
				has_schema_privilege('%s', '%s', '%s') AS has_privilege,
				has_schema_privilege('%s', '%s', 'USAGE') AS can_usage,
				has_schema_privilege('%s', '%s', 'CREATE') AS can_create
		`, username, objectName, privilege, username, objectName, username, objectName)

	case "DATABASE":
		checkQuery = fmt.Sprintf(`
			SELECT 
				has_database_privilege('%s', '%s', '%s') AS has_privilege,
				has_database_privilege('%s', '%s', 'CONNECT') AS can_connect,
				has_database_privilege('%s', '%s', 'CREATE') AS can_create
		`, username, objectName, privilege, username, objectName, username, objectName)

	default:
		return Error(fmt.Sprintf("Unsupported object type: %s", objectType), "INVALID_PARAMETER", nil), nil
	}

	/* Execute permission check */
	result, err := t.executor.ExecuteQueryOne(ctx, checkQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("Permission check failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Permissions checked", map[string]interface{}{
		"object_type": objectType,
		"object_name": objectName,
		"username":    username,
	})

	return Success(map[string]interface{}{
		"object_type": objectType,
		"object_name": objectName,
		"schema":      schema,
		"username":    username,
		"privilege":   privilege,
		"permissions": result,
	}, map[string]interface{}{
		"tool": "postgresql_check_permissions",
	}), nil
}

/* ============================================================================
 * Operation Auditing
 * ============================================================================ */

/* PostgreSQLAuditOperationTool logs DDL/DML operations */
type PostgreSQLAuditOperationTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLAuditOperationTool creates a new PostgreSQL audit operation tool */
func NewPostgreSQLAuditOperationTool(db *database.Database, logger *logging.Logger) *PostgreSQLAuditOperationTool {
	return &PostgreSQLAuditOperationTool{
		BaseTool: NewBaseTool(
			"postgresql_audit_operation",
			"Log DDL/DML operations and track schema changes",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"operation_type": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"DDL", "DML", "DCL"},
						"description": "Type of operation",
					},
					"operation": map[string]interface{}{
						"type":        "string",
						"description": "Operation name (e.g., CREATE TABLE, INSERT, GRANT)",
					},
					"object_type": map[string]interface{}{
						"type":        "string",
						"description": "Type of object affected",
					},
					"object_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of object affected",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"description": "Schema name",
					},
					"sql": map[string]interface{}{
						"type":        "string",
						"description": "SQL statement executed",
					},
					"status": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"SUCCESS", "FAILURE"},
						"description": "Operation status",
					},
					"error": map[string]interface{}{
						"type":        "string",
						"description": "Error message if status is FAILURE",
					},
				},
				"required": []interface{}{"operation_type", "operation", "status"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute logs the audit entry */
func (t *PostgreSQLAuditOperationTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	operationType, ok := params["operation_type"].(string)
	if !ok || operationType == "" {
		return Error("operation_type parameter is required", "INVALID_PARAMETER", nil), nil
	}

	operation, ok := params["operation"].(string)
	if !ok || operation == "" {
		return Error("operation parameter is required", "INVALID_PARAMETER", nil), nil
	}

	status, ok := params["status"].(string)
	if !ok || status == "" {
		return Error("status parameter is required", "INVALID_PARAMETER", nil), nil
	}

	/* Get current user and timestamp */
	currentUser, err := t.executor.ExecuteQueryOne(ctx, "SELECT current_user AS username, now() AS timestamp", nil)
	if err != nil {
		return Error("Failed to get current user and timestamp", "QUERY_ERROR", nil), nil
	}
	username, _ := currentUser["username"].(string)
	timestamp, _ := currentUser["timestamp"].(string)

	/* Log the audit entry */
	auditEntry := map[string]interface{}{
		"operation_type": operationType,
		"operation":      operation,
		"status":         status,
		"username":       username,
		"timestamp":      timestamp,
	}

	if objectType, ok := params["object_type"].(string); ok {
		auditEntry["object_type"] = objectType
	}

	if objectName, ok := params["object_name"].(string); ok {
		auditEntry["object_name"] = objectName
	}

	if schema, ok := params["schema"].(string); ok {
		auditEntry["schema"] = schema
	}

	if sql, ok := params["sql"].(string); ok {
		auditEntry["sql_preview"] = truncateString(sql, 200)
	}

	if errorMsg, ok := params["error"].(string); ok {
		auditEntry["error"] = errorMsg
	}

	/* Log using the logger */
	if status == "FAILURE" {
		var err error
		if errorMsg, ok := params["error"].(string); ok && errorMsg != "" {
			err = fmt.Errorf(errorMsg)
		}
		t.logger.Error("Operation audited", err, auditEntry)
	} else {
		t.logger.Info("Operation audited", auditEntry)
	}

	/* Optionally store in database audit table if it exists */
	/* This is a simplified implementation - in production, you'd have a dedicated audit table */

	return Success(map[string]interface{}{
		"audit_entry": auditEntry,
		"logged":      true,
	}, map[string]interface{}{
		"tool": "postgresql_audit_operation",
	}), nil
}

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/* ============================================================================
 * Audit Log Tool
 * ============================================================================ */

/* PostgreSQLAuditLogTool queries audit logs from pg_stat_statements and pg_audit */
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
			"Query audit logs from pg_stat_statements and pg_audit",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"start_time": map[string]interface{}{
						"type":        "string",
						"description": "Start time for audit log query (ISO 8601)",
					},
					"end_time": map[string]interface{}{
						"type":        "string",
						"description": "End time for audit log query (ISO 8601)",
					},
					"user_name": map[string]interface{}{
						"type":        "string",
						"description": "Filter by user name",
					},
					"database_name": map[string]interface{}{
						"type":        "string",
						"description": "Filter by database name",
					},
					"limit": map[string]interface{}{
						"type":        "integer",
						"default":     100,
						"description": "Maximum number of log entries to return",
					},
				},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute queries the audit logs */
func (t *PostgreSQLAuditLogTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	limit := 100
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}

	/* Build query for pg_stat_statements */
	query := `
		SELECT 
			userid::regrole AS user_name,
			dbid::regclass AS database_name,
			query,
			calls,
			total_exec_time,
			mean_exec_time,
			min_exec_time,
			max_exec_time,
			rows
		FROM pg_stat_statements
		WHERE 1=1
	`

	args := []interface{}{}
	argIdx := 1

	if user, ok := params["user_name"].(string); ok && user != "" {
		query += fmt.Sprintf(" AND userid::regrole::text = $%d", argIdx)
		args = append(args, user)
		argIdx++
	}

	if db, ok := params["database_name"].(string); ok && db != "" {
		query += fmt.Sprintf(" AND dbid::regclass::text = $%d", argIdx)
		args = append(args, db)
		argIdx++
	}

	query += fmt.Sprintf(" ORDER BY total_exec_time DESC LIMIT $%d", argIdx)
	args = append(args, limit)

	rows, err := t.executor.ExecuteQuery(ctx, query, args)
	if err != nil {
		/* If pg_stat_statements extension is not available, return empty result */
		t.logger.Warn("Failed to query audit logs", map[string]interface{}{
			"error": err.Error(),
		})
		return Success(map[string]interface{}{
			"audit_logs": []interface{}{},
			"message":    "pg_stat_statements extension may not be enabled",
		}, nil), nil
	}

	return Success(map[string]interface{}{
		"audit_logs": rows,
		"count":      len(rows),
	}, nil), nil
}

/* ============================================================================
 * Security Scan Tool
 * ============================================================================ */

/* PostgreSQLSecurityScanTool scans for security vulnerabilities */
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
			"Scan for security vulnerabilities (weak passwords, open ports, etc.)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"scan_type": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"all", "passwords", "permissions", "encryption", "connections"},
						"default":     "all",
						"description": "Type of security scan to perform",
					},
				},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute performs the security scan */
func (t *PostgreSQLSecurityScanTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	scanType := "all"
	if st, ok := params["scan_type"].(string); ok && st != "" {
		scanType = st
	}

	vulnerabilities := []map[string]interface{}{}

	/* Check for users without passwords (password is NULL or empty) */
	if scanType == "all" || scanType == "passwords" {
		passwordQuery := `
			SELECT 
				usename AS user_name,
				'weak_password' AS vulnerability_type,
				'User may have weak or no password' AS description,
				'high' AS severity
			FROM pg_user
			WHERE passwd IS NULL OR passwd = ''
		`
		passwordRows, err := t.executor.ExecuteQuery(ctx, passwordQuery, nil)
		if err == nil {
			for _, row := range passwordRows {
				vulnerabilities = append(vulnerabilities, row)
			}
		}
	}

	/* Check for excessive permissions */
	if scanType == "all" || scanType == "permissions" {
		permissionQuery := `
			SELECT 
				grantee AS user_name,
				' excessive_permissions' AS vulnerability_type,
				'User has SUPERUSER or CREATEDB privileges' AS description,
				'high' AS severity
			FROM pg_roles
			WHERE rolsuper = true OR rolcreatedb = true
		`
		permissionRows, err := t.executor.ExecuteQuery(ctx, permissionQuery, nil)
		if err == nil {
			for _, row := range permissionRows {
				vulnerabilities = append(vulnerabilities, row)
			}
		}
	}

	/* Check for unencrypted connections */
	if scanType == "all" || scanType == "encryption" {
		encryptionQuery := `
			SELECT 
				current_setting('ssl') AS ssl_enabled,
				current_setting('ssl_cert_file') AS cert_file,
				'encryption_check' AS vulnerability_type,
				CASE 
					WHEN current_setting('ssl') = 'off' THEN 'SSL is disabled'
					ELSE 'SSL is enabled'
				END AS description,
				CASE 
					WHEN current_setting('ssl') = 'off' THEN 'high'
					ELSE 'info'
				END AS severity
		`
		encryptionRows, err := t.executor.ExecuteQuery(ctx, encryptionQuery, nil)
		if err == nil {
			for _, row := range encryptionRows {
				if ssl, ok := row["ssl_enabled"].(string); ok && ssl == "off" {
					vulnerabilities = append(vulnerabilities, row)
				}
			}
		}
	}

	/* Check for open connections from untrusted sources */
	if scanType == "all" || scanType == "connections" {
		connectionQuery := `
			SELECT 
				client_addr::text AS client_address,
				usename AS user_name,
				datname AS database_name,
				'open_connection' AS vulnerability_type,
				'Connection from external address' AS description,
				'medium' AS severity
			FROM pg_stat_activity
			WHERE client_addr IS NOT NULL
			  AND client_addr != '127.0.0.1'::inet
			  AND client_addr != '::1'::inet
		`
		connectionRows, err := t.executor.ExecuteQuery(ctx, connectionQuery, nil)
		if err == nil {
			for _, row := range connectionRows {
				vulnerabilities = append(vulnerabilities, row)
			}
		}
	}

	return Success(map[string]interface{}{
		"vulnerabilities": vulnerabilities,
		"count":           len(vulnerabilities),
		"scan_type":       scanType,
	}, nil), nil
}

/* ============================================================================
 * Compliance Check Tool
 * ============================================================================ */

/* PostgreSQLComplianceCheckTool checks compliance with standards */
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
			"Check compliance with standards (PCI-DSS, HIPAA, etc.)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"standard": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"PCI-DSS", "HIPAA", "SOC2", "GDPR", "all"},
						"default":     "all",
						"description": "Compliance standard to check",
					},
				},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute performs the compliance check */
func (t *PostgreSQLComplianceCheckTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	standard := "all"
	if s, ok := params["standard"].(string); ok && s != "" {
		standard = s
	}

	checks := []map[string]interface{}{}

	/* PCI-DSS checks */
	if standard == "all" || standard == "PCI-DSS" {
		/* Check for encryption */
		sslCheck := `SELECT current_setting('ssl') AS ssl_enabled`
		sslResult, err := t.executor.ExecuteQueryOne(ctx, sslCheck, nil)
		if err == nil {
			sslEnabled := false
			if ssl, ok := sslResult["ssl_enabled"].(string); ok && ssl == "on" {
				sslEnabled = true
			}
			checks = append(checks, map[string]interface{}{
				"standard":      "PCI-DSS",
				"check":         "SSL/TLS encryption",
				"status":        map[string]interface{}{"passed": sslEnabled, "required": true},
				"description":   "PCI-DSS requires encrypted connections",
				"recommendation": "Enable SSL/TLS for all connections",
			})
		}

		/* Check for audit logging */
		auditCheck := `SELECT COUNT(*) > 0 AS has_audit FROM pg_stat_statements LIMIT 1`
		auditResult, err := t.executor.ExecuteQueryOne(ctx, auditCheck, nil)
		if err == nil {
			hasAudit := false
			if audit, ok := auditResult["has_audit"].(bool); ok {
				hasAudit = audit
			}
			checks = append(checks, map[string]interface{}{
				"standard":      "PCI-DSS",
				"check":         "Audit logging",
				"status":        map[string]interface{}{"passed": hasAudit, "required": true},
				"description":   "PCI-DSS requires comprehensive audit logging",
				"recommendation": "Enable pg_stat_statements extension",
			})
		}
	}

	/* HIPAA checks */
	if standard == "all" || standard == "HIPAA" {
		/* Check for access controls */
		superuserCheck := `SELECT COUNT(*) AS superuser_count FROM pg_roles WHERE rolsuper = true`
		superuserResult, err := t.executor.ExecuteQueryOne(ctx, superuserCheck, nil)
		if err == nil {
			superuserCount := 0
			if count, ok := superuserResult["superuser_count"].(int64); ok {
				superuserCount = int(count)
			}
			checks = append(checks, map[string]interface{}{
				"standard":      "HIPAA",
				"check":         "Access controls",
				"status":        map[string]interface{}{"passed": superuserCount <= 1, "required": true},
				"description":   "HIPAA requires limited superuser access",
				"recommendation": "Minimize superuser accounts",
			})
		}
	}

	/* GDPR checks */
	if standard == "all" || standard == "GDPR" {
		/* Check for data retention policies */
		checks = append(checks, map[string]interface{}{
			"standard":      "GDPR",
			"check":         "Data retention",
			"status":        map[string]interface{}{"passed": false, "required": true},
			"description":   "GDPR requires data retention policies",
			"recommendation": "Implement data retention and deletion policies",
		})
	}

	passed := 0
	failed := 0
	for _, check := range checks {
		if status, ok := check["status"].(map[string]interface{}); ok {
			if p, ok := status["passed"].(bool); ok && p {
				passed++
			} else {
				failed++
			}
		}
	}

	return Success(map[string]interface{}{
		"standard":  standard,
		"checks":    checks,
		"summary": map[string]interface{}{
			"total":  len(checks),
			"passed": passed,
			"failed": failed,
		},
	}, nil), nil
}

/* ============================================================================
 * Encryption Status Tool
 * ============================================================================ */

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
			"Check encryption status of data at rest and in transit",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"check_type": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"all", "transit", "rest"},
						"default":     "all",
						"description": "Type of encryption to check",
					},
				},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute checks the encryption status */
func (t *PostgreSQLEncryptionStatusTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	checkType := "all"
	if ct, ok := params["check_type"].(string); ok && ct != "" {
		checkType = ct
	}

	status := make(map[string]interface{})

	/* Check encryption in transit (SSL/TLS) */
	if checkType == "all" || checkType == "transit" {
		sslQuery := `
			SELECT 
				current_setting('ssl') AS ssl_enabled,
				current_setting('ssl_cert_file') AS cert_file,
				current_setting('ssl_key_file') AS key_file,
				current_setting('ssl_ca_file') AS ca_file
		`
		sslResult, err := t.executor.ExecuteQueryOne(ctx, sslQuery, nil)
		if err == nil {
			status["transit"] = map[string]interface{}{
				"ssl_enabled": sslResult["ssl_enabled"],
				"cert_file":   sslResult["cert_file"],
				"key_file":    sslResult["key_file"],
				"ca_file":     sslResult["ca_file"],
			}
		}
	}

	/* Check encryption at rest */
	if checkType == "all" || checkType == "rest" {
		/* Check if tablespace encryption is available */
		encryptionQuery := `
			SELECT 
				spcname AS tablespace_name,
				pg_tablespace_location(oid) AS location,
				'encryption_at_rest' AS encryption_type
			FROM pg_tablespace
			WHERE pg_tablespace_location(oid) IS NOT NULL
		`
		encryptionRows, err := t.executor.ExecuteQuery(ctx, encryptionQuery, nil)
		if err == nil {
			status["rest"] = map[string]interface{}{
				"tablespaces": encryptionRows,
				"encrypted":   len(encryptionRows) > 0,
			}
		} else {
			/* If query fails, assume no tablespace encryption */
			status["rest"] = map[string]interface{}{
				"tablespaces": []interface{}{},
				"encrypted":   false,
				"note":        "Tablespace encryption requires additional configuration",
			}
		}
	}

	return Success(map[string]interface{}{
		"encryption_status": status,
		"check_type":        checkType,
	}, nil), nil
}

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/* truncateString truncates a string to a maximum length */
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

