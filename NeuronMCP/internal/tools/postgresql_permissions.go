/*-------------------------------------------------------------------------
 *
 * postgresql_permissions.go
 *    Permission management tools for NeuronMCP
 *
 * Implements comprehensive permission DCL operations:
 * - GRANT/REVOKE privileges
 * - GRANT/REVOKE role membership
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/postgresql_permissions.go
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
 * Privilege Management Tools
 * ============================================================================ */

/* PostgreSQLGrantTool grants privileges on database objects */
type PostgreSQLGrantTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLGrantTool creates a new PostgreSQL grant tool */
func NewPostgreSQLGrantTool(db *database.Database, logger *logging.Logger) *PostgreSQLGrantTool {
	return &PostgreSQLGrantTool{
		BaseTool: NewBaseTool(
			"postgresql_grant",
			"Grant privileges on tables, sequences, functions, schemas, databases, or columns",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"privileges": map[string]interface{}{
						"type":        "array",
						"description": "List of privileges to grant (e.g., SELECT, INSERT, UPDATE, DELETE, ALL, etc.)",
					},
					"object_type": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"TABLE", "SEQUENCE", "FUNCTION", "SCHEMA", "DATABASE", "LANGUAGE", "TYPE", "DOMAIN"},
						"description": "Type of object to grant privileges on",
					},
					"object_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the object (table, sequence, function, schema, database, etc.)",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name (for tables, sequences, functions)",
					},
					"columns": map[string]interface{}{
						"type":        "array",
						"description": "List of column names for column-level privileges (for tables only)",
					},
					"grantees": map[string]interface{}{
						"type":        "array",
						"description": "List of role/user names to grant privileges to",
					},
					"with_grant_option": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Allow grantees to grant privileges to others",
					},
					"grant_on_all": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Grant on all objects of the specified type in schema",
					},
				},
				"required": []interface{}{"privileges", "grantees"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute grants the privileges */
func (t *PostgreSQLGrantTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	privileges, ok := params["privileges"].([]interface{})
	if !ok || len(privileges) == 0 {
		return Error("privileges parameter is required and must be a non-empty array", "INVALID_PARAMETER", nil), nil
	}

	grantees, ok := params["grantees"].([]interface{})
	if !ok || len(grantees) == 0 {
		return Error("grantees parameter is required and must be a non-empty array", "INVALID_PARAMETER", nil), nil
	}

	/* Build privilege list */
	privList := []string{}
	for _, priv := range privileges {
		if privStr, ok := priv.(string); ok {
			privList = append(privList, strings.ToUpper(privStr))
		}
	}
	privilegesStr := strings.Join(privList, ", ")

	/* Build grantee list */
	granteeList := []string{}
	for _, grantee := range grantees {
		if granteeStr, ok := grantee.(string); ok && granteeStr != "" {
			granteeList = append(granteeList, quoteIdentifier(granteeStr))
		}
	}
	if len(granteeList) == 0 {
		return Error("No valid grantees specified", "INVALID_PARAMETER", nil), nil
	}
	granteesStr := strings.Join(granteeList, ", ")

	/* Build GRANT statement */
	parts := []string{"GRANT", privilegesStr}

	/* Object specification */
	objectType, _ := params["object_type"].(string)
	objectType = strings.ToUpper(objectType)

	if grantOnAll, ok := params["grant_on_all"].(bool); ok && grantOnAll {
		/* GRANT ON ALL ... IN SCHEMA */
		if objectType == "" {
			objectType = "TABLE"
		}
		parts = append(parts, fmt.Sprintf("ON ALL %sS IN SCHEMA", objectType))
		
		schema, _ := params["schema"].(string)
		if schema == "" {
			schema = "public"
		}
		parts = append(parts, quoteIdentifier(schema))
	} else if objectType != "" {
		/* Specific object */
		parts = append(parts, fmt.Sprintf("ON %s", objectType))
		
		objectName, ok := params["object_name"].(string)
		if !ok || objectName == "" {
			return Error("object_name parameter is required when not using grant_on_all", "INVALID_PARAMETER", nil), nil
		}

		/* Handle column-level privileges */
		if columns, ok := params["columns"].([]interface{}); ok && len(columns) > 0 && objectType == "TABLE" {
			colList := []string{}
			for _, col := range columns {
				if colStr, ok := col.(string); ok {
					colList = append(colList, quoteIdentifier(colStr))
				}
			}
			if len(colList) > 0 {
				schema, _ := params["schema"].(string)
				if schema == "" {
					schema = "public"
				}
				parts = append(parts, fmt.Sprintf("%s.%s (%s)", quoteIdentifier(schema), quoteIdentifier(objectName), strings.Join(colList, ", ")))
			} else {
				return Error("No valid columns specified", "INVALID_PARAMETER", nil), nil
			}
		} else {
			/* Full object privileges */
			if objectType == "TABLE" || objectType == "SEQUENCE" || objectType == "FUNCTION" {
				schema, _ := params["schema"].(string)
				if schema == "" {
					schema = "public"
				}
				parts = append(parts, fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(objectName)))
			} else {
				parts = append(parts, quoteIdentifier(objectName))
			}
		}
	} else {
		/* Default to TABLE if no object type specified */
		objectName, ok := params["object_name"].(string)
		if !ok || objectName == "" {
			return Error("object_name parameter is required", "INVALID_PARAMETER", nil), nil
		}
		
		schema, _ := params["schema"].(string)
		if schema == "" {
			schema = "public"
		}
		parts = append(parts, fmt.Sprintf("ON TABLE %s.%s", quoteIdentifier(schema), quoteIdentifier(objectName)))
	}

	parts = append(parts, "TO", granteesStr)

	if withGrantOption, ok := params["with_grant_option"].(bool); ok && withGrantOption {
		parts = append(parts, "WITH GRANT OPTION")
	}

	grantQuery := strings.Join(parts, " ")

	/* Execute GRANT */
	err := t.executor.Exec(ctx, grantQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("GRANT failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Privileges granted", map[string]interface{}{
		"privileges": privilegesStr,
		"grantees":  granteesStr,
	})

	return Success(map[string]interface{}{
		"privileges": privilegesStr,
		"grantees":  granteesStr,
		"query":     grantQuery,
	}, map[string]interface{}{
		"tool": "postgresql_grant",
	}), nil
}

/* PostgreSQLRevokeTool revokes privileges on database objects */
type PostgreSQLRevokeTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLRevokeTool creates a new PostgreSQL revoke tool */
func NewPostgreSQLRevokeTool(db *database.Database, logger *logging.Logger) *PostgreSQLRevokeTool {
	return &PostgreSQLRevokeTool{
		BaseTool: NewBaseTool(
			"postgresql_revoke",
			"Revoke privileges on tables, sequences, functions, schemas, databases, or columns",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"privileges": map[string]interface{}{
						"type":        "array",
						"description": "List of privileges to revoke (e.g., SELECT, INSERT, UPDATE, DELETE, ALL, etc.)",
					},
					"object_type": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"TABLE", "SEQUENCE", "FUNCTION", "SCHEMA", "DATABASE", "LANGUAGE", "TYPE", "DOMAIN"},
						"description": "Type of object to revoke privileges on",
					},
					"object_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the object (table, sequence, function, schema, database, etc.)",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name (for tables, sequences, functions)",
					},
					"columns": map[string]interface{}{
						"type":        "array",
						"description": "List of column names for column-level privileges (for tables only)",
					},
					"revokees": map[string]interface{}{
						"type":        "array",
						"description": "List of role/user names to revoke privileges from",
					},
					"cascade": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Revoke privileges cascaded from granted objects",
					},
					"revoke_grant_option": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Revoke GRANT OPTION only (not the privilege itself)",
					},
					"revoke_on_all": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Revoke on all objects of the specified type in schema",
					},
				},
				"required": []interface{}{"privileges", "revokees"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute revokes the privileges */
func (t *PostgreSQLRevokeTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	privileges, ok := params["privileges"].([]interface{})
	if !ok || len(privileges) == 0 {
		return Error("privileges parameter is required and must be a non-empty array", "INVALID_PARAMETER", nil), nil
	}

	revokees, ok := params["revokees"].([]interface{})
	if !ok || len(revokees) == 0 {
		return Error("revokees parameter is required and must be a non-empty array", "INVALID_PARAMETER", nil), nil
	}

	/* Build privilege list */
	privList := []string{}
	for _, priv := range privileges {
		if privStr, ok := priv.(string); ok {
			privList = append(privList, strings.ToUpper(privStr))
		}
	}
	privilegesStr := strings.Join(privList, ", ")

	/* Build revokee list */
	revokeeList := []string{}
	for _, revokee := range revokees {
		if revokeeStr, ok := revokee.(string); ok && revokeeStr != "" {
			revokeeList = append(revokeeList, quoteIdentifier(revokeeStr))
		}
	}
	if len(revokeeList) == 0 {
		return Error("No valid revokees specified", "INVALID_PARAMETER", nil), nil
	}
	revokeesStr := strings.Join(revokeeList, ", ")

	/* Build REVOKE statement */
	parts := []string{"REVOKE"}

	if revokeGrantOption, ok := params["revoke_grant_option"].(bool); ok && revokeGrantOption {
		parts = append(parts, "GRANT OPTION FOR")
	}

	parts = append(parts, privilegesStr)

	/* Object specification */
	objectType, _ := params["object_type"].(string)
	objectType = strings.ToUpper(objectType)

	if revokeOnAll, ok := params["revoke_on_all"].(bool); ok && revokeOnAll {
		/* REVOKE ON ALL ... IN SCHEMA */
		if objectType == "" {
			objectType = "TABLE"
		}
		parts = append(parts, fmt.Sprintf("ON ALL %sS IN SCHEMA", objectType))
		
		schema, _ := params["schema"].(string)
		if schema == "" {
			schema = "public"
		}
		parts = append(parts, quoteIdentifier(schema))
	} else if objectType != "" {
		/* Specific object */
		parts = append(parts, fmt.Sprintf("ON %s", objectType))
		
		objectName, ok := params["object_name"].(string)
		if !ok || objectName == "" {
			return Error("object_name parameter is required when not using revoke_on_all", "INVALID_PARAMETER", nil), nil
		}

		/* Handle column-level privileges */
		if columns, ok := params["columns"].([]interface{}); ok && len(columns) > 0 && objectType == "TABLE" {
			colList := []string{}
			for _, col := range columns {
				if colStr, ok := col.(string); ok {
					colList = append(colList, quoteIdentifier(colStr))
				}
			}
			if len(colList) > 0 {
				schema, _ := params["schema"].(string)
				if schema == "" {
					schema = "public"
				}
				parts = append(parts, fmt.Sprintf("%s.%s (%s)", quoteIdentifier(schema), quoteIdentifier(objectName), strings.Join(colList, ", ")))
			} else {
				return Error("No valid columns specified", "INVALID_PARAMETER", nil), nil
			}
		} else {
			/* Full object privileges */
			if objectType == "TABLE" || objectType == "SEQUENCE" || objectType == "FUNCTION" {
				schema, _ := params["schema"].(string)
				if schema == "" {
					schema = "public"
				}
				parts = append(parts, fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(objectName)))
			} else {
				parts = append(parts, quoteIdentifier(objectName))
			}
		}
	} else {
		/* Default to TABLE if no object type specified */
		objectName, ok := params["object_name"].(string)
		if !ok || objectName == "" {
			return Error("object_name parameter is required", "INVALID_PARAMETER", nil), nil
		}
		
		schema, _ := params["schema"].(string)
		if schema == "" {
			schema = "public"
		}
		parts = append(parts, fmt.Sprintf("ON TABLE %s.%s", quoteIdentifier(schema), quoteIdentifier(objectName)))
	}

	parts = append(parts, "FROM", revokeesStr)

	if cascade, ok := params["cascade"].(bool); ok && cascade {
		parts = append(parts, "CASCADE")
	} else {
		parts = append(parts, "RESTRICT")
	}

	revokeQuery := strings.Join(parts, " ")

	/* Execute REVOKE */
	err := t.executor.Exec(ctx, revokeQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("REVOKE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Privileges revoked", map[string]interface{}{
		"privileges": privilegesStr,
		"revokees":  revokeesStr,
	})

	return Success(map[string]interface{}{
		"privileges": privilegesStr,
		"revokees":  revokeesStr,
		"query":     revokeQuery,
	}, map[string]interface{}{
		"tool": "postgresql_revoke",
	}), nil
}

/* ============================================================================
 * Role Membership Management Tools
 * ============================================================================ */

/* PostgreSQLGrantRoleTool grants role membership */
type PostgreSQLGrantRoleTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLGrantRoleTool creates a new PostgreSQL grant role tool */
func NewPostgreSQLGrantRoleTool(db *database.Database, logger *logging.Logger) *PostgreSQLGrantRoleTool {
	return &PostgreSQLGrantRoleTool{
		BaseTool: NewBaseTool(
			"postgresql_grant_role",
			"Grant role membership to users/roles",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"role_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the role to grant",
					},
					"grantees": map[string]interface{}{
						"type":        "array",
						"description": "List of role/user names to grant the role to",
					},
					"with_admin_option": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Allow grantees to grant this role to others",
					},
				},
				"required": []interface{}{"role_name", "grantees"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute grants the role */
func (t *PostgreSQLGrantRoleTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	roleName, ok := params["role_name"].(string)
	if !ok || roleName == "" {
		return Error("role_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	grantees, ok := params["grantees"].([]interface{})
	if !ok || len(grantees) == 0 {
		return Error("grantees parameter is required and must be a non-empty array", "INVALID_PARAMETER", nil), nil
	}

	/* Build grantee list */
	granteeList := []string{}
	for _, grantee := range grantees {
		if granteeStr, ok := grantee.(string); ok && granteeStr != "" {
			granteeList = append(granteeList, quoteIdentifier(granteeStr))
		}
	}
	if len(granteeList) == 0 {
		return Error("No valid grantees specified", "INVALID_PARAMETER", nil), nil
	}
	granteesStr := strings.Join(granteeList, ", ")

	/* Build GRANT ROLE statement */
	parts := []string{
		"GRANT",
		quoteIdentifier(roleName),
		"TO",
		granteesStr,
	}

	if withAdminOption, ok := params["with_admin_option"].(bool); ok && withAdminOption {
		parts = append(parts, "WITH ADMIN OPTION")
	}

	grantQuery := strings.Join(parts, " ")

	/* Execute GRANT ROLE */
	err := t.executor.Exec(ctx, grantQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("GRANT ROLE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Role granted", map[string]interface{}{
		"role_name": roleName,
		"grantees":  granteesStr,
	})

	return Success(map[string]interface{}{
		"role_name": roleName,
		"grantees":  granteesStr,
		"query":     grantQuery,
	}, map[string]interface{}{
		"tool": "postgresql_grant_role",
	}), nil
}

/* PostgreSQLRevokeRoleTool revokes role membership */
type PostgreSQLRevokeRoleTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLRevokeRoleTool creates a new PostgreSQL revoke role tool */
func NewPostgreSQLRevokeRoleTool(db *database.Database, logger *logging.Logger) *PostgreSQLRevokeRoleTool {
	return &PostgreSQLRevokeRoleTool{
		BaseTool: NewBaseTool(
			"postgresql_revoke_role",
			"Revoke role membership from users/roles",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"role_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the role to revoke",
					},
					"revokees": map[string]interface{}{
						"type":        "array",
						"description": "List of role/user names to revoke the role from",
					},
					"admin_option": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Revoke ADMIN OPTION only (not the role itself)",
					},
				},
				"required": []interface{}{"role_name", "revokees"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute revokes the role */
func (t *PostgreSQLRevokeRoleTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	roleName, ok := params["role_name"].(string)
	if !ok || roleName == "" {
		return Error("role_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	revokees, ok := params["revokees"].([]interface{})
	if !ok || len(revokees) == 0 {
		return Error("revokees parameter is required and must be a non-empty array", "INVALID_PARAMETER", nil), nil
	}

	/* Build revokee list */
	revokeeList := []string{}
	for _, revokee := range revokees {
		if revokeeStr, ok := revokee.(string); ok && revokeeStr != "" {
			revokeeList = append(revokeeList, quoteIdentifier(revokeeStr))
		}
	}
	if len(revokeeList) == 0 {
		return Error("No valid revokees specified", "INVALID_PARAMETER", nil), nil
	}
	revokeesStr := strings.Join(revokeeList, ", ")

	/* Build REVOKE ROLE statement */
	parts := []string{"REVOKE"}

	if adminOption, ok := params["admin_option"].(bool); ok && adminOption {
		parts = append(parts, "ADMIN OPTION FOR")
	}

	parts = append(parts, quoteIdentifier(roleName), "FROM", revokeesStr)

	revokeQuery := strings.Join(parts, " ")

	/* Execute REVOKE ROLE */
	err := t.executor.Exec(ctx, revokeQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("REVOKE ROLE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Role revoked", map[string]interface{}{
		"role_name": roleName,
		"revokees":  revokeesStr,
	})

	return Success(map[string]interface{}{
		"role_name": roleName,
		"revokees":  revokeesStr,
		"query":     revokeQuery,
	}, map[string]interface{}{
		"tool": "postgresql_revoke_role",
	}), nil
}


