/*-------------------------------------------------------------------------
 *
 * postgresql_user_management.go
 *    User and role management tools for NeuronMCP
 *
 * Implements comprehensive user and role DDL operations:
 * - CREATE/ALTER/DROP USER
 * - CREATE/ALTER/DROP ROLE
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/postgresql_user_management.go
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
 * User Management Tools
 * ============================================================================ */

/* PostgreSQLCreateUserTool creates new users */
type PostgreSQLCreateUserTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCreateUserTool creates a new PostgreSQL create user tool */
func NewPostgreSQLCreateUserTool(db *database.Database, logger *logging.Logger) *PostgreSQLCreateUserTool {
	return &PostgreSQLCreateUserTool{
		BaseTool: NewBaseTool(
			"postgresql_create_user",
			"Create a new PostgreSQL user with password and options (SUPERUSER, CREATEDB, CREATEROLE, etc.)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"username": map[string]interface{}{
						"type":        "string",
						"description": "Name of the user to create",
					},
					"password": map[string]interface{}{
						"type":        "string",
						"description": "User password",
					},
					"superuser": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Create as superuser",
					},
					"createdb": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Can create databases",
					},
					"createrole": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Can create roles",
					},
					"inherit": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Inherit privileges from roles",
					},
					"login": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Can login",
					},
					"replication": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Can initiate replication",
					},
					"bypassrls": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Bypass row-level security",
					},
					"connection_limit": map[string]interface{}{
						"type":        "integer",
						"description": "Connection limit (-1 for unlimited)",
					},
					"password_expires": map[string]interface{}{
						"type":        "string",
						"description": "Password expiration date (TIMESTAMP)",
					},
					"valid_until": map[string]interface{}{
						"type":        "string",
						"description": "Account expiration date (TIMESTAMP)",
					},
					"in_role": map[string]interface{}{
						"type":        "array",
						"description": "List of role names to add user to",
					},
					"in_group": map[string]interface{}{
						"type":        "array",
						"description": "List of group role names to add user to",
					},
				},
				"required": []interface{}{"username"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute creates the user */
func (t *PostgreSQLCreateUserTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	username, ok := params["username"].(string)
	if !ok || username == "" {
		return Error("username parameter is required", "INVALID_PARAMETER", nil), nil
	}

	if !isValidIdentifier(username) {
		return Error("Invalid username: must be a valid PostgreSQL identifier", "INVALID_PARAMETER", nil), nil
	}

	/* Build CREATE USER statement */
	parts := []string{"CREATE USER", quoteIdentifier(username)}

	options := []string{}

	/* Password */
	if password, ok := params["password"].(string); ok && password != "" {
		options = append(options, fmt.Sprintf("PASSWORD %s", quoteLiteral(password)))
	}

	/* Role attributes */
	if superuser, ok := params["superuser"].(bool); ok && superuser {
		options = append(options, "SUPERUSER")
	} else {
		options = append(options, "NOSUPERUSER")
	}

	if createdb, ok := params["createdb"].(bool); ok && createdb {
		options = append(options, "CREATEDB")
	} else {
		options = append(options, "NOCREATEDB")
	}

	if createrole, ok := params["createrole"].(bool); ok && createrole {
		options = append(options, "CREATEROLE")
	} else {
		options = append(options, "NOCREATEROLE")
	}

	inherit := true
	if inheritVal, ok := params["inherit"].(bool); ok {
		inherit = inheritVal
	}
	if inherit {
		options = append(options, "INHERIT")
	} else {
		options = append(options, "NOINHERIT")
	}

	login := true
	if loginVal, ok := params["login"].(bool); ok {
		login = loginVal
	}
	if login {
		options = append(options, "LOGIN")
	} else {
		options = append(options, "NOLOGIN")
	}

	if replication, ok := params["replication"].(bool); ok && replication {
		options = append(options, "REPLICATION")
	} else {
		options = append(options, "NOREPLICATION")
	}

	if bypassrls, ok := params["bypassrls"].(bool); ok && bypassrls {
		options = append(options, "BYPASSRLS")
	} else {
		options = append(options, "NOBYPASSRLS")
	}

	/* Connection limit */
	if connLimit, ok := params["connection_limit"].(float64); ok {
		options = append(options, fmt.Sprintf("CONNECTION LIMIT %d", int(connLimit)))
	} else if connLimit, ok := params["connection_limit"].(int); ok {
		options = append(options, fmt.Sprintf("CONNECTION LIMIT %d", connLimit))
	}

	/* Password expiration */
	if passwordExpires, ok := params["password_expires"].(string); ok && passwordExpires != "" {
		options = append(options, fmt.Sprintf("PASSWORD EXPIRE %s", quoteLiteral(passwordExpires)))
	}

	/* Account expiration */
	if validUntil, ok := params["valid_until"].(string); ok && validUntil != "" {
		options = append(options, fmt.Sprintf("VALID UNTIL %s", quoteLiteral(validUntil)))
	}

	if len(options) > 0 {
		parts = append(parts, "WITH")
		parts = append(parts, strings.Join(options, " "))
	}

	createQuery := strings.Join(parts, " ")

	/* Execute CREATE USER */
	err := t.executor.Exec(ctx, createQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("CREATE USER failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	/* Add user to roles if specified */
	if inRole, ok := params["in_role"].([]interface{}); ok && len(inRole) > 0 {
		for _, role := range inRole {
			if roleName, ok := role.(string); ok && roleName != "" {
				grantQuery := fmt.Sprintf("GRANT %s TO %s", quoteIdentifier(roleName), quoteIdentifier(username))
				if err := t.executor.Exec(ctx, grantQuery, nil); err != nil {
					t.logger.Warn("Failed to grant role to user", map[string]interface{}{
						"username": username,
						"role":     roleName,
						"error":    err.Error(),
					})
				}
			}
		}
	}

	/* Add user to groups if specified */
	if inGroup, ok := params["in_group"].([]interface{}); ok && len(inGroup) > 0 {
		for _, group := range inGroup {
			if groupName, ok := group.(string); ok && groupName != "" {
				grantQuery := fmt.Sprintf("GRANT %s TO %s", quoteIdentifier(groupName), quoteIdentifier(username))
				if err := t.executor.Exec(ctx, grantQuery, nil); err != nil {
					t.logger.Warn("Failed to grant group to user", map[string]interface{}{
						"username": username,
						"group":    groupName,
						"error":    err.Error(),
					})
				}
			}
		}
	}

	t.logger.Info("User created", map[string]interface{}{
		"username": username,
	})

	return Success(map[string]interface{}{
		"username": username,
		"query":     createQuery,
	}, map[string]interface{}{
		"tool": "postgresql_create_user",
	}), nil
}

/* PostgreSQLAlterUserTool alters user properties */
type PostgreSQLAlterUserTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLAlterUserTool creates a new PostgreSQL alter user tool */
func NewPostgreSQLAlterUserTool(db *database.Database, logger *logging.Logger) *PostgreSQLAlterUserTool {
	return &PostgreSQLAlterUserTool{
		BaseTool: NewBaseTool(
			"postgresql_alter_user",
			"Modify user properties (password, options, connection limit, expiration, rename)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"username": map[string]interface{}{
						"type":        "string",
						"description": "Name of the user to alter",
					},
					"new_name": map[string]interface{}{
						"type":        "string",
						"description": "New name for the user (rename)",
					},
					"password": map[string]interface{}{
						"type":        "string",
						"description": "New password",
					},
					"superuser": map[string]interface{}{
						"type":        "boolean",
						"description": "Set superuser status",
					},
					"createdb": map[string]interface{}{
						"type":        "boolean",
						"description": "Set CREATEDB privilege",
					},
					"createrole": map[string]interface{}{
						"type":        "boolean",
						"description": "Set CREATEROLE privilege",
					},
					"inherit": map[string]interface{}{
						"type":        "boolean",
						"description": "Set INHERIT privilege",
					},
					"login": map[string]interface{}{
						"type":        "boolean",
						"description": "Set LOGIN privilege",
					},
					"replication": map[string]interface{}{
						"type":        "boolean",
						"description": "Set REPLICATION privilege",
					},
					"bypassrls": map[string]interface{}{
						"type":        "boolean",
						"description": "Set BYPASSRLS privilege",
					},
					"connection_limit": map[string]interface{}{
						"type":        "integer",
						"description": "New connection limit (-1 for unlimited)",
					},
					"reset_connection_limit": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Reset connection limit to default",
					},
					"password_expires": map[string]interface{}{
						"type":        "string",
						"description": "Password expiration date (TIMESTAMP) or 'infinity'",
					},
					"valid_until": map[string]interface{}{
						"type":        "string",
						"description": "Account expiration date (TIMESTAMP) or 'infinity'",
					},
				},
				"required": []interface{}{"username"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute alters the user */
func (t *PostgreSQLAlterUserTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	username, ok := params["username"].(string)
	if !ok || username == "" {
		return Error("username parameter is required", "INVALID_PARAMETER", nil), nil
	}

	/* Build ALTER USER statement */
	parts := []string{"ALTER USER", quoteIdentifier(username)}

	alterations := []string{}

	/* Rename */
	if newName, ok := params["new_name"].(string); ok && newName != "" {
		if !isValidIdentifier(newName) {
			return Error("Invalid new username: must be a valid PostgreSQL identifier", "INVALID_PARAMETER", nil), nil
		}
		alterations = append(alterations, fmt.Sprintf("RENAME TO %s", quoteIdentifier(newName)))
		/* Update username for subsequent operations */
		username = newName
	}

	/* Password */
	if password, ok := params["password"].(string); ok && password != "" {
		alterations = append(alterations, fmt.Sprintf("PASSWORD %s", quoteLiteral(password)))
	}

	/* Role attributes */
	if superuser, ok := params["superuser"].(bool); ok {
		if superuser {
			alterations = append(alterations, "SUPERUSER")
		} else {
			alterations = append(alterations, "NOSUPERUSER")
		}
	}

	if createdb, ok := params["createdb"].(bool); ok {
		if createdb {
			alterations = append(alterations, "CREATEDB")
		} else {
			alterations = append(alterations, "NOCREATEDB")
		}
	}

	if createrole, ok := params["createrole"].(bool); ok {
		if createrole {
			alterations = append(alterations, "CREATEROLE")
		} else {
			alterations = append(alterations, "NOCREATEROLE")
		}
	}

	if inherit, ok := params["inherit"].(bool); ok {
		if inherit {
			alterations = append(alterations, "INHERIT")
		} else {
			alterations = append(alterations, "NOINHERIT")
		}
	}

	if login, ok := params["login"].(bool); ok {
		if login {
			alterations = append(alterations, "LOGIN")
		} else {
			alterations = append(alterations, "NOLOGIN")
		}
	}

	if replication, ok := params["replication"].(bool); ok {
		if replication {
			alterations = append(alterations, "REPLICATION")
		} else {
			alterations = append(alterations, "NOREPLICATION")
		}
	}

	if bypassrls, ok := params["bypassrls"].(bool); ok {
		if bypassrls {
			alterations = append(alterations, "BYPASSRLS")
		} else {
			alterations = append(alterations, "NOBYPASSRLS")
		}
	}

	/* Connection limit */
	if resetConnLimit, ok := params["reset_connection_limit"].(bool); ok && resetConnLimit {
		alterations = append(alterations, "RESET CONNECTION LIMIT")
	} else if connLimit, ok := params["connection_limit"].(float64); ok {
		alterations = append(alterations, fmt.Sprintf("CONNECTION LIMIT %d", int(connLimit)))
	} else if connLimit, ok := params["connection_limit"].(int); ok {
		alterations = append(alterations, fmt.Sprintf("CONNECTION LIMIT %d", connLimit))
	}

	/* Password expiration */
	if passwordExpires, ok := params["password_expires"].(string); ok && passwordExpires != "" {
		alterations = append(alterations, fmt.Sprintf("PASSWORD EXPIRE %s", quoteLiteral(passwordExpires)))
	}

	/* Account expiration */
	if validUntil, ok := params["valid_until"].(string); ok && validUntil != "" {
		alterations = append(alterations, fmt.Sprintf("VALID UNTIL %s", quoteLiteral(validUntil)))
	}

	if len(alterations) == 0 {
		return Error("No alterations specified", "INVALID_PARAMETER", nil), nil
	}

	/* Execute each alteration separately */
	queries := []string{}
	for _, alt := range alterations {
		query := parts[0] + " " + parts[1] + " " + alt
		queries = append(queries, query)
		
		err := t.executor.Exec(ctx, query, nil)
		if err != nil {
			return Error(
				fmt.Sprintf("ALTER USER failed: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error(), "query": query},
			), nil
		}
	}

	t.logger.Info("User altered", map[string]interface{}{
		"username":   username,
		"alterations": len(alterations),
	})

	return Success(map[string]interface{}{
		"username": username,
		"queries":   queries,
	}, map[string]interface{}{
		"tool": "postgresql_alter_user",
	}), nil
}

/* PostgreSQLDropUserTool drops users */
type PostgreSQLDropUserTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLDropUserTool creates a new PostgreSQL drop user tool */
func NewPostgreSQLDropUserTool(db *database.Database, logger *logging.Logger) *PostgreSQLDropUserTool {
	return &PostgreSQLDropUserTool{
		BaseTool: NewBaseTool(
			"postgresql_drop_user",
			"Drop a PostgreSQL user with safety checks",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"username": map[string]interface{}{
						"type":        "string",
						"description": "Name of the user to drop",
					},
					"if_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF EXISTS clause",
					},
				},
				"required": []interface{}{"username"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute drops the user */
func (t *PostgreSQLDropUserTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	username, ok := params["username"].(string)
	if !ok || username == "" {
		return Error("username parameter is required", "INVALID_PARAMETER", nil), nil
	}

	/* Build DROP USER statement */
	parts := []string{"DROP USER"}
	
	if ifExists, ok := params["if_exists"].(bool); ok && ifExists {
		parts = append(parts, "IF EXISTS")
	}
	
	parts = append(parts, quoteIdentifier(username))

	dropQuery := strings.Join(parts, " ")

	/* Execute DROP USER */
	err := t.executor.Exec(ctx, dropQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("DROP USER failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("User dropped", map[string]interface{}{
		"username": username,
	})

	return Success(map[string]interface{}{
		"username": username,
		"query":    dropQuery,
	}, map[string]interface{}{
		"tool": "postgresql_drop_user",
	}), nil
}

/* ============================================================================
 * Role Management Tools
 * ============================================================================ */

/* PostgreSQLCreateRoleTool creates new roles */
type PostgreSQLCreateRoleTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCreateRoleTool creates a new PostgreSQL create role tool */
func NewPostgreSQLCreateRoleTool(db *database.Database, logger *logging.Logger) *PostgreSQLCreateRoleTool {
	return &PostgreSQLCreateRoleTool{
		BaseTool: NewBaseTool(
			"postgresql_create_role",
			"Create a new PostgreSQL role with all options",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"role_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the role to create",
					},
					"password": map[string]interface{}{
						"type":        "string",
						"description": "Role password (if role can login)",
					},
					"superuser": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Create as superuser",
					},
					"createdb": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Can create databases",
					},
					"createrole": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Can create roles",
					},
					"inherit": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Inherit privileges from roles",
					},
					"login": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Can login",
					},
					"replication": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Can initiate replication",
					},
					"bypassrls": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Bypass row-level security",
					},
					"connection_limit": map[string]interface{}{
						"type":        "integer",
						"description": "Connection limit (-1 for unlimited)",
					},
					"valid_until": map[string]interface{}{
						"type":        "string",
						"description": "Account expiration date (TIMESTAMP)",
					},
					"in_role": map[string]interface{}{
						"type":        "array",
						"description": "List of role names to add this role to",
					},
					"role": map[string]interface{}{
						"type":        "array",
						"description": "List of role names to grant to this role",
					},
				},
				"required": []interface{}{"role_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute creates the role */
func (t *PostgreSQLCreateRoleTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	roleName, ok := params["role_name"].(string)
	if !ok || roleName == "" {
		return Error("role_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	if !isValidIdentifier(roleName) {
		return Error("Invalid role name: must be a valid PostgreSQL identifier", "INVALID_PARAMETER", nil), nil
	}

	/* Build CREATE ROLE statement */
	parts := []string{"CREATE ROLE", quoteIdentifier(roleName)}

	options := []string{}

	/* Password */
	if password, ok := params["password"].(string); ok && password != "" {
		options = append(options, fmt.Sprintf("PASSWORD %s", quoteLiteral(password)))
	}

	/* Role attributes (same as user) */
	if superuser, ok := params["superuser"].(bool); ok && superuser {
		options = append(options, "SUPERUSER")
	} else {
		options = append(options, "NOSUPERUSER")
	}

	if createdb, ok := params["createdb"].(bool); ok && createdb {
		options = append(options, "CREATEDB")
	} else {
		options = append(options, "NOCREATEDB")
	}

	if createrole, ok := params["createrole"].(bool); ok && createrole {
		options = append(options, "CREATEROLE")
	} else {
		options = append(options, "NOCREATEROLE")
	}

	inherit := true
	if inheritVal, ok := params["inherit"].(bool); ok {
		inherit = inheritVal
	}
	if inherit {
		options = append(options, "INHERIT")
	} else {
		options = append(options, "NOINHERIT")
	}

	login := false
	if loginVal, ok := params["login"].(bool); ok {
		login = loginVal
	}
	if login {
		options = append(options, "LOGIN")
	} else {
		options = append(options, "NOLOGIN")
	}

	if replication, ok := params["replication"].(bool); ok && replication {
		options = append(options, "REPLICATION")
	} else {
		options = append(options, "NOREPLICATION")
	}

	if bypassrls, ok := params["bypassrls"].(bool); ok && bypassrls {
		options = append(options, "BYPASSRLS")
	} else {
		options = append(options, "NOBYPASSRLS")
	}

	/* Connection limit */
	if connLimit, ok := params["connection_limit"].(float64); ok {
		options = append(options, fmt.Sprintf("CONNECTION LIMIT %d", int(connLimit)))
	} else if connLimit, ok := params["connection_limit"].(int); ok {
		options = append(options, fmt.Sprintf("CONNECTION LIMIT %d", connLimit))
	}

	/* Account expiration */
	if validUntil, ok := params["valid_until"].(string); ok && validUntil != "" {
		options = append(options, fmt.Sprintf("VALID UNTIL %s", quoteLiteral(validUntil)))
	}

	if len(options) > 0 {
		parts = append(parts, "WITH")
		parts = append(parts, strings.Join(options, " "))
	}

	createQuery := strings.Join(parts, " ")

	/* Execute CREATE ROLE */
	err := t.executor.Exec(ctx, createQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("CREATE ROLE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	/* Add role to other roles if specified */
	if inRole, ok := params["in_role"].([]interface{}); ok && len(inRole) > 0 {
		for _, role := range inRole {
			if roleNameStr, ok := role.(string); ok && roleNameStr != "" {
				grantQuery := fmt.Sprintf("GRANT %s TO %s", quoteIdentifier(roleNameStr), quoteIdentifier(roleName))
				if err := t.executor.Exec(ctx, grantQuery, nil); err != nil {
					t.logger.Warn("Failed to grant role membership", map[string]interface{}{
						"role":  roleName,
						"in_role": roleNameStr,
						"error": err.Error(),
					})
				}
			}
		}
	}

	/* Grant roles to this role if specified */
	if role, ok := params["role"].([]interface{}); ok && len(role) > 0 {
		for _, roleNameStr := range role {
			if roleStr, ok := roleNameStr.(string); ok && roleStr != "" {
				grantQuery := fmt.Sprintf("GRANT %s TO %s", quoteIdentifier(roleStr), quoteIdentifier(roleName))
				if err := t.executor.Exec(ctx, grantQuery, nil); err != nil {
					t.logger.Warn("Failed to grant role", map[string]interface{}{
						"role":  roleName,
						"granted_role": roleStr,
						"error": err.Error(),
					})
				}
			}
		}
	}

	t.logger.Info("Role created", map[string]interface{}{
		"role_name": roleName,
	})

	return Success(map[string]interface{}{
		"role_name": roleName,
		"query":     createQuery,
	}, map[string]interface{}{
		"tool": "postgresql_create_role",
	}), nil
}

/* PostgreSQLAlterRoleTool alters role properties */
type PostgreSQLAlterRoleTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLAlterRoleTool creates a new PostgreSQL alter role tool */
func NewPostgreSQLAlterRoleTool(db *database.Database, logger *logging.Logger) *PostgreSQLAlterRoleTool {
	return &PostgreSQLAlterRoleTool{
		BaseTool: NewBaseTool(
			"postgresql_alter_role",
			"Modify role properties (password, options, connection limit, expiration, rename)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"role_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the role to alter",
					},
					"new_name": map[string]interface{}{
						"type":        "string",
						"description": "New name for the role (rename)",
					},
					"password": map[string]interface{}{
						"type":        "string",
						"description": "New password",
					},
					"superuser": map[string]interface{}{
						"type":        "boolean",
						"description": "Set superuser status",
					},
					"createdb": map[string]interface{}{
						"type":        "boolean",
						"description": "Set CREATEDB privilege",
					},
					"createrole": map[string]interface{}{
						"type":        "boolean",
						"description": "Set CREATEROLE privilege",
					},
					"inherit": map[string]interface{}{
						"type":        "boolean",
						"description": "Set INHERIT privilege",
					},
					"login": map[string]interface{}{
						"type":        "boolean",
						"description": "Set LOGIN privilege",
					},
					"replication": map[string]interface{}{
						"type":        "boolean",
						"description": "Set REPLICATION privilege",
					},
					"bypassrls": map[string]interface{}{
						"type":        "boolean",
						"description": "Set BYPASSRLS privilege",
					},
					"connection_limit": map[string]interface{}{
						"type":        "integer",
						"description": "New connection limit (-1 for unlimited)",
					},
					"reset_connection_limit": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Reset connection limit to default",
					},
					"valid_until": map[string]interface{}{
						"type":        "string",
						"description": "Account expiration date (TIMESTAMP) or 'infinity'",
					},
				},
				"required": []interface{}{"role_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute alters the role */
func (t *PostgreSQLAlterRoleTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	roleName, ok := params["role_name"].(string)
	if !ok || roleName == "" {
		return Error("role_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	/* Build ALTER ROLE statement */
	parts := []string{"ALTER ROLE", quoteIdentifier(roleName)}

	alterations := []string{}

	/* Rename */
	if newName, ok := params["new_name"].(string); ok && newName != "" {
		if !isValidIdentifier(newName) {
			return Error("Invalid new role name: must be a valid PostgreSQL identifier", "INVALID_PARAMETER", nil), nil
		}
		alterations = append(alterations, fmt.Sprintf("RENAME TO %s", quoteIdentifier(newName)))
		roleName = newName
	}

	/* Password */
	if password, ok := params["password"].(string); ok && password != "" {
		alterations = append(alterations, fmt.Sprintf("PASSWORD %s", quoteLiteral(password)))
	}

	/* Role attributes (same as ALTER USER) */
	if superuser, ok := params["superuser"].(bool); ok {
		if superuser {
			alterations = append(alterations, "SUPERUSER")
		} else {
			alterations = append(alterations, "NOSUPERUSER")
		}
	}

	if createdb, ok := params["createdb"].(bool); ok {
		if createdb {
			alterations = append(alterations, "CREATEDB")
		} else {
			alterations = append(alterations, "NOCREATEDB")
		}
	}

	if createrole, ok := params["createrole"].(bool); ok {
		if createrole {
			alterations = append(alterations, "CREATEROLE")
		} else {
			alterations = append(alterations, "NOCREATEROLE")
		}
	}

	if inherit, ok := params["inherit"].(bool); ok {
		if inherit {
			alterations = append(alterations, "INHERIT")
		} else {
			alterations = append(alterations, "NOINHERIT")
		}
	}

	if login, ok := params["login"].(bool); ok {
		if login {
			alterations = append(alterations, "LOGIN")
		} else {
			alterations = append(alterations, "NOLOGIN")
		}
	}

	if replication, ok := params["replication"].(bool); ok {
		if replication {
			alterations = append(alterations, "REPLICATION")
		} else {
			alterations = append(alterations, "NOREPLICATION")
		}
	}

	if bypassrls, ok := params["bypassrls"].(bool); ok {
		if bypassrls {
			alterations = append(alterations, "BYPASSRLS")
		} else {
			alterations = append(alterations, "NOBYPASSRLS")
		}
	}

	/* Connection limit */
	if resetConnLimit, ok := params["reset_connection_limit"].(bool); ok && resetConnLimit {
		alterations = append(alterations, "RESET CONNECTION LIMIT")
	} else if connLimit, ok := params["connection_limit"].(float64); ok {
		alterations = append(alterations, fmt.Sprintf("CONNECTION LIMIT %d", int(connLimit)))
	} else if connLimit, ok := params["connection_limit"].(int); ok {
		alterations = append(alterations, fmt.Sprintf("CONNECTION LIMIT %d", connLimit))
	}

	/* Account expiration */
	if validUntil, ok := params["valid_until"].(string); ok && validUntil != "" {
		alterations = append(alterations, fmt.Sprintf("VALID UNTIL %s", quoteLiteral(validUntil)))
	}

	if len(alterations) == 0 {
		return Error("No alterations specified", "INVALID_PARAMETER", nil), nil
	}

	/* Execute each alteration separately */
	queries := []string{}
	for _, alt := range alterations {
		query := parts[0] + " " + parts[1] + " " + alt
		queries = append(queries, query)
		
		err := t.executor.Exec(ctx, query, nil)
		if err != nil {
			return Error(
				fmt.Sprintf("ALTER ROLE failed: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error(), "query": query},
			), nil
		}
	}

	t.logger.Info("Role altered", map[string]interface{}{
		"role_name":  roleName,
		"alterations": len(alterations),
	})

	return Success(map[string]interface{}{
		"role_name": roleName,
		"queries":   queries,
	}, map[string]interface{}{
		"tool": "postgresql_alter_role",
	}), nil
}

/* PostgreSQLDropRoleTool drops roles */
type PostgreSQLDropRoleTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLDropRoleTool creates a new PostgreSQL drop role tool */
func NewPostgreSQLDropRoleTool(db *database.Database, logger *logging.Logger) *PostgreSQLDropRoleTool {
	return &PostgreSQLDropRoleTool{
		BaseTool: NewBaseTool(
			"postgresql_drop_role",
			"Drop a PostgreSQL role with safety checks",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"role_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the role to drop",
					},
					"if_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF EXISTS clause",
					},
				},
				"required": []interface{}{"role_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute drops the role */
func (t *PostgreSQLDropRoleTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	roleName, ok := params["role_name"].(string)
	if !ok || roleName == "" {
		return Error("role_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	/* Build DROP ROLE statement */
	parts := []string{"DROP ROLE"}
	
	if ifExists, ok := params["if_exists"].(bool); ok && ifExists {
		parts = append(parts, "IF EXISTS")
	}
	
	parts = append(parts, quoteIdentifier(roleName))

	dropQuery := strings.Join(parts, " ")

	/* Execute DROP ROLE */
	err := t.executor.Exec(ctx, dropQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("DROP ROLE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Role dropped", map[string]interface{}{
		"role_name": roleName,
	})

	return Success(map[string]interface{}{
		"role_name": roleName,
		"query":     dropQuery,
	}, map[string]interface{}{
		"tool": "postgresql_drop_role",
	}), nil
}

