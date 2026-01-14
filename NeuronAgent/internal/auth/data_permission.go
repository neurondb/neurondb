/*-------------------------------------------------------------------------
 *
 * data_permission.go
 *    Data permission checking for NeuronAgent
 *
 * Provides data permissions (schema, table, row filters, column masking)
 * integrated with SQL tool.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/auth/data_permission.go
 *
 *-------------------------------------------------------------------------
 */

package auth

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type DataPermissionChecker struct {
	queries *db.Queries
}

func NewDataPermissionChecker(queries *db.Queries) *DataPermissionChecker {
	return &DataPermissionChecker{queries: queries}
}

/* GetRowFilterForTable returns the row filter SQL for a principal on a table */
func (c *DataPermissionChecker) GetRowFilterForTable(ctx context.Context, principalID uuid.UUID, schemaName, tableName string) (string, error) {
	permissions, err := c.queries.ListDataPermissionsByResource(ctx, &schemaName, &tableName)
	if err != nil {
		return "", fmt.Errorf("failed to list data permissions: %w", err)
	}

	/* Find permission for this principal */
	for _, perm := range permissions {
		if perm.PrincipalID == principalID && perm.RowFilter != nil {
			return *perm.RowFilter, nil
		}
	}

	/* Check for schema-level permission */
	if schemaName != "" {
		schemaPerms, err := c.queries.ListDataPermissionsByResource(ctx, &schemaName, nil)
		if err == nil {
			for _, perm := range schemaPerms {
				if perm.PrincipalID == principalID && perm.RowFilter != nil {
					return *perm.RowFilter, nil
				}
			}
		}
	}

	return "", nil
}

/* GetColumnMaskForTable returns column masking rules for a principal on a table */
func (c *DataPermissionChecker) GetColumnMaskForTable(ctx context.Context, principalID uuid.UUID, schemaName, tableName string) (map[string]string, error) {
	permissions, err := c.queries.ListDataPermissionsByResource(ctx, &schemaName, &tableName)
	if err != nil {
		return nil, fmt.Errorf("failed to list data permissions: %w", err)
	}

	/* Find permission for this principal */
	for _, perm := range permissions {
		if perm.PrincipalID == principalID && perm.ColumnMask != nil {
			maskRules := make(map[string]string)
			for col, rule := range perm.ColumnMask {
				if ruleStr, ok := rule.(string); ok {
					maskRules[col] = ruleStr
				}
			}
			return maskRules, nil
		}
	}

	return nil, nil
}

/* ApplyRowFilter applies a row filter to a SQL query */
func (c *DataPermissionChecker) ApplyRowFilter(query, rowFilter string) string {
	if rowFilter == "" {
		return query
	}

	/* Simple approach: add WHERE clause if not present, or AND if WHERE exists */
	queryUpper := strings.ToUpper(strings.TrimSpace(query))

	if strings.Contains(queryUpper, "WHERE") {
		/* Add AND condition */
		whereIndex := strings.Index(strings.ToUpper(query), "WHERE")
		query = query[:whereIndex+6] + " (" + rowFilter + ") AND " + query[whereIndex+6:]
	} else {
		/* Add WHERE clause before ORDER BY, LIMIT, etc */
		insertPos := len(query)
		for _, keyword := range []string{"ORDER BY", "GROUP BY", "LIMIT", "OFFSET"} {
			idx := strings.Index(strings.ToUpper(query), keyword)
			if idx > 0 && idx < insertPos {
				insertPos = idx
			}
		}
		query = query[:insertPos] + " WHERE " + rowFilter + " " + query[insertPos:]
	}

	return query
}

/* MaskColumns masks columns in query results according to masking rules */
func (c *DataPermissionChecker) MaskColumns(results []map[string]interface{}, maskRules map[string]string) []map[string]interface{} {
	if len(maskRules) == 0 {
		return results
	}

	masked := make([]map[string]interface{}, len(results))
	for i, row := range results {
		maskedRow := make(map[string]interface{})
		for col, val := range row {
			if rule, ok := maskRules[col]; ok {
				maskedRow[col] = c.applyMaskRule(val, rule)
			} else {
				maskedRow[col] = val
			}
		}
		masked[i] = maskedRow
	}

	return masked
}

/* applyMaskRule applies a masking rule to a value */
func (c *DataPermissionChecker) applyMaskRule(value interface{}, rule string) interface{} {
	switch rule {
	case "redact", "mask":
		return "***REDACTED***"
	case "hash":
		if str, ok := value.(string); ok {
			/* Simple hash (in production, use proper hashing) */
			return fmt.Sprintf("hash_%d", len(str))
		}
		return "***HASHED***"
	case "partial":
		if str, ok := value.(string); ok {
			if len(str) <= 4 {
				return "****"
			}
			return str[:2] + "****" + str[len(str)-2:]
		}
		return "***PARTIAL***"
	default:
		return value
	}
}

/* CheckDataPermission checks if a principal has permission to access a resource */
func (c *DataPermissionChecker) CheckDataPermission(ctx context.Context, principalID uuid.UUID, schemaName, tableName, action string) (bool, error) {
	permissions, err := c.queries.ListDataPermissionsByResource(ctx, &schemaName, &tableName)
	if err != nil {
		return false, fmt.Errorf("failed to list data permissions: %w", err)
	}

	/* Check permissions for this principal */
	for _, perm := range permissions {
		if perm.PrincipalID == principalID {
			for _, p := range perm.Permissions {
				if p == action || p == "*" {
					return true, nil
				}
			}
		}
	}

	/* Check schema-level permissions */
	if schemaName != "" {
		schemaPerms, err := c.queries.ListDataPermissionsByResource(ctx, &schemaName, nil)
		if err == nil {
			for _, perm := range schemaPerms {
				if perm.PrincipalID == principalID {
					for _, p := range perm.Permissions {
						if p == action || p == "*" {
							return true, nil
						}
					}
				}
			}
		}
	}

	return false, nil
}
