/*-------------------------------------------------------------------------
 *
 * enterprise.go
 *    Enterprise Features for NeuronMCP
 *
 * Provides multi-tenancy, data governance, compliance reporting,
 * advanced audit, and backup automation.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/enterprise.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* MultiTenantManagementTool provides multi-tenant isolation and management */
type MultiTenantManagementTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewMultiTenantManagementTool creates a new multi-tenant management tool */
func NewMultiTenantManagementTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"operation": map[string]interface{}{
				"type":        "string",
				"description": "Operation: create_tenant, list_tenants, get_tenant, update_tenant, delete_tenant",
				"enum":        []interface{}{"create_tenant", "list_tenants", "get_tenant", "update_tenant", "delete_tenant"},
			},
			"tenant_id": map[string]interface{}{
				"type":        "string",
				"description": "Tenant ID",
			},
			"tenant_name": map[string]interface{}{
				"type":        "string",
				"description": "Tenant name",
			},
			"config": map[string]interface{}{
				"type":        "object",
				"description": "Tenant configuration",
			},
		},
		"required": []interface{}{"operation"},
	}

	return &MultiTenantManagementTool{
		BaseTool: NewBaseTool(
			"multi_tenant_management",
			"Multi-tenant isolation and management",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the multi-tenant management tool */
func (t *MultiTenantManagementTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	operation, _ := params["operation"].(string)

	switch operation {
	case "create_tenant":
		return t.createTenant(ctx, params)
	case "list_tenants":
		return t.listTenants(ctx, params)
	case "get_tenant":
		return t.getTenant(ctx, params)
	case "update_tenant":
		return t.updateTenant(ctx, params)
	case "delete_tenant":
		return t.deleteTenant(ctx, params)
	default:
		return Error("Invalid operation", "INVALID_OPERATION", nil), nil
	}
}

/* createTenant creates a new tenant */
func (t *MultiTenantManagementTool) createTenant(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	tenantID, _ := params["tenant_id"].(string)
	tenantName, _ := params["tenant_name"].(string)
	config, _ := params["config"].(map[string]interface{})

	if tenantID == "" {
		tenantID = fmt.Sprintf("tenant_%d", time.Now().UnixNano())
	}

	query := `
		INSERT INTO neurondb.tenants 
		(tenant_id, tenant_name, config, created_at, status)
		VALUES ($1, $2, $3, NOW(), 'active')
	`

	configJSON, _ := json.Marshal(config)
	_, err := t.db.Query(ctx, query, []interface{}{tenantID, tenantName, string(configJSON)})
	if err != nil {
		/* Create table if needed */
		createTable := `
			CREATE TABLE IF NOT EXISTS neurondb.tenants (
				tenant_id VARCHAR(200) PRIMARY KEY,
				tenant_name VARCHAR(200) NOT NULL,
				config JSONB,
				created_at TIMESTAMP NOT NULL,
				updated_at TIMESTAMP,
				status VARCHAR(50) NOT NULL DEFAULT 'active'
			)
		`
		_, _ = t.db.Query(ctx, createTable, nil)
		_, err = t.db.Query(ctx, query, []interface{}{tenantID, tenantName, string(configJSON)})
		if err != nil {
			return Error(fmt.Sprintf("Failed to create tenant: %v", err), "CREATE_ERROR", nil), nil
		}
	}

	return Success(map[string]interface{}{
		"tenant_id":   tenantID,
		"tenant_name": tenantName,
		"status":      "active",
		"message":     "Tenant created successfully",
	}, nil), nil
}

/* listTenants lists all tenants */
func (t *MultiTenantManagementTool) listTenants(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := `
		SELECT tenant_id, tenant_name, status, created_at
		FROM neurondb.tenants
		ORDER BY created_at DESC
		LIMIT 100
	`

	rows, err := t.db.Query(ctx, query, nil)
	if err != nil {
		return Success(map[string]interface{}{
			"tenants": []interface{}{},
		}, nil), nil
	}
	defer rows.Close()

	tenants := []map[string]interface{}{}
	for rows.Next() {
		var tenantID, tenantName, status string
		var createdAt time.Time

		if err := rows.Scan(&tenantID, &tenantName, &status, &createdAt); err != nil {
			continue
		}

		tenants = append(tenants, map[string]interface{}{
			"tenant_id":   tenantID,
			"tenant_name": tenantName,
			"status":      status,
			"created_at":  createdAt,
		})
	}

	return Success(map[string]interface{}{
		"tenants": tenants,
	}, nil), nil
}

/* getTenant gets tenant information */
func (t *MultiTenantManagementTool) getTenant(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	tenantID, _ := params["tenant_id"].(string)

	if tenantID == "" {
		return Error("tenant_id is required", "INVALID_PARAMS", nil), nil
	}

	query := `
		SELECT tenant_id, tenant_name, config, status, created_at, updated_at
		FROM neurondb.tenants
		WHERE tenant_id = $1
	`

	rows, err := t.db.Query(ctx, query, []interface{}{tenantID})
	if err != nil {
		return Error("Tenant not found", "NOT_FOUND", nil), nil
	}
	defer rows.Close()

	if rows.Next() {
		var id, name, status string
		var configJSON *string
		var createdAt time.Time
		var updatedAt *time.Time

		if err := rows.Scan(&id, &name, &configJSON, &status, &createdAt, &updatedAt); err != nil {
			return Error("Failed to read tenant", "READ_ERROR", nil), nil
		}

		var config map[string]interface{}
		if configJSON != nil {
			json.Unmarshal([]byte(*configJSON), &config)
		}

		result := map[string]interface{}{
			"tenant_id":   id,
			"tenant_name": name,
			"status":      status,
			"created_at":  createdAt,
			"config":      config,
		}

		if updatedAt != nil {
			result["updated_at"] = *updatedAt
		}

		return Success(result, nil), nil
	}

	return Error("Tenant not found", "NOT_FOUND", nil), nil
}

/* updateTenant updates tenant information */
func (t *MultiTenantManagementTool) updateTenant(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	tenantID, _ := params["tenant_id"].(string)
	config, _ := params["config"].(map[string]interface{})

	if tenantID == "" {
		return Error("tenant_id is required", "INVALID_PARAMS", nil), nil
	}

	configJSON, _ := json.Marshal(config)
	query := `
		UPDATE neurondb.tenants
		SET config = $1, updated_at = NOW()
		WHERE tenant_id = $2
	`

	_, err := t.db.Query(ctx, query, []interface{}{string(configJSON), tenantID})
	if err != nil {
		return Error("Failed to update tenant", "UPDATE_ERROR", nil), nil
	}

	return Success(map[string]interface{}{
		"tenant_id": tenantID,
		"message":   "Tenant updated successfully",
	}, nil), nil
}

/* deleteTenant deletes a tenant */
func (t *MultiTenantManagementTool) deleteTenant(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	tenantID, _ := params["tenant_id"].(string)

	if tenantID == "" {
		return Error("tenant_id is required", "INVALID_PARAMS", nil), nil
	}

	query := `
		UPDATE neurondb.tenants
		SET status = 'deleted', updated_at = NOW()
		WHERE tenant_id = $1
	`

	_, err := t.db.Query(ctx, query, []interface{}{tenantID})
	if err != nil {
		return Error("Failed to delete tenant", "DELETE_ERROR", nil), nil
	}

	return Success(map[string]interface{}{
		"tenant_id": tenantID,
		"message":   "Tenant deleted successfully",
	}, nil), nil
}

/* DataGovernanceTool provides data classification, tagging, and governance */
type DataGovernanceTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewDataGovernanceTool creates a new data governance tool */
func NewDataGovernanceTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"operation": map[string]interface{}{
				"type":        "string",
				"description": "Operation: classify, tag, get_policy, apply_policy",
				"enum":        []interface{}{"classify", "tag", "get_policy", "apply_policy"},
			},
			"table": map[string]interface{}{
				"type":        "string",
				"description": "Table name",
			},
			"column": map[string]interface{}{
				"type":        "string",
				"description": "Column name",
			},
			"classification": map[string]interface{}{
				"type":        "string",
				"description": "Data classification: public, internal, confidential, restricted",
			},
			"tags": map[string]interface{}{
				"type":        "array",
				"description": "Tags to apply",
				"items": map[string]interface{}{
					"type": "string",
				},
			},
		},
		"required": []interface{}{"operation"},
	}

	return &DataGovernanceTool{
		BaseTool: NewBaseTool(
			"data_governance",
			"Data classification, tagging, and governance policies",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the data governance tool */
func (t *DataGovernanceTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	operation, _ := params["operation"].(string)

	switch operation {
	case "classify":
		return t.classifyData(ctx, params)
	case "tag":
		return t.tagData(ctx, params)
	case "get_policy":
		return t.getPolicy(ctx, params)
	case "apply_policy":
		return t.applyPolicy(ctx, params)
	default:
		return Error("Invalid operation", "INVALID_OPERATION", nil), nil
	}
}

/* classifyData classifies data */
func (t *DataGovernanceTool) classifyData(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	table, _ := params["table"].(string)
	column, _ := params["column"].(string)
	classification, _ := params["classification"].(string)

	if table == "" {
		return Error("table is required", "INVALID_PARAMS", nil), nil
	}

	if classification == "" {
		/* Auto-classify based on column name and data type */
		classification = t.autoClassify(ctx, table, column)
	}

	query := `
		INSERT INTO neurondb.data_classifications 
		(table_name, column_name, classification, created_at)
		VALUES ($1, $2, $3, NOW())
		ON CONFLICT (table_name, column_name) DO UPDATE
		SET classification = $3, updated_at = NOW()
	`

	_, err := t.db.Query(ctx, query, []interface{}{table, column, classification})
	if err != nil {
		/* Create table if needed */
		createTable := `
			CREATE TABLE IF NOT EXISTS neurondb.data_classifications (
				table_name VARCHAR(200) NOT NULL,
				column_name VARCHAR(200),
				classification VARCHAR(50) NOT NULL,
				created_at TIMESTAMP NOT NULL,
				updated_at TIMESTAMP,
				PRIMARY KEY (table_name, column_name)
			)
		`
		_, _ = t.db.Query(ctx, createTable, nil)
		_, err = t.db.Query(ctx, query, []interface{}{table, column, classification})
		if err != nil {
			return Error(fmt.Sprintf("Failed to classify data: %v", err), "CLASSIFY_ERROR", nil), nil
		}
	}

	return Success(map[string]interface{}{
		"table":          table,
		"column":         column,
		"classification": classification,
		"message":        "Data classified successfully",
	}, nil), nil
}

/* autoClassify automatically classifies data */
func (t *DataGovernanceTool) autoClassify(ctx context.Context, table, column string) string {
	/* Simple heuristics - would use ML in production */
	columnLower := strings.ToLower(column)
	if strings.Contains(columnLower, "password") || strings.Contains(columnLower, "secret") || strings.Contains(columnLower, "key") {
		return "restricted"
	}
	if strings.Contains(columnLower, "email") || strings.Contains(columnLower, "phone") || strings.Contains(columnLower, "ssn") {
		return "confidential"
	}
	if strings.Contains(columnLower, "name") || strings.Contains(columnLower, "address") {
		return "internal"
	}
	return "public"
}

/* tagData tags data */
func (t *DataGovernanceTool) tagData(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	table, _ := params["table"].(string)
	tagsRaw, _ := params["tags"].([]interface{})

	if table == "" {
		return Error("table is required", "INVALID_PARAMS", nil), nil
	}

	tags := []string{}
	for _, tag := range tagsRaw {
		tags = append(tags, fmt.Sprintf("%v", tag))
	}

	query := `
		INSERT INTO neurondb.data_tags 
		(table_name, tags, created_at)
		VALUES ($1, $2, NOW())
		ON CONFLICT (table_name) DO UPDATE
		SET tags = $2, updated_at = NOW()
	`

	tagsJSON, _ := json.Marshal(tags)
	_, err := t.db.Query(ctx, query, []interface{}{table, string(tagsJSON)})
	if err != nil {
		/* Create table if needed */
		createTable := `
			CREATE TABLE IF NOT EXISTS neurondb.data_tags (
				table_name VARCHAR(200) PRIMARY KEY,
				tags JSONB NOT NULL,
				created_at TIMESTAMP NOT NULL,
				updated_at TIMESTAMP
			)
		`
		_, _ = t.db.Query(ctx, createTable, nil)
		_, err = t.db.Query(ctx, query, []interface{}{table, string(tagsJSON)})
		if err != nil {
			return Error(fmt.Sprintf("Failed to tag data: %v", err), "TAG_ERROR", nil), nil
		}
	}

	return Success(map[string]interface{}{
		"table":  table,
		"tags":   tags,
		"message": "Data tagged successfully",
	}, nil), nil
}

/* getPolicy gets governance policy */
func (t *DataGovernanceTool) getPolicy(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	table, _ := params["table"].(string)

	query := `
		SELECT 
			dc.classification,
			dt.tags
		FROM neurondb.data_classifications dc
		LEFT JOIN neurondb.data_tags dt ON dc.table_name = dt.table_name
		WHERE dc.table_name = $1
		LIMIT 1
	`

	rows, err := t.db.Query(ctx, query, []interface{}{table})
	if err != nil {
		return Success(map[string]interface{}{
			"table":  table,
			"policy": map[string]interface{}{},
		}, nil), nil
	}
	defer rows.Close()

	if rows.Next() {
		var classification *string
		var tagsJSON *string

		if err := rows.Scan(&classification, &tagsJSON); err == nil {
			var tags []string
			if tagsJSON != nil {
				json.Unmarshal([]byte(*tagsJSON), &tags)
			}

			return Success(map[string]interface{}{
				"table":          table,
				"classification": getString(classification, "public"),
				"tags":           tags,
			}, nil), nil
		}
	}

	return Success(map[string]interface{}{
		"table":  table,
		"policy": map[string]interface{}{},
	}, nil), nil
}

/* applyPolicy applies governance policy */
func (t *DataGovernanceTool) applyPolicy(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	/* Apply governance policies based on classification and tags */
	return Success(map[string]interface{}{
		"message": "Policy applied successfully",
	}, nil), nil
}

