/*-------------------------------------------------------------------------
 *
 * integration.go
 *    Integration tools for NeuronMCP
 *
 * Provides SDK generation and integration management tools.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/integration.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/sdk"
)

/* SDKGeneratorTool generates SDKs for different languages */
type SDKGeneratorTool struct {
	*BaseTool
	db          *database.Database
	logger      *logging.Logger
	sdkGenerator *sdk.SDKGenerator
	toolRegistry *ToolRegistry
}

/* NewSDKGeneratorTool creates a new SDK generator tool */
func NewSDKGeneratorTool(db *database.Database, logger *logging.Logger, toolRegistry *ToolRegistry) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"language": map[string]interface{}{
				"type":        "string",
				"description": "Target language: python, typescript, go, java",
				"enum":        []interface{}{"python", "typescript", "go", "java"},
			},
			"output_format": map[string]interface{}{
				"type":        "string",
				"description": "Output format: code, file",
				"enum":        []interface{}{"code", "file"},
				"default":     "code",
			},
		},
		"required": []interface{}{"language"},
	}

	return &SDKGeneratorTool{
		BaseTool: NewBaseTool(
			"sdk_generator",
			"Auto-generate SDKs for Python, TypeScript, Go, Java",
			inputSchema,
		),
		db:          db,
		logger:      logger,
		sdkGenerator: sdk.NewSDKGenerator(),
		toolRegistry: toolRegistry,
	}
}

/* Execute executes the SDK generator tool */
func (t *SDKGeneratorTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	language, _ := params["language"].(string)
	outputFormat, _ := params["output_format"].(string)

	if language == "" {
		return Error("language is required", "INVALID_PARAMS", nil), nil
	}

	if outputFormat == "" {
		outputFormat = "code"
	}

	/* Get tool definitions */
	definitions := t.toolRegistry.GetAllDefinitions()
	sdkDefinitions := make([]sdk.ToolDefinition, len(definitions))
	for i, def := range definitions {
		sdkDefinitions[i] = sdk.ToolDefinition{
			Name:        def.Name,
			Description: def.Description,
			InputSchema:  def.InputSchema,
		}
	}

	code, err := t.sdkGenerator.Generate(language, sdkDefinitions)
	if err != nil {
		return Error(fmt.Sprintf("Failed to generate SDK: %v", err), "GENERATION_ERROR", nil), nil
	}

	return Success(map[string]interface{}{
		"language":      language,
		"output_format": outputFormat,
		"code":          code,
	}, nil), nil
}

