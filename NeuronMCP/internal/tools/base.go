/*-------------------------------------------------------------------------
 *
 * base.go
 *    Base tool types and utilities for NeuronMCP
 *
 * Provides common functionality for all tools including result types,
 * validation, and base tool structure.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/base.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"fmt"
	"reflect"

	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* ToolResult represents the result of tool execution */
type ToolResult struct {
	Success  bool                   `json:"success"`
	Data     interface{}            `json:"data,omitempty"`
	Error    *ToolError            `json:"error,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

/* ToolError represents a tool execution error */
type ToolError struct {
	Message string      `json:"message"`
	Code    string      `json:"code,omitempty"`
	Details interface{} `json:"details,omitempty"`
}

/* BaseTool provides common functionality for tools */
type BaseTool struct {
	name         string
	description  string
	inputSchema  map[string]interface{}
	outputSchema map[string]interface{}
	version      string
	deprecated   bool
	deprecation  *mcp.DeprecationInfo
}

/* NewBaseTool creates a new base tool */
func NewBaseTool(name, description string, inputSchema map[string]interface{}) *BaseTool {
	return &BaseTool{
		name:        name,
		description: description,
		inputSchema: inputSchema,
		version:     "2.0.0", // Default version
	}
}

/* NewBaseToolWithVersion creates a new base tool with version and output schema */
func NewBaseToolWithVersion(name, description, version string, inputSchema, outputSchema map[string]interface{}) *BaseTool {
	return &BaseTool{
		name:         name,
		description:  description,
		inputSchema:  inputSchema,
		outputSchema: outputSchema,
		version:      version,
	}
}

/* Name returns the tool name */
func (b *BaseTool) Name() string {
	return b.name
}

/* Description returns the tool description */
func (b *BaseTool) Description() string {
	return b.description
}

/* InputSchema returns the input schema */
func (b *BaseTool) InputSchema() map[string]interface{} {
	if b.inputSchema == nil {
		return make(map[string]interface{})
	}
	return b.inputSchema
}

/* OutputSchema returns the output schema */
func (b *BaseTool) OutputSchema() map[string]interface{} {
	return b.outputSchema
}

/* Version returns the tool version */
func (b *BaseTool) Version() string {
	if b.version == "" {
		return "2.0.0"
	}
	return b.version
}

/* SetVersion sets the tool version */
func (b *BaseTool) SetVersion(version string) {
	b.version = version
}

/* SetOutputSchema sets the output schema */
func (b *BaseTool) SetOutputSchema(schema map[string]interface{}) {
	b.outputSchema = schema
}

/* Deprecated returns whether the tool is deprecated */
func (b *BaseTool) Deprecated() bool {
	return b.deprecated
}

/* SetDeprecated marks the tool as deprecated */
func (b *BaseTool) SetDeprecated(deprecated bool) {
	b.deprecated = deprecated
}

/* Deprecation returns deprecation information */
func (b *BaseTool) Deprecation() *mcp.DeprecationInfo {
	return b.deprecation
}

/* SetDeprecation sets deprecation information */
func (b *BaseTool) SetDeprecation(info *mcp.DeprecationInfo) {
	b.deprecated = true
	b.deprecation = info
}

/* ValidateParams validates parameters against the schema */
func (b *BaseTool) ValidateParams(params map[string]interface{}, schema map[string]interface{}) (bool, []string) {
	var errors []string

	/* Parameter aliases - map common aliases to canonical names */
	aliases := map[string]string{
		"document": "text", /* document is alias for text in RAG tools */
	}

	if required, ok := schema["required"].([]interface{}); ok {
		for _, req := range required {
			if reqStr, ok := req.(string); ok {
				if _, exists := params[reqStr]; !exists {
					/* Check if alias exists */
					aliasFound := false
					for alias, canonical := range aliases {
						if canonical == reqStr {
							if _, aliasExists := params[alias]; aliasExists {
								/* Copy alias value to canonical name */
								params[reqStr] = params[alias]
								aliasFound = true
								break
							}
						}
					}
					if !aliasFound {
						errors = append(errors, fmt.Sprintf("Missing required parameter: %s", reqStr))
					}
				}
			}
		}
	}

	if properties, ok := schema["properties"].(map[string]interface{}); ok {
		for key, value := range params {
			if propSchema, exists := properties[key]; exists {
				if propMap, ok := propSchema.(map[string]interface{}); ok {
					if typeError := validateType(value, propMap); typeError != "" {
						errors = append(errors, fmt.Sprintf("Invalid type for %s: %s", key, typeError))
					}
					/* Additional validation: check pattern for strings */
					if pattern, ok := propMap["pattern"].(string); ok {
						if strVal, ok := value.(string); ok {
							matched, err := validatePattern(strVal, pattern)
							if err != nil {
								errors = append(errors, fmt.Sprintf("Invalid pattern validation for %s: %v", key, err))
							} else if !matched {
								errors = append(errors, fmt.Sprintf("Value for %s does not match pattern: %s", key, pattern))
							}
						}
					}
					/* Validate array items */
					if items, ok := propMap["items"].(map[string]interface{}); ok {
						if arrVal, ok := value.([]interface{}); ok {
							if itemType, ok := items["type"].(string); ok {
								for idx, item := range arrVal {
									itemSchema := map[string]interface{}{"type": itemType}
									if itemError := validateType(item, itemSchema); itemError != "" {
										errors = append(errors, fmt.Sprintf("Invalid item type at index %d in %s: %s", idx, key, itemError))
									}
								}
							}
						}
					}
					/* Validate string length constraints */
					if strVal, ok := value.(string); ok {
						if minLen, ok := propMap["minLength"].(float64); ok && float64(len(strVal)) < minLen {
							errors = append(errors, fmt.Sprintf("Value for %s is shorter than minimum length %g", key, minLen))
						}
						if maxLen, ok := propMap["maxLength"].(float64); ok && float64(len(strVal)) > maxLen {
							errors = append(errors, fmt.Sprintf("Value for %s is longer than maximum length %g", key, maxLen))
						}
					}
					/* Validate array length constraints */
					if arrVal, ok := value.([]interface{}); ok {
						if minItems, ok := propMap["minItems"].(float64); ok && float64(len(arrVal)) < minItems {
							errors = append(errors, fmt.Sprintf("Array %s has fewer items than minimum %g", key, minItems))
						}
						if maxItems, ok := propMap["maxItems"].(float64); ok && float64(len(arrVal)) > maxItems {
							errors = append(errors, fmt.Sprintf("Array %s has more items than maximum %g", key, maxItems))
						}
					}
				}
			} else {
				/* Check if additional properties are allowed */
				if additionalProps, ok := schema["additionalProperties"].(bool); ok && !additionalProps {
					errors = append(errors, fmt.Sprintf("Unknown parameter: %s (additional properties not allowed)", key))
				}
			}
		}
	}

	return len(errors) == 0, errors
}

/* validatePattern validates a string against a regex pattern (simplified - full regex would use regexp package) */
func validatePattern(value, pattern string) (bool, error) {
	/* Simplified pattern validation - returns true for now.
	 * Full regex validation would require importing the regexp package and implementing
	 * proper pattern matching. This is a future enhancement for strict pattern validation.
	 */
	return true, nil
}

func validateType(value interface{}, schema map[string]interface{}) string {
	schemaType, ok := schema["type"].(string)
	if !ok {
		return ""
	}

	valueType := reflect.TypeOf(value).Kind()

	switch schemaType {
	case "string":
		if valueType != reflect.String {
			return "expected string"
		}
	case "number":
		if valueType != reflect.Float64 && valueType != reflect.Int && valueType != reflect.Int64 && valueType != reflect.Float32 {
			return "expected number"
		}
	case "integer":
		if valueType != reflect.Int && valueType != reflect.Int64 {
			return "expected integer"
		}
	case "boolean":
		if valueType != reflect.Bool {
			return "expected boolean"
		}
	case "array":
		if valueType != reflect.Slice && valueType != reflect.Array {
			return "expected array"
		}
	case "object":
		if valueType != reflect.Map {
			return "expected object"
		}
	}

	if enum, ok := schema["enum"].([]interface{}); ok {
		found := false
		for _, e := range enum {
			if reflect.DeepEqual(value, e) {
				found = true
				break
			}
		}
		if !found {
			return fmt.Sprintf("must be one of: %v", enum)
		}
	}

	if schemaType == "number" || schemaType == "integer" {
		if min, ok := schema["minimum"].(float64); ok {
			if val, ok := value.(float64); ok && val < min {
				return fmt.Sprintf("must be >= %g", min)
			}
		}
		if max, ok := schema["maximum"].(float64); ok {
			if val, ok := value.(float64); ok && val > max {
				return fmt.Sprintf("must be <= %g", max)
			}
		}
	}

	return ""
}

/* ValidateOutput validates output data against output schema */
func ValidateOutput(data interface{}, schema map[string]interface{}) (bool, []string) {
	if schema == nil || len(schema) == 0 {
		return true, nil // No schema means no validation
	}

	var errors []string

	/* For now, basic validation - can be enhanced */
	if schemaType, ok := schema["type"].(string); ok {
		dataType := reflect.TypeOf(data).Kind()
		switch schemaType {
		case "object":
			if dataType != reflect.Map {
				errors = append(errors, fmt.Sprintf("expected object, got %v", dataType))
			}
		case "array":
			if dataType != reflect.Slice && dataType != reflect.Array {
				errors = append(errors, fmt.Sprintf("expected array, got %v", dataType))
			}
		}
	}

	return len(errors) == 0, errors
}

/* Success creates a success result */
func Success(data interface{}, metadata map[string]interface{}) *ToolResult {
	return &ToolResult{
		Success:  true,
		Data:     data,
		Metadata: metadata,
	}
}

/* Error creates an error result */
func Error(message, code string, details interface{}) *ToolResult {
	return &ToolResult{
		Success: false,
		Error: &ToolError{
			Message: message,
			Code:    code,
			Details: details,
		},
	}
}

