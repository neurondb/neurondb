/*-------------------------------------------------------------------------
 *
 * json.go
 *    JSON schema validation for NeuronMCP
 *
 * Provides JSON schema validation and structure checking.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/validation/json.go
 *
 *-------------------------------------------------------------------------
 */

package validation

import (
	"encoding/json"
	"fmt"
	"reflect"
)

/* ValidateJSONSchema validates a value against a JSON schema structure */
func ValidateJSONSchema(value interface{}, schema map[string]interface{}) error {
	if schema == nil {
		return nil // No schema means no validation
	}
	
	props, ok := schema["properties"].(map[string]interface{})
	if !ok {
		return nil // No properties means no validation
	}
	
	valueMap, ok := value.(map[string]interface{})
	if !ok {
		return fmt.Errorf("value must be a map/object for schema validation")
	}
	
	/* Check required fields */
	if required, ok := schema["required"].([]interface{}); ok {
		for _, req := range required {
			reqStr, ok := req.(string)
			if !ok {
				continue
			}
			if _, exists := valueMap[reqStr]; !exists {
				return fmt.Errorf("required field '%s' is missing", reqStr)
			}
		}
	}
	
	/* Validate each property */
	for propName, propSchema := range props {
		propSchemaMap, ok := propSchema.(map[string]interface{})
		if !ok {
			continue
		}
		
		propValue, exists := valueMap[propName]
		if !exists {
			continue // Optional field
		}
		
		if err := validateProperty(propValue, propSchemaMap, propName); err != nil {
			return fmt.Errorf("property '%s': %w", propName, err)
		}
	}
	
	/* Check additionalProperties */
	if additionalProps, ok := schema["additionalProperties"].(bool); ok && !additionalProps {
		for key := range valueMap {
			if _, exists := props[key]; !exists {
				return fmt.Errorf("additional property '%s' is not allowed", key)
			}
		}
	}
	
	return nil
}

/* validateProperty validates a single property against its schema */
func validateProperty(value interface{}, schema map[string]interface{}, fieldName string) error {
	/* Check type */
	if typeStr, ok := schema["type"].(string); ok {
		if err := validateType(value, typeStr, fieldName); err != nil {
			return err
		}
	}
	
	/* Validate string constraints */
	if typeStr, _ := schema["type"].(string); typeStr == "string" {
		strValue, ok := value.(string)
		if !ok {
			return fmt.Errorf("expected string, got %T", value)
		}
		
		if minLen, ok := schema["minLength"].(float64); ok {
			if len(strValue) < int(minLen) {
				return fmt.Errorf("length %d is less than minimum %d", len(strValue), int(minLen))
			}
		}
		
		if maxLen, ok := schema["maxLength"].(float64); ok {
			if len(strValue) > int(maxLen) {
				return fmt.Errorf("length %d exceeds maximum %d", len(strValue), int(maxLen))
			}
		}
	}
	
	/* Validate number constraints */
	if typeStr, _ := schema["type"].(string); typeStr == "number" || typeStr == "integer" {
		var numValue float64
		switch v := value.(type) {
		case float64:
			numValue = v
		case float32:
			numValue = float64(v)
		case int:
			numValue = float64(v)
		case int32:
			numValue = float64(v)
		case int64:
			numValue = float64(v)
		default:
			return fmt.Errorf("expected number, got %T", value)
		}
		
		if minimum, ok := schema["minimum"].(float64); ok {
			if numValue < minimum {
				return fmt.Errorf("value %f is less than minimum %f", numValue, minimum)
			}
		}
		
		if maximum, ok := schema["maximum"].(float64); ok {
			if numValue > maximum {
				return fmt.Errorf("value %f exceeds maximum %f", numValue, maximum)
			}
		}
	}
	
	/* Validate array constraints */
	if typeStr, _ := schema["type"].(string); typeStr == "array" {
		arrValue, ok := value.([]interface{})
		if !ok {
			return fmt.Errorf("expected array, got %T", value)
		}
		
		if minItems, ok := schema["minItems"].(float64); ok {
			if len(arrValue) < int(minItems) {
				return fmt.Errorf("array length %d is less than minimum %d", len(arrValue), int(minItems))
			}
		}
		
		if maxItems, ok := schema["maxItems"].(float64); ok {
			if len(arrValue) > int(maxItems) {
				return fmt.Errorf("array length %d exceeds maximum %d", len(arrValue), int(maxItems))
			}
		}
	}
	
	/* Validate enum */
	if enum, ok := schema["enum"].([]interface{}); ok {
		valueJSON, _ := json.Marshal(value)
		found := false
		for _, enumVal := range enum {
			enumJSON, _ := json.Marshal(enumVal)
			if string(valueJSON) == string(enumJSON) {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("value is not in enum list")
		}
	}
	
	return nil
}

/* validateType validates that a value matches the expected JSON schema type */
func validateType(value interface{}, expectedType, fieldName string) error {
	switch expectedType {
	case "string":
		if _, ok := value.(string); !ok {
			return fmt.Errorf("expected string, got %T", value)
		}
	case "number":
		switch value.(type) {
		case float64, float32, int, int32, int64:
			/* Valid number types */
		default:
			return fmt.Errorf("expected number, got %T", value)
		}
	case "integer":
		switch value.(type) {
		case int, int32, int64, float64, float32:
			/* Accept floats that are whole numbers */
			if reflect.TypeOf(value).Kind() == reflect.Float64 || reflect.TypeOf(value).Kind() == reflect.Float32 {
				f := reflect.ValueOf(value).Float()
				if f != float64(int64(f)) {
					return fmt.Errorf("expected integer, got float with decimal part")
				}
			}
		default:
			return fmt.Errorf("expected integer, got %T", value)
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("expected boolean, got %T", value)
		}
	case "array":
		if _, ok := value.([]interface{}); !ok {
			return fmt.Errorf("expected array, got %T", value)
		}
	case "object":
		if _, ok := value.(map[string]interface{}); !ok {
			return fmt.Errorf("expected object, got %T", value)
		}
	case "null":
		if value != nil {
			return fmt.Errorf("expected null, got %T", value)
		}
	default:
		return fmt.Errorf("unknown type: %s", expectedType)
	}
	
	return nil
}



