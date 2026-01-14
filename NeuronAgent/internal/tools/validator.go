/*-------------------------------------------------------------------------
 *
 * validator.go
 *    Comprehensive JSON Schema validator for tool arguments
 *
 * Provides full JSON Schema validation including types, constraints,
 * patterns, enums, nested objects, arrays, and more.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/validator.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"fmt"
	"math"
	"reflect"
	"regexp"
	"strings"
	"time"
)

/* ValidateArgs validates arguments against a JSON Schema */
func ValidateArgs(args map[string]interface{}, schema map[string]interface{}) error {
	/* Validate schema structure */
	if schema == nil {
		return fmt.Errorf("invalid schema: schema is nil")
	}

	properties, ok := schema["properties"].(map[string]interface{})
	if !ok {
		/* If no properties, allow any args (empty schema) */
		if len(args) > 0 {
			return fmt.Errorf("invalid schema: missing properties but arguments provided")
		}
		return nil
	}

	/* Get required fields */
	required, _ := schema["required"].([]interface{})
	requiredSet := make(map[string]bool)
	for _, req := range required {
		if reqStr, ok := req.(string); ok {
			requiredSet[reqStr] = true
		}
	}

	/* Check required fields */
	for fieldName := range requiredSet {
		if _, exists := args[fieldName]; !exists {
			return fmt.Errorf("missing required field: %s", fieldName)
		}
	}

	/* Validate each argument */
	for key, value := range args {
		propSchema, exists := properties[key]
		if !exists {
			/* Check if additionalProperties is false */
			if additionalProps, ok := schema["additionalProperties"].(bool); ok && !additionalProps {
				return fmt.Errorf("unknown field: %s (additionalProperties is false)", key)
			}
			/* Allow extra fields by default */
			continue
		}

		propMap, ok := propSchema.(map[string]interface{})
		if !ok {
			return fmt.Errorf("invalid schema for field %s: property definition must be an object", key)
		}

		if err := validateProperty(key, value, propMap); err != nil {
			return fmt.Errorf("validation failed for field '%s': %w", key, err)
		}
	}

	return nil
}

/* validateProperty validates a single property against its schema */
func validateProperty(fieldName string, value interface{}, schema map[string]interface{}) error {
	/* Handle null values */
	if value == nil {
		/* Check if null is allowed */
		if typeVal, ok := schema["type"].(string); ok && typeVal == "null" {
			return nil
		}
		/* Check if nullable */
		if nullable, ok := schema["nullable"].(bool); ok && nullable {
			return nil
		}
		return fmt.Errorf("field cannot be null")
	}

	/* Validate type */
	expectedType, ok := schema["type"].(string)
	if !ok {
		/* No type constraint, skip type validation */
	} else {
		if err := validateType(value, expectedType); err != nil {
			return err
		}
	}

	/* Type-specific validations */
	switch expectedType {
	case "string":
		return validateString(value, schema)
	case "number", "integer":
		return validateNumber(value, schema)
	case "array":
		return validateArray(fieldName, value, schema)
	case "object":
		return validateObject(fieldName, value, schema)
	case "boolean":
		/* Boolean has no additional constraints */
		return nil
	}

	return nil
}

/* validateType validates the basic type */
func validateType(value interface{}, expectedType string) error {
	actualType := reflect.TypeOf(value).Kind()

	switch expectedType {
	case "string":
		if actualType != reflect.String {
			return fmt.Errorf("expected string, got %v", actualType)
		}
	case "number":
		/* Accept float64, int, int64 */
		if actualType != reflect.Float64 && actualType != reflect.Int && actualType != reflect.Int64 && actualType != reflect.Float32 && actualType != reflect.Int32 {
			return fmt.Errorf("expected number, got %v", actualType)
		}
	case "integer":
		/* Accept int, int64, int32 */
		if actualType != reflect.Int && actualType != reflect.Int64 && actualType != reflect.Int32 {
			/* Also accept float64 if it's a whole number */
			if actualType == reflect.Float64 {
				floatVal := value.(float64)
				if floatVal != math.Trunc(floatVal) {
					return fmt.Errorf("expected integer, got float64 with decimal part")
				}
				return nil
			}
			return fmt.Errorf("expected integer, got %v", actualType)
		}
	case "boolean":
		if actualType != reflect.Bool {
			return fmt.Errorf("expected boolean, got %v", actualType)
		}
	case "array":
		if actualType != reflect.Slice && actualType != reflect.Array {
			return fmt.Errorf("expected array, got %v", actualType)
		}
	case "object":
		if actualType != reflect.Map {
			return fmt.Errorf("expected object, got %v", actualType)
		}
	case "null":
		if value != nil {
			return fmt.Errorf("expected null, got %v", actualType)
		}
	default:
		/* Unknown type, skip validation */
	}

	return nil
}

/* validateString validates string-specific constraints */
func validateString(value interface{}, schema map[string]interface{}) error {
	str, ok := value.(string)
	if !ok {
		return fmt.Errorf("value is not a string")
	}

	/* Min length */
	if minLen, ok := schema["minLength"].(float64); ok {
		if float64(len(str)) < minLen {
			return fmt.Errorf("string length %d is less than minimum %d", len(str), int(minLen))
		}
	}

	/* Max length */
	if maxLen, ok := schema["maxLength"].(float64); ok {
		if float64(len(str)) > maxLen {
			return fmt.Errorf("string length %d exceeds maximum %d", len(str), int(maxLen))
		}
	}

	/* Pattern (regex) */
	if pattern, ok := schema["pattern"].(string); ok {
		matched, err := regexp.MatchString(pattern, str)
		if err != nil {
			return fmt.Errorf("invalid regex pattern: %w", err)
		}
		if !matched {
			return fmt.Errorf("string does not match required pattern: %s", pattern)
		}
	}

	/* Enum */
	if enum, ok := schema["enum"].([]interface{}); ok {
		found := false
		for _, enumVal := range enum {
			if enumStr, ok := enumVal.(string); ok && enumStr == str {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("string '%s' is not in allowed enum values", str)
		}
	}

	/* Format validation */
	if format, ok := schema["format"].(string); ok {
		if err := validateFormat(str, format); err != nil {
			return err
		}
	}

	return nil
}

/* validateNumber validates number-specific constraints */
func validateNumber(value interface{}, schema map[string]interface{}) error {
	var numVal float64

	switch v := value.(type) {
	case float64:
		numVal = v
	case int:
		numVal = float64(v)
	case int64:
		numVal = float64(v)
	case int32:
		numVal = float64(v)
	case float32:
		numVal = float64(v)
	default:
		return fmt.Errorf("value is not a number")
	}

	/* Minimum */
	if min, ok := schema["minimum"].(float64); ok {
		if numVal < min {
			return fmt.Errorf("number %v is less than minimum %v", numVal, min)
		}
	}

	/* Exclusive minimum */
	if exclMin, ok := schema["exclusiveMinimum"].(float64); ok {
		if numVal <= exclMin {
			return fmt.Errorf("number %v is less than or equal to exclusive minimum %v", numVal, exclMin)
		}
	}

	/* Maximum */
	if max, ok := schema["maximum"].(float64); ok {
		if numVal > max {
			return fmt.Errorf("number %v exceeds maximum %v", numVal, max)
		}
	}

	/* Exclusive maximum */
	if exclMax, ok := schema["exclusiveMaximum"].(float64); ok {
		if numVal >= exclMax {
			return fmt.Errorf("number %v is greater than or equal to exclusive maximum %v", numVal, exclMax)
		}
	}

	/* Multiple of */
	if multipleOf, ok := schema["multipleOf"].(float64); ok {
		if multipleOf != 0 {
			remainder := math.Mod(numVal, multipleOf)
			/* Handle floating point precision issues */
			epsilon := 1e-9
			if remainder > epsilon && remainder < (multipleOf-epsilon) {
				return fmt.Errorf("number %v is not a multiple of %v", numVal, multipleOf)
			}
		}
	}

	return nil
}

/* validateArray validates array-specific constraints */
func validateArray(fieldName string, value interface{}, schema map[string]interface{}) error {
	val := reflect.ValueOf(value)
	if val.Kind() != reflect.Slice && val.Kind() != reflect.Array {
		return fmt.Errorf("value is not an array")
	}

	length := val.Len()

	/* Min items */
	if minItems, ok := schema["minItems"].(float64); ok {
		if float64(length) < minItems {
			return fmt.Errorf("array length %d is less than minimum %d", length, int(minItems))
		}
	}

	/* Max items */
	if maxItems, ok := schema["maxItems"].(float64); ok {
		if float64(length) > maxItems {
			return fmt.Errorf("array length %d exceeds maximum %d", length, int(maxItems))
		}
	}

	/* Unique items */
	if uniqueItems, ok := schema["uniqueItems"].(bool); ok && uniqueItems {
		seen := make(map[interface{}]bool)
		for i := 0; i < length; i++ {
			item := val.Index(i).Interface()
			/* Convert to comparable key */
			key := fmt.Sprintf("%v", item)
			if seen[key] {
				return fmt.Errorf("array contains duplicate items")
			}
			seen[key] = true
		}
	}

	/* Items schema */
	if itemsSchema, ok := schema["items"].(map[string]interface{}); ok {
		for i := 0; i < length; i++ {
			item := val.Index(i).Interface()
			if err := validateProperty(fmt.Sprintf("%s[%d]", fieldName, i), item, itemsSchema); err != nil {
				return fmt.Errorf("array item at index %d: %w", i, err)
			}
		}
	}

	return nil
}

/* validateObject validates object-specific constraints */
func validateObject(fieldName string, value interface{}, schema map[string]interface{}) error {
	val := reflect.ValueOf(value)
	if val.Kind() != reflect.Map {
		return fmt.Errorf("value is not an object")
	}

	/* Properties schema */
	if properties, ok := schema["properties"].(map[string]interface{}); ok {
		for _, key := range val.MapKeys() {
			keyStr := key.String()
			propValue := val.MapIndex(key).Interface()

			if propSchema, exists := properties[keyStr]; exists {
				propSchemaMap, ok := propSchema.(map[string]interface{})
				if !ok {
					continue
				}
				if err := validateProperty(fmt.Sprintf("%s.%s", fieldName, keyStr), propValue, propSchemaMap); err != nil {
					return err
				}
			}
		}
	}

	/* Required fields in nested object */
	if required, ok := schema["required"].([]interface{}); ok {
		for _, req := range required {
			reqStr, ok := req.(string)
			if !ok {
				continue
			}
			key := reflect.ValueOf(reqStr)
			if !val.MapIndex(key).IsValid() {
				return fmt.Errorf("missing required field in object: %s", reqStr)
			}
		}
	}

	/* Additional properties */
	if additionalProps, ok := schema["additionalProperties"].(bool); ok && !additionalProps {
		if properties, ok := schema["properties"].(map[string]interface{}); ok {
			for _, key := range val.MapKeys() {
				keyStr := key.String()
				if _, exists := properties[keyStr]; !exists {
					return fmt.Errorf("unknown field in object: %s (additionalProperties is false)", keyStr)
				}
			}
		}
	}

	/* Min properties */
	if minProps, ok := schema["minProperties"].(float64); ok {
		if float64(val.Len()) < minProps {
			return fmt.Errorf("object has %d properties, but minimum is %d", val.Len(), int(minProps))
		}
	}

	/* Max properties */
	if maxProps, ok := schema["maxProperties"].(float64); ok {
		if float64(val.Len()) > maxProps {
			return fmt.Errorf("object has %d properties, but maximum is %d", val.Len(), int(maxProps))
		}
	}

	return nil
}

/* validateFormat validates string formats */
func validateFormat(str, format string) error {
	switch format {
	case "email":
		emailRegex := regexp.MustCompile(`^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$`)
		if !emailRegex.MatchString(str) {
			return fmt.Errorf("string is not a valid email address")
		}
	case "uri", "url":
		uriRegex := regexp.MustCompile(`^https?://[^\s/$.?#].[^\s]*$`)
		if !uriRegex.MatchString(str) {
			return fmt.Errorf("string is not a valid URI/URL")
		}
	case "date":
		/* Try common date formats */
		formats := []string{
			time.RFC3339,
			time.RFC3339Nano,
			"2006-01-02",
			"2006-01-02T15:04:05Z07:00",
		}
		var err error
		for _, format := range formats {
			_, err = time.Parse(format, str)
			if err == nil {
				break
			}
		}
		if err != nil {
			return fmt.Errorf("string is not a valid date")
		}
	case "date-time":
		_, err := time.Parse(time.RFC3339, str)
		if err != nil {
			return fmt.Errorf("string is not a valid date-time (RFC3339)")
		}
	case "uuid":
		uuidRegex := regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)
		if !uuidRegex.MatchString(strings.ToLower(str)) {
			return fmt.Errorf("string is not a valid UUID")
		}
	case "hostname":
		hostnameRegex := regexp.MustCompile(`^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$`)
		if !hostnameRegex.MatchString(str) {
			return fmt.Errorf("string is not a valid hostname")
		}
	case "ipv4":
		ipv4Regex := regexp.MustCompile(`^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$`)
		if !ipv4Regex.MatchString(str) {
			return fmt.Errorf("string is not a valid IPv4 address")
		}
	case "ipv6":
		/* Simplified IPv6 validation */
		ipv6Regex := regexp.MustCompile(`^([0-9a-fA-F]{0,4}:){7}[0-9a-fA-F]{0,4}$`)
		if !ipv6Regex.MatchString(str) {
			return fmt.Errorf("string is not a valid IPv6 address")
		}
	}
	return nil
}
