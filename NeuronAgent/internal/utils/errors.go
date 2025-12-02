package utils

import (
	"fmt"
	"strings"
)

// SanitizeValue sanitizes sensitive data in error messages
func SanitizeValue(value interface{}) string {
	if value == nil {
		return "<nil>"
	}
	
	str := fmt.Sprintf("%v", value)
	
	// Sanitize passwords, tokens, and other sensitive data
	if strings.Contains(strings.ToLower(str), "password") ||
		strings.Contains(strings.ToLower(str), "token") ||
		strings.Contains(strings.ToLower(str), "secret") ||
		strings.Contains(strings.ToLower(str), "key") {
		return "<redacted>"
	}
	
	// Truncate long strings
	if len(str) > 100 {
		return str[:100] + "..."
	}
	
	return str
}

// FormatConnectionInfo formats database connection information for error messages
func FormatConnectionInfo(host string, port int, database string, user string) string {
	return fmt.Sprintf("database '%s' on host '%s:%d' as user '%s'", database, host, port, user)
}

// FormatQueryContext formats query execution context for error messages
func FormatQueryContext(query string, paramCount int, operation string, table string) string {
	// Truncate long queries
	queryPreview := query
	if len(queryPreview) > 200 {
		queryPreview = queryPreview[:200] + "..."
	}
	
	parts := []string{
		fmt.Sprintf("query='%s'", queryPreview),
		fmt.Sprintf("params_count=%d", paramCount),
		fmt.Sprintf("operation=%s", operation),
	}
	
	if table != "" {
		parts = append(parts, fmt.Sprintf("table='%s'", table))
	}
	
	return strings.Join(parts, ", ")
}

// FormatToolContext formats tool execution context for error messages
func FormatToolContext(toolName string, handlerType string, argCount int, argKeys []string) string {
	parts := []string{
		fmt.Sprintf("tool_name='%s'", toolName),
		fmt.Sprintf("handler_type='%s'", handlerType),
		fmt.Sprintf("args_count=%d", argCount),
	}
	
	if len(argKeys) > 0 {
		keysStr := strings.Join(argKeys, ", ")
		if len(keysStr) > 100 {
			keysStr = keysStr[:100] + "..."
		}
		parts = append(parts, fmt.Sprintf("arg_keys=[%s]", keysStr))
	}
	
	return strings.Join(parts, ", ")
}

// FormatParamValues formats parameter values for error messages (sanitized)
func FormatParamValues(params []interface{}) string {
	if len(params) == 0 {
		return "[]"
	}
	
	var values []string
	for i, param := range params {
		if i >= 5 { // Limit to first 5 params
			values = append(values, fmt.Sprintf("... (%d more)", len(params)-5))
			break
		}
		values = append(values, SanitizeValue(param))
	}
	
	return "[" + strings.Join(values, ", ") + "]"
}

// BuildErrorContext builds a detailed error message with context
func BuildErrorContext(operation string, resourceType string, resourceName string, resourceID string, details string, err error) string {
	parts := []string{operation}
	
	if resourceType != "" {
		part := fmt.Sprintf("%s '%s'", resourceType, resourceName)
		if resourceID != "" {
			part += fmt.Sprintf(" [%s]", resourceID)
		}
		parts = append(parts, part)
	}
	
	if details != "" {
		parts = append(parts, details)
	}
	
	msg := strings.Join(parts, " on ")
	
	if err != nil {
		return fmt.Sprintf("%s, error=%v", msg, err)
	}
	
	return msg
}

