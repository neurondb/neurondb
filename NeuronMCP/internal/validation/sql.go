/*-------------------------------------------------------------------------
 *
 * sql.go
 *    SQL validation for NeuronMCP
 *
 * Provides comprehensive SQL injection prevention and identifier validation.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/validation/sql.go
 *
 *-------------------------------------------------------------------------
 */

package validation

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
)

var (
	/* SQL identifier regex: alphanumeric, underscore, dollar sign, must start with letter or underscore */
	sqlIdentifierRegex = regexp.MustCompile(`^[a-zA-Z_][a-zA-Z0-9_$]*$`)
	
	/* Dangerous SQL keywords that should not appear in user input */
	dangerousKeywords = map[string]bool{
		"DROP": true, "TRUNCATE": true, "DELETE": true, "UPDATE": true,
		"INSERT": true, "ALTER": true, "CREATE": true, "GRANT": true,
		"REVOKE": true, "EXECUTE": true, "CALL": true, "COPY": true,
		"VACUUM": true, "ANALYZE": true, "REINDEX": true, "CLUSTER": true,
	}
	
	/* SQL injection patterns */
	sqlInjectionPatterns = []string{
		";--", "';--", "'; DROP", "'; DELETE", "'; UPDATE",
		"UNION SELECT", "OR 1=1", "OR '1'='1", "OR \"1\"=\"1",
		"'; EXEC", "'; EXECUTE", "'; CALL", "'; COPY",
	}
)

/* ValidateSQLIdentifier validates a SQL identifier (table, column name) */
func ValidateSQLIdentifier(identifier, fieldName string) error {
	if identifier == "" {
		return fmt.Errorf("%s cannot be empty", fieldName)
	}
	
	identifier = strings.TrimSpace(identifier)
	
	/* Check length (PostgreSQL limit is 63 bytes, but we'll be more conservative) */
	if len(identifier) > 63 {
		return fmt.Errorf("%s exceeds maximum length of 63 characters: %s", fieldName, identifier)
	}
	
	/* Check for SQL identifier format */
	if !sqlIdentifierRegex.MatchString(identifier) {
		return fmt.Errorf("%s contains invalid characters: %s (must start with letter/underscore, followed by alphanumeric/underscore/dollar)", fieldName, identifier)
	}
	
	/* Check for reserved keywords (case-insensitive) */
	upperIdentifier := strings.ToUpper(identifier)
	if dangerousKeywords[upperIdentifier] {
		return fmt.Errorf("%s is a reserved SQL keyword and cannot be used: %s", fieldName, identifier)
	}
	
	/* Check for SQL injection patterns */
	upperIdentifierLower := strings.ToLower(identifier)
	for _, pattern := range sqlInjectionPatterns {
		if strings.Contains(upperIdentifierLower, strings.ToLower(pattern)) {
			return fmt.Errorf("%s contains potentially malicious SQL pattern: %s", fieldName, identifier)
		}
	}
	
	return nil
}

/* ValidateSQLIdentifierRequired validates a SQL identifier and ensures it's not empty */
func ValidateSQLIdentifierRequired(identifier, fieldName string) error {
	if identifier == "" {
		return fmt.Errorf("%s is required and cannot be empty", fieldName)
	}
	return ValidateSQLIdentifier(identifier, fieldName)
}

/* ValidateSQLQuery validates a SQL query for safety (read-only) */
func ValidateSQLQuery(query string) error {
	if query == "" {
		return fmt.Errorf("SQL query cannot be empty")
	}
	
	query = strings.TrimSpace(query)
	queryUpper := strings.ToUpper(query)
	
	/* Must start with SELECT for read-only queries */
	if !strings.HasPrefix(queryUpper, "SELECT") {
		return fmt.Errorf("only SELECT queries are allowed, query must start with SELECT")
	}
	
	for keyword := range dangerousKeywords {
		/* Check for keyword as separate word (with spaces or newlines around it) */
		pattern := " " + keyword + " "
		if strings.Contains(queryUpper, pattern) || 
		   strings.Contains(queryUpper, "\n"+keyword+" ") ||
		   strings.Contains(queryUpper, "\t"+keyword+" ") {
			return fmt.Errorf("dangerous SQL operation detected: %s (only SELECT queries are allowed)", keyword)
		}
	}
	
	/* Check for SQL injection patterns */
	queryLower := strings.ToLower(query)
	for _, pattern := range sqlInjectionPatterns {
		if strings.Contains(queryLower, pattern) {
			return fmt.Errorf("potentially malicious SQL pattern detected in query")
		}
	}
	
	return nil
}

/* EscapeSQLIdentifier escapes a SQL identifier for safe use */
func EscapeSQLIdentifier(identifier string) string {
	/* Remove any non-printable characters */
	var builder strings.Builder
	for _, r := range identifier {
		if unicode.IsPrint(r) {
			builder.WriteRune(r)
		}
	}
	result := builder.String()
	
	/* PostgreSQL identifier escaping: wrap in double quotes */
	return fmt.Sprintf(`"%s"`, strings.ReplaceAll(result, `"`, `""`))
}

/* ValidateSchemaName validates a PostgreSQL schema name */
func ValidateSchemaName(schemaName string) error {
	return ValidateSQLIdentifier(schemaName, "schema_name")
}

/* ValidateTableName validates a PostgreSQL table name */
func ValidateTableName(tableName string) error {
	return ValidateSQLIdentifier(tableName, "table_name")
}

/* ValidateColumnName validates a PostgreSQL column name */
func ValidateColumnName(columnName string) error {
	return ValidateSQLIdentifier(columnName, "column_name")
}



