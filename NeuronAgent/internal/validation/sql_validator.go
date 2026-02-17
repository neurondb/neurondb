/*-------------------------------------------------------------------------
 *
 * sql_validator.go
 *    SQL query validation for NeuronAgent
 *
 * Provides secure SQL query validation to prevent SQL injection and
 * ensure only safe, read-only queries are executed.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/validation/sql_validator.go
 *
 *-------------------------------------------------------------------------
 */

package validation

import (
	"fmt"
	"regexp"
	"strings"
)

/* stripSQLComments removes SQL comments to prevent bypass of keyword checks.
 * Removes -- line comments and block (slash-star) comments. */
func stripSQLComments(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	upper := strings.ToUpper(s)
	i := 0
	for i < len(s) {
		if i+1 < len(s) && upper[i:i+2] == "--" {
			for i < len(s) && s[i] != '\n' {
				i++
			}
			continue
		}
		if i+1 < len(s) && upper[i:i+2] == "/*" {
			i += 2
			for i+1 < len(s) && upper[i:i+2] != "*/" {
				i++
			}
			if i+1 < len(s) {
				i += 2
			}
			continue
		}
		b.WriteByte(s[i])
		i++
	}
	return b.String()
}

/* Pre-compiled regex patterns for dangerous SQL keywords */
var (
	/* Word boundary patterns for each dangerous keyword */
	dangerousKeywordPatterns = []*regexp.Regexp{
		regexp.MustCompile(`\bDROP\b`),
		regexp.MustCompile(`\bDELETE\b`),
		regexp.MustCompile(`\bUPDATE\b`),
		regexp.MustCompile(`\bINSERT\b`),
		regexp.MustCompile(`\bALTER\b`),
		regexp.MustCompile(`\bCREATE\b`),
		regexp.MustCompile(`\bTRUNCATE\b`),
		regexp.MustCompile(`\bINTO\b`),
		regexp.MustCompile(`\bRETURNING\b`),
		regexp.MustCompile(`\bEXEC\b`),
		regexp.MustCompile(`\bEXECUTE\b`),
		regexp.MustCompile(`\bCALL\b`),
		regexp.MustCompile(`\bCOPY\b`),
		regexp.MustCompile(`\bGRANT\b`),
		regexp.MustCompile(`\bREVOKE\b`),
		regexp.MustCompile(`\bBEGIN\b`),
		regexp.MustCompile(`\bCOMMIT\b`),
		regexp.MustCompile(`\bROLLBACK\b`),
		regexp.MustCompile(`\bSAVEPOINT\b`),
		regexp.MustCompile(`\bRELEASE\b`),
	}

	/* Keyword names for error messages */
	dangerousKeywords = []string{
		"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE",
		"INTO", "RETURNING", "EXEC", "EXECUTE", "CALL", "COPY", "GRANT", "REVOKE",
		"BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE",
	}
)

/* AllowedQueryType represents the type of SQL query allowed */
type AllowedQueryType int

const (
	/* AllowReadOnly allows SELECT, EXPLAIN, SHOW, DESCRIBE */
	AllowReadOnly AllowedQueryType = iota
	/* AllowSelectOnly allows only SELECT queries */
	AllowSelectOnly
)

/* ValidationResult contains validation result details */
type ValidationResult struct {
	Valid       bool
	QueryType   string
	ForbiddenKeywords []string
	Error       error
}

/* ValidateSQLQuery validates a SQL query for security and allowed operations
 *
 * Parameters:
 *   - query: The SQL query string to validate
 *   - allowedType: The type of queries allowed (AllowReadOnly or AllowSelectOnly)
 *
 * Returns:
 *   - ValidationResult with validation details
 */
func ValidateSQLQuery(query string, allowedType AllowedQueryType) ValidationResult {
	/* Strip comments to prevent bypass (e.g. SELECT plus comment then DROP) */
	queryNoComments := stripSQLComments(query)
	/* Normalize query */
	queryUpper := strings.TrimSpace(strings.ToUpper(queryNoComments))
	if queryUpper == "" {
		return ValidationResult{
			Valid: false,
			Error: fmt.Errorf("query cannot be empty"),
		}
	}

	/* Determine and validate query type */
	var queryType string
	var queryTypeValid bool

	if strings.HasPrefix(queryUpper, "SELECT") {
		queryType = "SELECT"
		queryTypeValid = true
	} else if strings.HasPrefix(queryUpper, "EXPLAIN") {
		queryType = "EXPLAIN"
		queryTypeValid = (allowedType == AllowReadOnly)
	} else if strings.HasPrefix(queryUpper, "SHOW") {
		queryType = "SHOW"
		queryTypeValid = (allowedType == AllowReadOnly)
	} else if strings.HasPrefix(queryUpper, "DESCRIBE") {
		queryType = "DESCRIBE"
		queryTypeValid = (allowedType == AllowReadOnly)
	} else if strings.HasPrefix(queryUpper, "\\d") {
		/* PostgreSQL \d command for describe */
		queryType = "DESCRIBE"
		queryTypeValid = (allowedType == AllowReadOnly)
	} else {
		queryType = "UNKNOWN"
		queryTypeValid = false
	}

	if !queryTypeValid {
		queryPreview := query
		if len(queryPreview) > 100 {
			queryPreview = queryPreview[:100] + "..."
		}
		var allowedTypes string
		if allowedType == AllowSelectOnly {
			allowedTypes = "SELECT"
		} else {
			allowedTypes = "SELECT, EXPLAIN, SHOW, DESCRIBE"
		}
		return ValidationResult{
			Valid:     false,
			QueryType: queryType,
			Error:     fmt.Errorf("only %s queries are allowed, got query type: %s (query_preview: %s)", allowedTypes, queryType, queryPreview),
		}
	}

	/* Check for dangerous keywords on comment-stripped query to prevent bypass */
	var foundKeywords []string
	for i, pattern := range dangerousKeywordPatterns {
		if pattern.MatchString(queryUpper) {
			foundKeywords = append(foundKeywords, dangerousKeywords[i])
		}
	}

	if len(foundKeywords) > 0 {
		queryPreview := query
		if len(queryPreview) > 100 {
			queryPreview = queryPreview[:100] + "..."
		}
		return ValidationResult{
			Valid:            false,
			QueryType:        queryType,
			ForbiddenKeywords: foundKeywords,
			Error:            fmt.Errorf("query contains forbidden keywords: %v (query_preview: %s)", foundKeywords, queryPreview),
		}
	}

	return ValidationResult{
		Valid:     true,
		QueryType: queryType,
	}
}

/* ValidateSQLQuerySimple is a convenience function that returns error directly
 *
 * This is useful for code that wants simple error-based validation.
 */
func ValidateSQLQuerySimple(query string, allowedType AllowedQueryType) error {
	result := ValidateSQLQuery(query, allowedType)
	if !result.Valid {
		return result.Error
	}
	return nil
}
