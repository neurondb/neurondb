package utils

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
)

/* ValidationError represents a validation error */
type ValidationError struct {
	Field   string
	Message string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("%s: %s", e.Field, e.Message)
}

/* ValidateProfile validates a profile configuration */
func ValidateProfile(name, dsn string, mcpConfig map[string]interface{}) []error {
	var errors []error

	if strings.TrimSpace(name) == "" {
		errors = append(errors, &ValidationError{
			Field:   "name",
			Message: "name is required",
		})
	} else if len(name) > 100 {
		errors = append(errors, &ValidationError{
			Field:   "name",
			Message: "name must be less than 100 characters",
		})
	}

	if strings.TrimSpace(dsn) == "" {
		errors = append(errors, &ValidationError{
			Field:   "neurondb_dsn",
			Message: "NeuronDB DSN is required",
		})
	} else if !isValidDSN(dsn) {
		errors = append(errors, &ValidationError{
			Field:   "neurondb_dsn",
			Message: "invalid DSN format",
		})
	}

	if mcpConfig != nil {
		if command, ok := mcpConfig["command"].(string); ok {
			if strings.TrimSpace(command) == "" {
				errors = append(errors, &ValidationError{
					Field:   "mcp_config.command",
					Message: "MCP command is required if mcp_config is provided",
				})
			}
		}
	}

	return errors
}

/* MeetsPasswordComplexity returns true if password has upper, lower, digit, and special character */
func MeetsPasswordComplexity(password string) bool {
	var hasUpper, hasLower, hasDigit, hasSpecial bool
	for _, r := range password {
		switch {
		case unicode.IsUpper(r):
			hasUpper = true
		case unicode.IsLower(r):
			hasLower = true
		case unicode.IsNumber(r):
			hasDigit = true
		case unicode.IsPunct(r) || unicode.IsSymbol(r):
			hasSpecial = true
		}
	}
	return hasUpper && hasLower && hasDigit && hasSpecial
}

/* ValidateAPIKey validates an API key format */
func ValidateAPIKey(key string) error {
	if len(key) < 32 {
		return &ValidationError{
			Field:   "api_key",
			Message: "API key must be at least 32 characters",
		}
	}

	base64URLRegex := regexp.MustCompile(`^[A-Za-z0-9_-]+$`)
	if !base64URLRegex.MatchString(key) {
		return &ValidationError{
			Field:   "api_key",
			Message: "API key contains invalid characters",
		}
	}

	return nil
}

/* ValidateSearchRequest validates a search request */
func ValidateSearchRequest(collection string, limit int, distanceType string) []error {
	var errors []error

	if strings.TrimSpace(collection) == "" {
		errors = append(errors, &ValidationError{
			Field:   "collection",
			Message: "collection is required",
		})
	}

	if limit < 1 || limit > 1000 {
		errors = append(errors, &ValidationError{
			Field:   "limit",
			Message: "limit must be between 1 and 1000",
		})
	}

	validDistanceTypes := map[string]bool{
		"l2":            true,
		"cosine":        true,
		"inner_product": true,
		"euclidean":     true,
		"dot":           true,
	}

	if distanceType != "" && !validDistanceTypes[strings.ToLower(distanceType)] {
		errors = append(errors, &ValidationError{
			Field:   "distance_type",
			Message: fmt.Sprintf("invalid distance type. Must be one of: %v", getKeys(validDistanceTypes)),
		})
	}

	return errors
}

/* stripSQLComments removes SQL comments to prevent bypass of keyword checks */
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

/* ValidateSQL validates SQL query for safety.
 * Strips comments first to prevent bypass (e.g. SELECT plus comment plus DROP). */
func ValidateSQL(query string) error {
	noComments := stripSQLComments(query)
	queryUpper := strings.ToUpper(strings.TrimSpace(noComments))

	if !strings.HasPrefix(queryUpper, "SELECT") {
		return &ValidationError{
			Field:   "query",
			Message: "only SELECT queries are allowed",
		}
	}

	/* Use word-boundary regex so "DROP" in "SELECT ... DROP ..." is caught even with comments stripped */
	dangerousKeywords := []*regexp.Regexp{
		regexp.MustCompile(`\bDROP\b`),
		regexp.MustCompile(`\bTRUNCATE\b`),
		regexp.MustCompile(`\bDELETE\b`),
		regexp.MustCompile(`\bUPDATE\b`),
		regexp.MustCompile(`\bINSERT\b`),
		regexp.MustCompile(`\bALTER\b`),
		regexp.MustCompile(`\bCREATE\b`),
		regexp.MustCompile(`\bGRANT\b`),
		regexp.MustCompile(`\bREVOKE\b`),
		regexp.MustCompile(`\bEXECUTE\b`),
		regexp.MustCompile(`\bCALL\b`),
		regexp.MustCompile(`\bCOPY\b`),
		regexp.MustCompile(`\bVACUUM\b`),
		regexp.MustCompile(`\bANALYZE\b`),
	}
	for _, re := range dangerousKeywords {
		if re.MatchString(queryUpper) {
			return &ValidationError{
				Field:   "query",
				Message: "dangerous SQL operation detected",
			}
		}
	}

	sqlInjectionPatterns := []string{
		";--", "';--", "'; DROP", "'; DELETE",
		"UNION SELECT", "OR 1=1", "OR '1'='1",
	}
	queryUpperOrig := strings.ToUpper(query)
	for _, pattern := range sqlInjectionPatterns {
		if strings.Contains(queryUpperOrig, pattern) {
			return &ValidationError{
				Field:   "query",
				Message: "potentially malicious SQL pattern detected",
			}
		}
	}

	return nil
}

/* ValidateToolCall validates a tool call request */
func ValidateToolCall(toolName string, arguments map[string]interface{}) []error {
	var errors []error

	if strings.TrimSpace(toolName) == "" {
		errors = append(errors, &ValidationError{
			Field:   "name",
			Message: "tool name is required",
		})
	}

	toolNameRegex := regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)
	if !toolNameRegex.MatchString(toolName) {
		errors = append(errors, &ValidationError{
			Field:   "name",
			Message: "tool name contains invalid characters",
		})
	}

	return errors
}

func isValidDSN(dsn string) bool {
	required := []string{"host=", "user=", "dbname="}
	dsnLower := strings.ToLower(dsn)
	for _, req := range required {
		if !strings.Contains(dsnLower, req) {
			return false
		}
	}
	return true
}

func getKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

/* SanitizeString sanitizes a string input */
func SanitizeString(s string, maxLength int) string {
	var builder strings.Builder
	for _, r := range s {
		if unicode.IsPrint(r) || unicode.IsSpace(r) {
			builder.WriteRune(r)
		}
	}
	result := builder.String()

	result = strings.TrimSpace(result)
	if maxLength > 0 && len(result) > maxLength {
		result = result[:maxLength]
	}

	return result
}

/* ValidateEmail validates an email address (basic) */
func ValidateEmail(email string) error {
	if email == "" {
		return &ValidationError{
			Field:   "email",
			Message: "email is required",
		}
	}

	emailRegex := regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)
	if !emailRegex.MatchString(email) {
		return &ValidationError{
			Field:   "email",
			Message: "invalid email format",
		}
	}

	return nil
}

/* ValidateUUID validates a UUID string */
func ValidateUUID(uuid string) error {
	uuidRegex := regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)
	if !uuidRegex.MatchString(strings.ToLower(uuid)) {
		return &ValidationError{
			Field:   "uuid",
			Message: "invalid UUID format",
		}
	}
	return nil
}
