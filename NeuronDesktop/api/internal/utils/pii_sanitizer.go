package utils

import (
	"encoding/json"
	"regexp"
	"strings"
)

/* Sensitive field patterns that should be sanitized */
var sensitivePatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)(password|passwd|pwd)`),
	regexp.MustCompile(`(?i)(api[_-]?key|apikey)`),
	regexp.MustCompile(`(?i)(secret|token|auth)`),
	regexp.MustCompile(`(?i)(credential|cred)`),
	regexp.MustCompile(`(?i)(ssn|social[_-]?security)`),
	regexp.MustCompile(`(?i)(credit[_-]?card|card[_-]?number)`),
	regexp.MustCompile(`(?i)(email)`),
}

/* SanitizeValue sanitizes a value if it matches sensitive patterns */
func SanitizeValue(key string, value interface{}) interface{} {
	keyLower := strings.ToLower(key)

	/* Check if key matches sensitive patterns */
	for _, pattern := range sensitivePatterns {
		if pattern.MatchString(keyLower) {
			return "[REDACTED]"
		}
	}

	/* If value is a string and looks like sensitive data, sanitize it */
	if str, ok := value.(string); ok {
		if len(str) > 0 {
			/* Check if it looks like a token/key (long alphanumeric string) */
			if len(str) > 20 && isLikelySensitive(str) {
				return "[REDACTED]"
			}
		}
	}

	return value
}

/* isLikelySensitive checks if a string looks like sensitive data */
func isLikelySensitive(s string) bool {
	/* Check for long alphanumeric strings (likely tokens/keys) */
	if matched, _ := regexp.MatchString(`^[A-Za-z0-9_-]{20,}$`, s); matched {
		return true
	}

	/* Check for email-like patterns */
	if matched, _ := regexp.MatchString(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`, s); matched {
		return true
	}

	return false
}

/* SanitizeMap recursively sanitizes a map structure */
func SanitizeMap(data map[string]interface{}) map[string]interface{} {
	sanitized := make(map[string]interface{})

	for key, value := range data {
		switch v := value.(type) {
		case map[string]interface{}:
			sanitized[key] = SanitizeMap(v)
		case []interface{}:
			sanitized[key] = SanitizeArray(v)
		default:
			sanitized[key] = SanitizeValue(key, v)
		}
	}

	return sanitized
}

/* SanitizeArray recursively sanitizes an array structure */
func SanitizeArray(data []interface{}) []interface{} {
	sanitized := make([]interface{}, len(data))

	for i, item := range data {
		switch v := item.(type) {
		case map[string]interface{}:
			sanitized[i] = SanitizeMap(v)
		case []interface{}:
			sanitized[i] = SanitizeArray(v)
		default:
			sanitized[i] = SanitizeValue("", v)
		}
	}

	return sanitized
}

/* SanitizeJSON sanitizes JSON data */
func SanitizeJSON(jsonData []byte) ([]byte, error) {
	var data interface{}
	if err := json.Unmarshal(jsonData, &data); err != nil {
		return jsonData, err
	}

	var sanitized interface{}
	switch v := data.(type) {
	case map[string]interface{}:
		sanitized = SanitizeMap(v)
	case []interface{}:
		sanitized = SanitizeArray(v)
	default:
		sanitized = v
	}

	return json.Marshal(sanitized)
}








