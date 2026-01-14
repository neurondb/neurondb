/*-------------------------------------------------------------------------
 *
 * common.go
 *    Common validation functions for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/validation/common.go
 *
 *-------------------------------------------------------------------------
 */

package validation

import (
	"fmt"
	"regexp"
	"strings"
)

var (
	emailRegex = regexp.MustCompile(`^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$`)
	urlRegex   = regexp.MustCompile(`^https?://[a-zA-Z0-9.\-]+(:[0-9]+)?(/.*)?$`)
)

/* ValidateRequired checks if a string is non-empty */
func ValidateRequired(value, fieldName string) error {
	if value == "" {
		return fmt.Errorf("%s is required and cannot be empty", fieldName)
	}
	return nil
}

/* ValidateMaxLength checks if a string length is within limit */
func ValidateMaxLength(value, fieldName string, maxLength int) error {
	if len(value) > maxLength {
		return fmt.Errorf("%s length %d exceeds maximum %d", fieldName, len(value), maxLength)
	}
	return nil
}

/* ValidateMinLength checks if a string length meets minimum */
func ValidateMinLength(value, fieldName string, minLength int) error {
	if len(value) < minLength {
		return fmt.Errorf("%s length %d is below minimum %d", fieldName, len(value), minLength)
	}
	return nil
}

/* ValidateIntRange checks if an integer is within range */
func ValidateIntRange(value, min, max int, fieldName string) error {
	if value < min || value > max {
		return fmt.Errorf("%s value %d is outside valid range [%d, %d]", fieldName, value, min, max)
	}
	return nil
}

/* ValidatePositive checks if an integer is positive */
func ValidatePositive(value int, fieldName string) error {
	if value <= 0 {
		return fmt.Errorf("%s must be positive, got %d", fieldName, value)
	}
	return nil
}

/* ValidateNonNegative checks if an integer is non-negative */
func ValidateNonNegative(value int, fieldName string) error {
	if value < 0 {
		return fmt.Errorf("%s cannot be negative, got %d", fieldName, value)
	}
	return nil
}

/* ValidateFloatRange checks if a float is within range */
func ValidateFloatRange(value, min, max float64, fieldName string) error {
	if value < min || value > max {
		return fmt.Errorf("%s value %f is outside valid range [%f, %f]", fieldName, value, min, max)
	}
	return nil
}

/* ValidateEmail validates email format */
func ValidateEmail(email, fieldName string) error {
	if !emailRegex.MatchString(email) {
		return fmt.Errorf("%s is not a valid email address: %s", fieldName, email)
	}
	return nil
}

/* ValidateURL validates URL format */
func ValidateURL(url, fieldName string) error {
	if !urlRegex.MatchString(url) {
		return fmt.Errorf("%s is not a valid URL: %s", fieldName, url)
	}
	return nil
}

/* ValidateIn checks if a value is in a list of allowed values */
func ValidateIn(value, fieldName string, allowed ...string) error {
	for _, a := range allowed {
		if value == a {
			return nil
		}
	}
	return fmt.Errorf("%s value '%s' is not in allowed list: %v", fieldName, value, allowed)
}

/* ValidateNoNullBytes checks for null bytes in string (security) */
func ValidateNoNullBytes(value, fieldName string) error {
	if strings.Contains(value, "\x00") {
		return fmt.Errorf("%s contains null bytes (security violation)", fieldName)
	}
	return nil
}

/* ValidatePattern validates string against regex pattern */
func ValidatePattern(value, fieldName, pattern string) error {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return fmt.Errorf("invalid pattern for %s: %w", fieldName, err)
	}
	if !re.MatchString(value) {
		return fmt.Errorf("%s does not match required pattern: %s", fieldName, pattern)
	}
	return nil
}

/* ValidateNotIn checks if a value is NOT in a list of forbidden values */
func ValidateNotIn(value, fieldName string, forbidden ...string) error {
	for _, f := range forbidden {
		if value == f {
			return fmt.Errorf("%s value '%s' is forbidden", fieldName, value)
		}
	}
	return nil
}

/* ValidateAlphanumeric checks if string contains only alphanumeric characters */
func ValidateAlphanumeric(value, fieldName string) error {
	for _, r := range value {
		if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9')) {
			return fmt.Errorf("%s must contain only alphanumeric characters", fieldName)
		}
	}
	return nil
}

/* ValidateAlphanumericWithUnderscore allows alphanumeric and underscore */
func ValidateAlphanumericWithUnderscore(value, fieldName string) error {
	for _, r := range value {
		if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_') {
			return fmt.Errorf("%s must contain only alphanumeric characters and underscores", fieldName)
		}
	}
	return nil
}

/* ValidateNoLeadingTrailingSpaces checks for leading/trailing whitespace */
func ValidateNoLeadingTrailingSpaces(value, fieldName string) error {
	trimmed := strings.TrimSpace(value)
	if value != trimmed {
		return fmt.Errorf("%s has leading or trailing whitespace", fieldName)
	}
	return nil
}

/* ValidateNotEmpty checks if slice/array is not empty */
func ValidateNotEmptySlice(slice []interface{}, fieldName string) error {
	if len(slice) == 0 {
		return fmt.Errorf("%s cannot be empty", fieldName)
	}
	return nil
}

/* ValidateSliceLength checks if slice length is within bounds */
func ValidateSliceLength(slice []interface{}, fieldName string, minLen, maxLen int) error {
	length := len(slice)
	if length < minLen {
		return fmt.Errorf("%s length %d is below minimum %d", fieldName, length, minLen)
	}
	if length > maxLen {
		return fmt.Errorf("%s length %d exceeds maximum %d", fieldName, length, maxLen)
	}
	return nil
}

/* ValidateLimit validates limit parameter for pagination */
func ValidateLimit(limit int) error {
	if limit < 0 {
		return fmt.Errorf("limit cannot be negative: %d", limit)
	}
	if limit > 10000 {
		return fmt.Errorf("limit %d exceeds maximum 10000", limit)
	}
	return nil
}

/* ValidateOffset validates offset parameter for pagination */
func ValidateOffset(offset int) error {
	if offset < 0 {
		return fmt.Errorf("offset cannot be negative: %d", offset)
	}
	return nil
}
