/*-------------------------------------------------------------------------
 *
 * validation.go
 *    Input validation utilities for NeuronAgent
 *
 * Provides validation functions for email addresses, URLs, and other
 * common input types with regex-based pattern matching.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/utils/validation.go
 *
 *-------------------------------------------------------------------------
 */

package utils

import (
	"fmt"
	"net/url"
	"regexp"
	"strings"
)

var (
	emailRegex = regexp.MustCompile(`^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$`)
	urlRegex   = regexp.MustCompile(`^https?://[^\s/$.?#].[^\s]*$`)
)

/* ValidateEmail validates an email address */
func ValidateEmail(email string) bool {
	return emailRegex.MatchString(email)
}

/* ValidateURL validates a URL */
func ValidateURL(urlStr string) bool {
	if !urlRegex.MatchString(urlStr) {
		return false
	}
	_, err := url.Parse(urlStr)
	return err == nil
}

/* ValidateRequired checks if a string is not empty */
func ValidateRequired(s string) bool {
	return strings.TrimSpace(s) != ""
}

/* ValidateLength checks if string length is within range */
func ValidateLength(s string, min, max int) bool {
	length := len(strings.TrimSpace(s))
	return length >= min && length <= max
}

/* ValidateMinLength checks if string meets minimum length */
func ValidateMinLength(s string, min int) bool {
	return len(strings.TrimSpace(s)) >= min
}

/* ValidateMaxLength checks if string doesn't exceed maximum length */
func ValidateMaxLength(s string, max int) bool {
	return len(strings.TrimSpace(s)) <= max
}

/* ValidateIn checks if value is in allowed list */
func ValidateIn(value string, allowed ...string) bool {
	for _, a := range allowed {
		if value == a {
			return true
		}
	}
	return false
}

/* ValidateRegex validates string against regex pattern */
func ValidateRegex(s, pattern string) bool {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return false
	}
	return re.MatchString(s)
}

/* ValidateUUID validates UUID format */
func ValidateUUID(s string) bool {
	return IsValidUUID(s)
}

/* ValidateIntRange validates integer is in range */
func ValidateIntRange(n, min, max int) bool {
	return n >= min && n <= max
}

/* ValidateFloatRange validates float is in range */
func ValidateFloatRange(f, min, max float64) bool {
	return f >= min && f <= max
}

/* ValidateAll validates all validators and returns first error */
func ValidateAll(validators ...func() error) error {
	for _, validator := range validators {
		if err := validator(); err != nil {
			return err
		}
	}
	return nil
}

/* ValidateEmailWithError validates email and returns error */
func ValidateEmailWithError(email string) error {
	if !ValidateEmail(email) {
		return fmt.Errorf("invalid email format: %s", email)
	}
	return nil
}

/* ValidateURLWithError validates URL and returns error */
func ValidateURLWithError(urlStr string) error {
	if !ValidateURL(urlStr) {
		return fmt.Errorf("invalid URL format: %s", urlStr)
	}
	return nil
}

/* ValidateRequiredWithError validates required field and returns error */
func ValidateRequiredWithError(s, fieldName string) error {
	if !ValidateRequired(s) {
		return fmt.Errorf("%s is required", fieldName)
	}
	return nil
}
