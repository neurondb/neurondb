/*-------------------------------------------------------------------------
 *
 * common.go
 *    Common validation functions for NeuronAgent
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/validation/common.go
 *
 *-------------------------------------------------------------------------
 */

package validation

import (
	"fmt"
	"io"
	"net/http"
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

/* ReadAndValidateBody reads and validates HTTP request body size */
func ReadAndValidateBody(r *http.Request, maxSize int64) ([]byte, error) {
	if r.Body == nil {
		return nil, fmt.Errorf("request body is nil")
	}

	/* Create a limited reader to prevent DoS */
	limitedReader := io.LimitReader(r.Body, maxSize+1)
	bodyBytes, err := io.ReadAll(limitedReader)
	if err != nil {
		return nil, fmt.Errorf("failed to read request body: %w", err)
	}

	if int64(len(bodyBytes)) > maxSize {
		return nil, fmt.Errorf("request body size %d exceeds maximum %d bytes", len(bodyBytes), maxSize)
	}

	return bodyBytes, nil
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

/* ValidateIntRange validates integer is within range */
func ValidateIntRange(value, min, max int, fieldName string) error {
	if value < min || value > max {
		return fmt.Errorf("%s value %d is outside valid range [%d, %d]", fieldName, value, min, max)
	}
	return nil
}

/* ValidateNonNegative validates integer is non-negative */
func ValidateNonNegative(value int, fieldName string) error {
	if value < 0 {
		return fmt.Errorf("%s cannot be negative, got %d", fieldName, value)
	}
	return nil
}
