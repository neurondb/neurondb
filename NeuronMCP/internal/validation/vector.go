/*-------------------------------------------------------------------------
 *
 * vector.go
 *    Vector validation for NeuronMCP
 *
 * Provides comprehensive vector dimension and value validation.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/validation/vector.go
 *
 *-------------------------------------------------------------------------
 */

package validation

import (
	"fmt"
	"math"
)

/* ValidateVector validates a vector (slice of floats) */
func ValidateVector(vector []interface{}, fieldName string, minDim, maxDim int) error {
	if vector == nil {
		return fmt.Errorf("%s cannot be nil", fieldName)
	}
	
	if len(vector) == 0 {
		return fmt.Errorf("%s cannot be empty", fieldName)
	}
	
	if minDim > 0 && len(vector) < minDim {
		return fmt.Errorf("%s dimension %d is less than minimum %d", fieldName, len(vector), minDim)
	}
	
	if maxDim > 0 && len(vector) > maxDim {
		return fmt.Errorf("%s dimension %d exceeds maximum %d", fieldName, len(vector), maxDim)
	}
	
	/* Validate all elements are numbers */
	for i, v := range vector {
		switch val := v.(type) {
		case float64:
			if math.IsNaN(val) {
				return fmt.Errorf("%s contains NaN at index %d", fieldName, i)
			}
			if math.IsInf(val, 0) {
				return fmt.Errorf("%s contains Infinity at index %d", fieldName, i)
			}
		case float32:
			if math.IsNaN(float64(val)) {
				return fmt.Errorf("%s contains NaN at index %d", fieldName, i)
			}
			if math.IsInf(float64(val), 0) {
				return fmt.Errorf("%s contains Infinity at index %d", fieldName, i)
			}
		case int, int32, int64:
			/* Integers are fine, will be converted to float */
		default:
			return fmt.Errorf("%s contains non-numeric value at index %d: %T", fieldName, i, v)
		}
	}
	
	return nil
}

/* ValidateVectorDimension validates vector dimension matches expected */
func ValidateVectorDimension(vector []interface{}, expectedDim int, fieldName string) error {
	if len(vector) != expectedDim {
		return fmt.Errorf("%s dimension %d does not match expected dimension %d", fieldName, len(vector), expectedDim)
	}
	return nil
}

/* ValidateVectorRequired validates a vector and ensures it's not empty */
func ValidateVectorRequired(vector []interface{}, fieldName string, minDim, maxDim int) error {
	if vector == nil || len(vector) == 0 {
		return fmt.Errorf("%s is required and cannot be empty", fieldName)
	}
	return ValidateVector(vector, fieldName, minDim, maxDim)
}

/* ValidateVectorConsistency validates that multiple vectors have consistent dimensions */
func ValidateVectorConsistency(vectors [][]interface{}, fieldName string) error {
	if len(vectors) == 0 {
		return nil
	}
	
	firstDim := len(vectors[0])
	for i, vec := range vectors[1:] {
		if len(vec) != firstDim {
			return fmt.Errorf("%s: vector at index %d has dimension %d, expected %d", fieldName, i+1, len(vec), firstDim)
		}
	}
	
	return nil
}

/* ValidateVectorNormalized validates that a vector is normalized (unit length) */
func ValidateVectorNormalized(vector []interface{}, fieldName string, tolerance float64) error {
	if tolerance <= 0 {
		tolerance = 1e-6
	}
	
	var sumSquares float64
	for _, v := range vector {
		var val float64
		switch x := v.(type) {
		case float64:
			val = x
		case float32:
			val = float64(x)
		case int:
			val = float64(x)
		case int32:
			val = float64(x)
		case int64:
			val = float64(x)
		default:
			return fmt.Errorf("%s contains non-numeric value", fieldName)
		}
		sumSquares += val * val
	}
	
	length := math.Sqrt(sumSquares)
	diff := math.Abs(length - 1.0)
	
	if diff > tolerance {
		return fmt.Errorf("%s is not normalized: length is %.6f, expected 1.0 (difference: %.6f)", fieldName, length, diff)
	}
	
	return nil
}



