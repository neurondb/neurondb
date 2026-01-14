/*-------------------------------------------------------------------------
 *
 * validation_test.go
 *    Tests for validation package
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 *-------------------------------------------------------------------------
 */

package validation

import (
	"math"
	"testing"
)

func TestValidateUUID(t *testing.T) {
	tests := []struct {
		name    string
		uuid    string
		wantErr bool
	}{
		{"valid UUID", "550e8400-e29b-41d4-a716-446655440000", false},
		{"invalid UUID", "not-a-uuid", true},
		{"empty UUID", "", true},
		{"malformed UUID", "550e8400-e29b-41d4", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateUUID(tt.uuid)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateUUID() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateSQLIdentifier(t *testing.T) {
	tests := []struct {
		name    string
		ident   string
		wantErr bool
	}{
		{"valid identifier", "my_table", false},
		{"valid with underscore", "my_table_name", false},
		{"invalid with space", "my table", true},
		{"invalid with dash", "my-table", true},
		{"invalid reserved keyword", "DROP", true},
		{"empty", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateSQLIdentifier(tt.ident, "test_field")
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateSQLIdentifier() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateVector(t *testing.T) {
	tests := []struct {
		name    string
		vector  []interface{}
		minDim  int
		maxDim  int
		wantErr bool
	}{
		{"valid vector", []interface{}{1.0, 2.0, 3.0}, 1, 10, false},
		{"empty vector", []interface{}{}, 1, 10, true},
		{"nil vector", nil, 1, 10, true},
		{"too small", []interface{}{1.0}, 2, 10, true},
		{"too large", []interface{}{1.0, 2.0, 3.0, 4.0, 5.0}, 1, 3, true},
		{"with NaN", []interface{}{1.0, math.NaN(), 3.0}, 1, 10, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateVector(tt.vector, "test_field", tt.minDim, tt.maxDim)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateVector() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateIntRange(t *testing.T) {
	tests := []struct {
		name    string
		value   int
		min     int
		max     int
		wantErr bool
	}{
		{"valid in range", 5, 1, 10, false},
		{"at minimum", 1, 1, 10, false},
		{"at maximum", 10, 1, 10, false},
		{"below minimum", 0, 1, 10, true},
		{"above maximum", 11, 1, 10, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateIntRange(tt.value, tt.min, tt.max, "test_field")
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateIntRange() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

