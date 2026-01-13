/*-------------------------------------------------------------------------
 *
 * data_masking.go
 *    Data masking for sensitive columns
 *
 * Implements data masking for sensitive columns as specified in Phase 2.1.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/security/data_masking.go
 *
 *-------------------------------------------------------------------------
 */

package security

import (
	"fmt"
	"strings"
)

/* MaskingStrategy represents a data masking strategy */
type MaskingStrategy string

const (
	MaskingStrategyFull    MaskingStrategy = "full"     /* Replace with asterisks */
	MaskingStrategyPartial MaskingStrategy = "partial"  /* Show first/last N chars */
	MaskingStrategyHash     MaskingStrategy = "hash"     /* Hash the value */
	MaskingStrategyRedact   MaskingStrategy = "redact"   /* Replace with [REDACTED] */
)

/* DataMasker masks sensitive data */
type DataMasker struct {
	maskedColumns map[string]MaskingStrategy /* column_name -> strategy */
}

/* NewDataMasker creates a new data masker */
func NewDataMasker() *DataMasker {
	return &DataMasker{
		maskedColumns: make(map[string]MaskingStrategy),
	}
}

/* AddMaskedColumn adds a column to be masked */
func (d *DataMasker) AddMaskedColumn(columnName string, strategy MaskingStrategy) {
	d.maskedColumns[strings.ToLower(columnName)] = strategy
}

/* MaskValue masks a value based on strategy */
func (d *DataMasker) MaskValue(value interface{}, columnName string) interface{} {
	strategy, exists := d.maskedColumns[strings.ToLower(columnName)]
	if !exists {
		return value /* No masking needed */
	}

	valueStr := fmt.Sprintf("%v", value)
	if valueStr == "" {
		return value
	}

	switch strategy {
	case MaskingStrategyFull:
		return strings.Repeat("*", len(valueStr))
	case MaskingStrategyPartial:
		if len(valueStr) <= 4 {
			return strings.Repeat("*", len(valueStr))
		}
		/* Show first 2 and last 2 characters */
		return valueStr[:2] + strings.Repeat("*", len(valueStr)-4) + valueStr[len(valueStr)-2:]
	case MaskingStrategyHash:
		/* Simple hash - in production use proper hashing */
		return fmt.Sprintf("hash_%d", len(valueStr))
	case MaskingStrategyRedact:
		return "[REDACTED]"
	default:
		return value
	}
}

/* MaskRow masks sensitive columns in a row */
func (d *DataMasker) MaskRow(row map[string]interface{}) map[string]interface{} {
	masked := make(map[string]interface{})
	for key, value := range row {
		masked[key] = d.MaskValue(value, key)
	}
	return masked
}

/* MaskRows masks sensitive columns in multiple rows */
func (d *DataMasker) MaskRows(rows []map[string]interface{}) []map[string]interface{} {
	masked := make([]map[string]interface{}, len(rows))
	for i, row := range rows {
		masked[i] = d.MaskRow(row)
	}
	return masked
}



