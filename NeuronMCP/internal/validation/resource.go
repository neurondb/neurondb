/*-------------------------------------------------------------------------
 *
 * resource.go
 *    Resource quota validation for NeuronMCP
 *
 * Provides resource quota validation (memory, CPU limits).
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/validation/resource.go
 *
 *-------------------------------------------------------------------------
 */

package validation

import (
	"fmt"
)

/* ResourceQuota represents resource limits */
type ResourceQuota struct {
	MaxMemoryBytes int64
	MaxCPUTimeMs   int64
	MaxVectorSize  int
	MaxBatchSize   int
}

/* DefaultResourceQuota returns default resource quotas */
func DefaultResourceQuota() ResourceQuota {
	return ResourceQuota{
		MaxMemoryBytes: 100 * 1024 * 1024, // 100MB
		MaxCPUTimeMs:   30000,              // 30 seconds
		MaxVectorSize:  10000,               // 10k dimensions
		MaxBatchSize:   1000,                // 1000 items
	}
}

/* ValidateMemoryUsage validates memory usage against quota */
func ValidateMemoryUsage(usedBytes int64, quota ResourceQuota) error {
	if usedBytes > quota.MaxMemoryBytes {
		return fmt.Errorf("memory usage %d bytes exceeds quota %d bytes", usedBytes, quota.MaxMemoryBytes)
	}
	return nil
}

/* ValidateVectorSize validates vector size against quota */
func ValidateVectorSize(size int, quota ResourceQuota) error {
	if size > quota.MaxVectorSize {
		return fmt.Errorf("vector size %d exceeds maximum %d", size, quota.MaxVectorSize)
	}
	return nil
}

/* ValidateBatchSize validates batch size against quota */
func ValidateBatchSize(size int, quota ResourceQuota) error {
	if size > quota.MaxBatchSize {
		return fmt.Errorf("batch size %d exceeds maximum %d", size, quota.MaxBatchSize)
	}
	return nil
}

/* EstimateVectorMemory estimates memory usage for a vector operation */
func EstimateVectorMemory(vectorDim int, batchSize int) int64 {
	/* Estimate: 4 bytes per float32 * dimensions * batch size */
	/* Plus overhead for metadata, results, etc. (2x multiplier) */
	return int64(vectorDim * batchSize * 4 * 2)
}



