/*-------------------------------------------------------------------------
 *
 * memory_learning_test.go
 *    Tests for memory learning
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/memory_learning_test.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"testing"
)

/* TestMemoryLearningManager_calculateQualityScore tests quality score calculation */
func TestMemoryLearningManager_calculateQualityScore(t *testing.T) {
	manager := &MemoryLearningManager{}

	tests := []struct {
		positiveCount int
		negativeCount int
		avgRelevance  float64
		expectedMin   float64
		expectedMax   float64
	}{
		{5, 0, 0.9, 0.8, 1.0},
		{0, 5, 0.1, 0.0, 0.3},
		{3, 2, 0.7, 0.5, 0.8},
		{0, 0, 0.5, 0.4, 0.6},
	}

	for _, tt := range tests {
		result := manager.calculateQualityScore(tt.positiveCount, tt.negativeCount, tt.avgRelevance)
		if result < tt.expectedMin || result > tt.expectedMax {
			t.Errorf("calculateQualityScore(%d, %d, %f) = %f, want between %f and %f",
				tt.positiveCount, tt.negativeCount, tt.avgRelevance, result, tt.expectedMin, tt.expectedMax)
		}
	}
}
