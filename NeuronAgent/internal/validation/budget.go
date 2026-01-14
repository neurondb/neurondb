/*-------------------------------------------------------------------------
 *
 * budget.go
 *    Budget validation for NeuronAgent
 *
 * Provides budget enforcement and cost tracking validation.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/validation/budget.go
 *
 *-------------------------------------------------------------------------
 */

package validation

import (
	"fmt"
)

/* BudgetLimits represents budget constraints */
type BudgetLimits struct {
	MaxCostUSD        float64
	MaxTokens         int64
	MaxRequests       int64
	MaxRequestsPerDay int64
}

/* ValidateCost validates cost against budget */
func ValidateCost(costUSD float64, budget BudgetLimits) error {
	if budget.MaxCostUSD > 0 && costUSD > budget.MaxCostUSD {
		return fmt.Errorf("cost $%.4f exceeds budget limit $%.4f", costUSD, budget.MaxCostUSD)
	}
	return nil
}

/* ValidateTokens validates token count against budget */
func ValidateTokens(tokens int64, budget BudgetLimits) error {
	if budget.MaxTokens > 0 && tokens > budget.MaxTokens {
		return fmt.Errorf("token count %d exceeds budget limit %d", tokens, budget.MaxTokens)
	}
	return nil
}

/* ValidateRequestCount validates request count against budget */
func ValidateRequestCount(count int64, budget BudgetLimits) error {
	if budget.MaxRequests > 0 && count > budget.MaxRequests {
		return fmt.Errorf("request count %d exceeds budget limit %d", count, budget.MaxRequests)
	}
	return nil
}

/* EstimateCost estimates cost based on tokens and model */
func EstimateCost(tokens int64, modelName string) float64 {
	/* Rough cost estimates per 1K tokens (as of 2024) */
	costPer1K := map[string]float64{
		"gpt-4":           0.03,
		"gpt-4-turbo":     0.01,
		"gpt-3.5-turbo":   0.002,
		"claude-3-opus":   0.015,
		"claude-3-sonnet": 0.003,
		"claude-3-haiku":  0.00025,
	}

	cost, ok := costPer1K[modelName]
	if !ok {
		cost = 0.002 // Default to gpt-3.5-turbo pricing
	}

	return float64(tokens) / 1000.0 * cost
}


