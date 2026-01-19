/*-------------------------------------------------------------------------
 *
 * memory_management_adapter.go
 *    Adapter for memory management tool interfaces
 *
 * Bridges agent memory management components with tool interfaces
 * to enable MemoryTool advanced features without circular dependencies.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/memory_management_adapter.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
)

/* MemoryManagementAdapter adapts agent components to MemoryManagementInterface */
type MemoryManagementAdapter struct {
	corruptionDetector *MemoryCorruptionDetector
	forgettingManager  *MemoryForgettingManager
	conflictResolver   *MemoryConflictResolver
	qualityScorer      *MemoryQualityScorer
}

/* NewMemoryManagementAdapter creates a new memory management adapter */
func NewMemoryManagementAdapter(
	corruptionDetector *MemoryCorruptionDetector,
	forgettingManager *MemoryForgettingManager,
	conflictResolver *MemoryConflictResolver,
	qualityScorer *MemoryQualityScorer,
) *MemoryManagementAdapter {
	return &MemoryManagementAdapter{
		corruptionDetector: corruptionDetector,
		forgettingManager:  forgettingManager,
		conflictResolver:   conflictResolver,
		qualityScorer:      qualityScorer,
	}
}

/* CheckCorruption checks for memory corruption */
func (a *MemoryManagementAdapter) CheckCorruption(ctx context.Context, agentID uuid.UUID) ([]map[string]interface{}, error) {
	if a.corruptionDetector == nil {
		return nil, fmt.Errorf("corruption detector not available")
	}

	issues, err := a.corruptionDetector.ValidateMemoryIntegrity(ctx, agentID)
	if err != nil {
		return nil, err
	}

	/* Convert to map format */
	result := make([]map[string]interface{}, len(issues))
	for i, issue := range issues {
		result[i] = map[string]interface{}{
			"memory_id":    issue.MemoryID.String(),
			"tier":         issue.Tier,
			"issue_type":   issue.IssueType,
			"description":  issue.Description,
			"severity":     issue.Severity,
			"repairable":   issue.Repairable,
		}
	}

	return result, nil
}

/* ForgetMemories triggers intelligent forgetting */
func (a *MemoryManagementAdapter) ForgetMemories(ctx context.Context, agentID uuid.UUID, tier string, config map[string]interface{}) (int, error) {
	if a.forgettingManager == nil {
		return 0, fmt.Errorf("forgetting manager not available")
	}

	forgettingConfig := DefaultForgettingConfig()
	
	/* Override with provided config if available */
	if strategy, ok := config["strategy"].(string); ok {
		forgettingConfig.Strategy = ForgettingStrategy(strategy)
	}
	if timeThreshold, ok := config["time_threshold_hours"].(float64); ok {
		forgettingConfig.TimeThreshold = time.Duration(timeThreshold) * time.Hour
	}
	if importanceThreshold, ok := config["importance_threshold"].(float64); ok {
		forgettingConfig.ImportanceThreshold = importanceThreshold
	}

	return a.forgettingManager.ForgetMemories(ctx, agentID, tier, forgettingConfig)
}

/* ResolveConflicts resolves memory conflicts */
func (a *MemoryManagementAdapter) ResolveConflicts(ctx context.Context, agentID uuid.UUID, strategy string) (int, error) {
	if a.conflictResolver == nil {
		return 0, fmt.Errorf("conflict resolver not available")
	}

	/* Detect conflicts */
	conflicts, err := a.conflictResolver.DetectConflicts(ctx, agentID)
	if err != nil {
		return 0, err
	}

	/* Resolve conflicts */
	resolved := 0
	resolutionStrategy := ResolutionStrategyTimestamp
	switch strategy {
	case "confidence":
		resolutionStrategy = ResolutionStrategyConfidence
	case "source":
		resolutionStrategy = ResolutionStrategySource
	case "llm":
		resolutionStrategy = ResolutionStrategyLLM
	case "merge":
		resolutionStrategy = ResolutionStrategyMerge
	}

	for _, conflict := range conflicts {
		if !conflict.Resolved {
			if err := a.conflictResolver.ResolveConflict(ctx, conflict, resolutionStrategy); err == nil {
				resolved++
			}
		}
	}

	return resolved, nil
}

/* GetMemoryQuality gets quality scores for a memory */
func (a *MemoryManagementAdapter) GetMemoryQuality(ctx context.Context, agentID, memoryID uuid.UUID, tier string) (map[string]interface{}, error) {
	if a.qualityScorer == nil {
		return nil, fmt.Errorf("quality scorer not available")
	}

	score, err := a.qualityScorer.ScoreMemory(ctx, agentID, memoryID, tier)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"completeness":  score.Completeness,
		"accuracy":      score.Accuracy,
		"relevance":     score.Relevance,
		"freshness":     score.Freshness,
		"consistency":   score.Consistency,
		"overall_score": score.OverallScore,
	}, nil
}
