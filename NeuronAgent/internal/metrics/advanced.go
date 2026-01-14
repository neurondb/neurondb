/*-------------------------------------------------------------------------
 *
 * advanced.go
 *    Advanced metrics collection with detailed analytics
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/metrics/advanced.go
 *
 *-------------------------------------------------------------------------
 */

package metrics

import (
	"sync"
	"time"

	"github.com/google/uuid"
)

/* AdvancedMetrics collects detailed metrics */
type AdvancedMetrics struct {
	agentMetrics   map[string]*AgentMetrics
	toolMetrics    map[string]*ToolMetrics
	costMetrics    *CostMetrics
	qualityMetrics *QualityMetrics
	mu             sync.RWMutex
}

/* AgentMetrics tracks metrics for an agent */
type AgentMetrics struct {
	AgentID               uuid.UUID
	TotalExecutions       int64
	SuccessfulExecutions  int64
	FailedExecutions      int64
	TotalTokens           int64
	TotalCost             float64
	AvgExecutionTime      time.Duration
	AvgTokensPerExecution int64
	LastExecutionTime     time.Time
	ExecutionTimes        []time.Duration
	mu                    sync.RWMutex
}

/* ToolMetrics tracks metrics for a tool */
type ToolMetrics struct {
	ToolName         string
	TotalCalls       int64
	SuccessfulCalls  int64
	FailedCalls      int64
	AvgExecutionTime time.Duration
	TotalCost        float64
	CallTimes        []time.Duration
	mu               sync.RWMutex
}

/* CostMetrics tracks cost patterns */
type CostMetrics struct {
	TotalCost   float64
	CostByAgent map[uuid.UUID]float64
	CostByModel map[string]float64
	CostByType  map[string]float64
	DailyCosts  map[string]float64
	mu          sync.RWMutex
}

/* QualityMetrics tracks quality scores */
type QualityMetrics struct {
	TotalScores  int64
	AvgScore     float64
	ScoreHistory []float64
	ScoreByAgent map[uuid.UUID][]float64
	mu           sync.RWMutex
}

/* NewAdvancedMetrics creates a new advanced metrics collector */
func NewAdvancedMetrics() *AdvancedMetrics {
	return &AdvancedMetrics{
		agentMetrics: make(map[string]*AgentMetrics),
		toolMetrics:  make(map[string]*ToolMetrics),
		costMetrics: &CostMetrics{
			CostByAgent: make(map[uuid.UUID]float64),
			CostByModel: make(map[string]float64),
			CostByType:  make(map[string]float64),
			DailyCosts:  make(map[string]float64),
		},
		qualityMetrics: &QualityMetrics{
			ScoreHistory: make([]float64, 0),
			ScoreByAgent: make(map[uuid.UUID][]float64),
		},
	}
}

/* RecordAgentExecution records agent execution metrics */
func (am *AdvancedMetrics) RecordAgentExecution(agentID uuid.UUID, success bool, duration time.Duration, tokens int, cost float64) {
	am.mu.Lock()
	defer am.mu.Unlock()

	agentIDStr := agentID.String()
	metrics, exists := am.agentMetrics[agentIDStr]
	if !exists {
		metrics = &AgentMetrics{
			AgentID:        agentID,
			ExecutionTimes: make([]time.Duration, 0),
		}
		am.agentMetrics[agentIDStr] = metrics
	}

	metrics.mu.Lock()
	defer metrics.mu.Unlock()

	metrics.TotalExecutions++
	if success {
		metrics.SuccessfulExecutions++
	} else {
		metrics.FailedExecutions++
	}
	metrics.TotalTokens += int64(tokens)
	metrics.TotalCost += cost
	metrics.LastExecutionTime = time.Now()
	metrics.ExecutionTimes = append(metrics.ExecutionTimes, duration)

	/* Keep only last 100 execution times */
	if len(metrics.ExecutionTimes) > 100 {
		metrics.ExecutionTimes = metrics.ExecutionTimes[len(metrics.ExecutionTimes)-100:]
	}

	/* Calculate averages */
	if metrics.TotalExecutions > 0 {
		var totalDuration time.Duration
		for _, d := range metrics.ExecutionTimes {
			totalDuration += d
		}
		metrics.AvgExecutionTime = totalDuration / time.Duration(len(metrics.ExecutionTimes))
		metrics.AvgTokensPerExecution = metrics.TotalTokens / metrics.TotalExecutions
	}
}

/* RecordToolExecution records tool execution metrics */
func (am *AdvancedMetrics) RecordToolExecution(toolName string, success bool, duration time.Duration, cost float64) {
	am.mu.Lock()
	defer am.mu.Unlock()

	metrics, exists := am.toolMetrics[toolName]
	if !exists {
		metrics = &ToolMetrics{
			ToolName:  toolName,
			CallTimes: make([]time.Duration, 0),
		}
		am.toolMetrics[toolName] = metrics
	}

	metrics.mu.Lock()
	defer metrics.mu.Unlock()

	metrics.TotalCalls++
	if success {
		metrics.SuccessfulCalls++
	} else {
		metrics.FailedCalls++
	}
	metrics.TotalCost += cost
	metrics.CallTimes = append(metrics.CallTimes, duration)

	/* Keep only last 100 call times */
	if len(metrics.CallTimes) > 100 {
		metrics.CallTimes = metrics.CallTimes[len(metrics.CallTimes)-100:]
	}

	/* Calculate average */
	if len(metrics.CallTimes) > 0 {
		var totalDuration time.Duration
		for _, d := range metrics.CallTimes {
			totalDuration += d
		}
		metrics.AvgExecutionTime = totalDuration / time.Duration(len(metrics.CallTimes))
	}
}

/* RecordCost records cost metrics */
func (am *AdvancedMetrics) RecordCost(agentID uuid.UUID, modelName, costType string, cost float64) {
	am.mu.Lock()
	defer am.mu.Unlock()

	am.costMetrics.mu.Lock()
	defer am.costMetrics.mu.Unlock()

	am.costMetrics.TotalCost += cost
	am.costMetrics.CostByAgent[agentID] += cost
	am.costMetrics.CostByModel[modelName] += cost
	am.costMetrics.CostByType[costType] += cost

	/* Record daily cost */
	today := time.Now().Format("2006-01-02")
	am.costMetrics.DailyCosts[today] += cost
}

/* RecordQualityScore records quality score */
func (am *AdvancedMetrics) RecordQualityScore(agentID uuid.UUID, score float64) {
	am.mu.Lock()
	defer am.mu.Unlock()

	am.qualityMetrics.mu.Lock()
	defer am.qualityMetrics.mu.Unlock()

	am.qualityMetrics.TotalScores++
	am.qualityMetrics.ScoreHistory = append(am.qualityMetrics.ScoreHistory, score)
	am.qualityMetrics.ScoreByAgent[agentID] = append(am.qualityMetrics.ScoreByAgent[agentID], score)

	/* Keep only last 1000 scores */
	if len(am.qualityMetrics.ScoreHistory) > 1000 {
		am.qualityMetrics.ScoreHistory = am.qualityMetrics.ScoreHistory[len(am.qualityMetrics.ScoreHistory)-1000:]
	}

	/* Calculate average */
	var total float64
	for _, s := range am.qualityMetrics.ScoreHistory {
		total += s
	}
	if len(am.qualityMetrics.ScoreHistory) > 0 {
		am.qualityMetrics.AvgScore = total / float64(len(am.qualityMetrics.ScoreHistory))
	}
}

/* GetAgentMetrics gets metrics for an agent */
func (am *AdvancedMetrics) GetAgentMetrics(agentID uuid.UUID) *AgentMetrics {
	am.mu.RLock()
	defer am.mu.RUnlock()

	metrics, exists := am.agentMetrics[agentID.String()]
	if !exists {
		return nil
	}

	/* Return a copy */
	metrics.mu.RLock()
	defer metrics.mu.RUnlock()

	return &AgentMetrics{
		AgentID:               metrics.AgentID,
		TotalExecutions:       metrics.TotalExecutions,
		SuccessfulExecutions:  metrics.SuccessfulExecutions,
		FailedExecutions:      metrics.FailedExecutions,
		TotalTokens:           metrics.TotalTokens,
		TotalCost:             metrics.TotalCost,
		AvgExecutionTime:      metrics.AvgExecutionTime,
		AvgTokensPerExecution: metrics.AvgTokensPerExecution,
		LastExecutionTime:     metrics.LastExecutionTime,
	}
}

/* GetToolMetrics gets metrics for a tool */
func (am *AdvancedMetrics) GetToolMetrics(toolName string) *ToolMetrics {
	am.mu.RLock()
	defer am.mu.RUnlock()

	metrics, exists := am.toolMetrics[toolName]
	if !exists {
		return nil
	}

	/* Return a copy */
	metrics.mu.RLock()
	defer metrics.mu.RUnlock()

	return &ToolMetrics{
		ToolName:         metrics.ToolName,
		TotalCalls:       metrics.TotalCalls,
		SuccessfulCalls:  metrics.SuccessfulCalls,
		FailedCalls:      metrics.FailedCalls,
		AvgExecutionTime: metrics.AvgExecutionTime,
		TotalCost:        metrics.TotalCost,
	}
}

/* GetCostMetrics gets cost metrics */
func (am *AdvancedMetrics) GetCostMetrics() *CostMetrics {
	am.mu.RLock()
	defer am.mu.RUnlock()

	am.costMetrics.mu.RLock()
	defer am.costMetrics.mu.RUnlock()

	/* Return a copy */
	costByAgent := make(map[uuid.UUID]float64)
	for k, v := range am.costMetrics.CostByAgent {
		costByAgent[k] = v
	}
	costByModel := make(map[string]float64)
	for k, v := range am.costMetrics.CostByModel {
		costByModel[k] = v
	}
	costByType := make(map[string]float64)
	for k, v := range am.costMetrics.CostByType {
		costByType[k] = v
	}
	dailyCosts := make(map[string]float64)
	for k, v := range am.costMetrics.DailyCosts {
		dailyCosts[k] = v
	}

	return &CostMetrics{
		TotalCost:   am.costMetrics.TotalCost,
		CostByAgent: costByAgent,
		CostByModel: costByModel,
		CostByType:  costByType,
		DailyCosts:  dailyCosts,
	}
}

/* GetQualityMetrics gets quality metrics */
func (am *AdvancedMetrics) GetQualityMetrics(agentID *uuid.UUID) *QualityMetrics {
	am.mu.RLock()
	defer am.mu.RUnlock()

	am.qualityMetrics.mu.RLock()
	defer am.qualityMetrics.mu.RUnlock()

	/* Return a copy */
	var scoreHistory []float64
	var scoreByAgent map[uuid.UUID][]float64

	if agentID != nil {
		scores, exists := am.qualityMetrics.ScoreByAgent[*agentID]
		if exists {
			scoreHistory = make([]float64, len(scores))
			copy(scoreHistory, scores)
		}
	} else {
		scoreHistory = make([]float64, len(am.qualityMetrics.ScoreHistory))
		copy(scoreHistory, am.qualityMetrics.ScoreHistory)
		scoreByAgent = make(map[uuid.UUID][]float64)
		for k, v := range am.qualityMetrics.ScoreByAgent {
			scores := make([]float64, len(v))
			copy(scores, v)
			scoreByAgent[k] = scores
		}
	}

	return &QualityMetrics{
		TotalScores:  am.qualityMetrics.TotalScores,
		AvgScore:     am.qualityMetrics.AvgScore,
		ScoreHistory: scoreHistory,
		ScoreByAgent: scoreByAgent,
	}
}

/* Global advanced metrics instance */
var globalAdvancedMetrics *AdvancedMetrics
var globalAdvancedMetricsOnce sync.Once

/* GetAdvancedMetrics returns the global advanced metrics instance */
func GetAdvancedMetrics() *AdvancedMetrics {
	globalAdvancedMetricsOnce.Do(func() {
		globalAdvancedMetrics = NewAdvancedMetrics()
	})
	return globalAdvancedMetrics
}
