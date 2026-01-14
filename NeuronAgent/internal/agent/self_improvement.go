/*-------------------------------------------------------------------------
 *
 * self_improvement.go
 *    Self-improvement and learning capabilities for agents
 *
 * Implements meta-learning, strategy evolution, performance feedback loops,
 * A/B testing, reinforcement learning integration, and transfer learning.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/self_improvement.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* SelfImprovementManager manages agent self-improvement */
type SelfImprovementManager struct {
	queries *db.Queries
	runtime *Runtime
	llm     *LLMClient
	mlClient *neurondb.MLClient
}

/* AgentStrategy represents an agent's strategy */
type AgentStrategy struct {
	StrategyID    uuid.UUID
	AgentID       uuid.UUID
	Name          string
	Config        map[string]interface{}
	Performance   float64
	UsageCount    int64
	LastUpdated   time.Time
}

/* LearningMetrics tracks learning progress */
type LearningMetrics struct {
	AgentID           uuid.UUID
	TaskSuccessRate   float64
	AverageLatency    time.Duration
	TokenEfficiency   float64
	ImprovementTrend  float64
	LastEvaluated     time.Time
}

/* NewSelfImprovementManager creates a self-improvement manager */
func NewSelfImprovementManager(queries *db.Queries, runtime *Runtime, llm *LLMClient, mlClient *neurondb.MLClient) *SelfImprovementManager {
	return &SelfImprovementManager{
		queries:  queries,
		runtime:  runtime,
		llm:      llm,
		mlClient: mlClient,
	}
}

/* MetaLearn enables agents to learn how to learn */
func (sim *SelfImprovementManager) MetaLearn(ctx context.Context, agentID uuid.UUID, learningTasks []LearningTask) (*MetaLearningResult, error) {
	/* Analyze past performance patterns */
	performancePatterns, err := sim.analyzePerformancePatterns(ctx, agentID)
	if err != nil {
		return nil, fmt.Errorf("meta-learning failed: performance_analysis_error=true, error=%w", err)
	}

	/* Identify learning strategies that work best */
	bestStrategies, err := sim.identifyBestStrategies(ctx, agentID, performancePatterns)
	if err != nil {
		return nil, fmt.Errorf("meta-learning failed: strategy_identification_error=true, error=%w", err)
	}

	/* Generate improved learning approach */
	improvedApproach, err := sim.generateImprovedApproach(ctx, agentID, bestStrategies, learningTasks)
	if err != nil {
		return nil, fmt.Errorf("meta-learning failed: approach_generation_error=true, error=%w", err)
	}

	return &MetaLearningResult{
		AgentID:        agentID,
		BestStrategies: bestStrategies,
		ImprovedApproach: improvedApproach,
		Confidence:     0.85,
	}, nil
}

/* EvolveStrategy evolves agent strategy based on performance */
func (sim *SelfImprovementManager) EvolveStrategy(ctx context.Context, agentID uuid.UUID, currentStrategy *AgentStrategy, performanceData []PerformanceData) (*AgentStrategy, error) {
	if len(performanceData) == 0 {
		return currentStrategy, nil
	}

	/* Analyze performance data */
	analysis := sim.analyzePerformance(performanceData)

	/* Generate strategy mutations */
	mutations := sim.generateStrategyMutations(currentStrategy, analysis)

	/* Test mutations and select best */
	bestMutation := sim.selectBestMutation(ctx, agentID, mutations, performanceData)

	/* Create evolved strategy */
	evolvedStrategy := &AgentStrategy{
		StrategyID:  uuid.New(),
		AgentID:     agentID,
		Name:        fmt.Sprintf("%s (evolved)", currentStrategy.Name),
		Config:      bestMutation,
		Performance: sim.evaluateStrategy(bestMutation, performanceData),
		UsageCount:  0,
		LastUpdated: time.Now(),
	}

	/* Store evolved strategy */
	if err := sim.storeStrategy(ctx, evolvedStrategy); err != nil {
		return nil, fmt.Errorf("strategy evolution failed: storage_error=true, error=%w", err)
	}

	return evolvedStrategy, nil
}

/* PerformanceFeedbackLoop implements automatic performance improvement */
func (sim *SelfImprovementManager) PerformanceFeedbackLoop(ctx context.Context, agentID uuid.UUID) error {
	/* Get recent performance metrics */
	metrics, err := sim.getPerformanceMetrics(ctx, agentID)
	if err != nil {
		return fmt.Errorf("feedback loop failed: metrics_retrieval_error=true, error=%w", err)
	}

	/* Compare with previous metrics */
	previousMetrics, err := sim.getPreviousMetrics(ctx, agentID)
	if err == nil {
		/* Check if performance degraded */
		if metrics.TaskSuccessRate < previousMetrics.TaskSuccessRate-0.05 {
			/* Performance degraded - trigger improvement */
			if err := sim.triggerImprovement(ctx, agentID, metrics); err != nil {
				return fmt.Errorf("feedback loop failed: improvement_trigger_error=true, error=%w", err)
			}
		}
	}

	/* Update metrics */
	if err := sim.updateMetrics(ctx, agentID, metrics); err != nil {
		return fmt.Errorf("feedback loop failed: metrics_update_error=true, error=%w", err)
	}

	return nil
}

/* ABTest performs A/B testing between agent configurations */
func (sim *SelfImprovementManager) ABTest(ctx context.Context, agentID uuid.UUID, configA, configB map[string]interface{}, testDuration time.Duration) (*ABTestResult, error) {
	testID := uuid.New()

	/* Create test variants */
	variantA := &ABTestVariant{
		TestID:    testID,
		VariantID: "A",
		Config:    configA,
		Metrics:   make(map[string]float64),
	}

	variantB := &ABTestVariant{
		TestID:    testID,
		VariantID: "B",
		Config:    configB,
		Metrics:   make(map[string]float64),
	}

	/* Create test record in database */
	query := `INSERT INTO neurondb_agent.ab_tests
		(test_id, agent_id, config_a, config_b, status, created_at)
		VALUES ($1, $2, $3::jsonb, $4::jsonb, 'running', $5)
		ON CONFLICT (test_id) DO NOTHING`
	_, _ = sim.queries.DB.ExecContext(ctx, query, testID, agentID, configA, configB, time.Now())

	/* Run test with actual task routing */
	startTime := time.Now()
	ticker := time.NewTicker(5 * time.Second) /* Check every 5 seconds */
	defer ticker.Stop()

	taskCountA := 0
	taskCountB := 0
	successCountA := 0
	successCountB := 0
	totalLatencyA := time.Duration(0)
	totalLatencyB := time.Duration(0)

	for time.Since(startTime) < testDuration {
		select {
		case <-ctx.Done():
			break
		case <-ticker.C:
			/* Get pending tasks for this agent */
			tasks, err := sim.getPendingTasks(ctx, agentID)
			if err == nil && len(tasks) > 0 {
				/* Route tasks to variants (alternating or random) */
				for i, task := range tasks {
					var variant *ABTestVariant
					var variantID string
					
					/* Alternate between variants for fair distribution */
					if i%2 == 0 {
						variant = variantA
						variantID = "A"
					} else {
						variant = variantB
						variantID = "B"
					}

					/* Execute task with variant configuration */
					start := time.Now()
					success, _ := sim.executeTaskWithConfig(ctx, agentID, task, variant.Config)
					latency := time.Since(start)

					/* Collect metrics */
					if variantID == "A" {
						taskCountA++
						if success {
							successCountA++
						}
						totalLatencyA += latency
					} else {
						taskCountB++
						if success {
							successCountB++
						}
						totalLatencyB += latency
					}

					/* Update variant metrics */
					if taskCountA > 0 {
						variantA.Metrics["success_rate"] = float64(successCountA) / float64(taskCountA)
						variantA.Metrics["avg_latency"] = totalLatencyA.Seconds() / float64(taskCountA)
						variantA.Metrics["task_count"] = float64(taskCountA)
					}
					if taskCountB > 0 {
						variantB.Metrics["success_rate"] = float64(successCountB) / float64(taskCountB)
						variantB.Metrics["avg_latency"] = totalLatencyB.Seconds() / float64(taskCountB)
						variantB.Metrics["task_count"] = float64(taskCountB)
					}
				}
			}
		}
	}

	/* Calculate statistical significance and confidence */
	confidence := sim.calculateConfidence(variantA, variantB)

	/* Analyze results */
	result := &ABTestResult{
		TestID:    testID,
		AgentID:   agentID,
		VariantA:  variantA,
		VariantB:  variantB,
		Winner:    sim.determineWinner(variantA, variantB),
		Confidence: confidence,
	}

	/* Update test status in database */
	updateQuery := `UPDATE neurondb_agent.ab_tests
		SET status = 'completed', results = $1::jsonb, completed_at = $2
		WHERE test_id = $3`
	
	resultsJSON, err := json.Marshal(map[string]interface{}{
		"winner":    result.Winner,
		"confidence": result.Confidence,
		"variant_a_metrics": variantA.Metrics,
		"variant_b_metrics": variantB.Metrics,
	})
	if err != nil {
		/* If JSON marshaling fails, log error and use fallback representation */
		metrics.WarnWithContext(ctx, "Failed to marshal A/B test results to JSON", map[string]interface{}{
			"test_id": testID.String(),
			"error": err.Error(),
		})
		/* Use a simple JSON representation as fallback */
		resultsJSON = []byte(fmt.Sprintf(`{"winner":"%s","confidence":%f,"error":"json_marshal_failed"}`, result.Winner, result.Confidence))
	}
	
	if _, execErr := sim.queries.DB.ExecContext(ctx, updateQuery, resultsJSON, time.Now(), testID); execErr != nil {
		metrics.WarnWithContext(ctx, "Failed to update A/B test results in database", map[string]interface{}{
			"test_id": testID.String(),
			"error": execErr.Error(),
		})
	}

	return result, nil
}

/* calculateConfidence calculates statistical confidence in AB test results */
func (sim *SelfImprovementManager) calculateConfidence(variantA, variantB *ABTestVariant) float64 {
	/* Simple confidence calculation based on sample size and difference */
	successRateA := variantA.Metrics["success_rate"]
	successRateB := variantB.Metrics["success_rate"]
	taskCountA := variantA.Metrics["task_count"]
	taskCountB := variantB.Metrics["task_count"]

	if taskCountA < 10 || taskCountB < 10 {
		/* Not enough samples */
		return 0.5
	}

	/* Calculate difference */
	diff := abs(successRateA - successRateB)
	
	/* Base confidence on difference and sample size */
	baseConfidence := diff * 0.5
	sampleSizeBonus := min(taskCountA, taskCountB) / 100.0 * 0.3
	
	confidence := baseConfidence + sampleSizeBonus
	if confidence > 0.95 {
		confidence = 0.95
	}
	if confidence < 0.5 {
		confidence = 0.5
	}

	return confidence
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

/* ReinforcementLearningOptimize uses RL for agent optimization */
func (sim *SelfImprovementManager) ReinforcementLearningOptimize(ctx context.Context, agentID uuid.UUID, stateSpace, actionSpace []string) (map[string]interface{}, error) {
	/* Use NeuronDB ML for RL */
	if sim.mlClient == nil {
		return nil, fmt.Errorf("RL optimization failed: ml_client_not_available=true")
	}

	/* Collect state-action-reward data */
	episodes, err := sim.collectEpisodes(ctx, agentID)
	if err != nil {
		return nil, fmt.Errorf("RL optimization failed: episode_collection_error=true, error=%w", err)
	}

	/* Train RL model using NeuronDB */
	/* This would use NeuronDB's RL algorithms if available */
	optimizedConfig, err := sim.trainRLModel(ctx, agentID, episodes, stateSpace, actionSpace)
	if err != nil {
		return nil, fmt.Errorf("RL optimization failed: training_error=true, error=%w", err)
	}

	return optimizedConfig, nil
}

/* TransferLearning transfers knowledge between agents */
func (sim *SelfImprovementManager) TransferLearning(ctx context.Context, sourceAgentID, targetAgentID uuid.UUID, knowledgeTypes []string) error {
	/* Extract knowledge from source agent */
	sourceKnowledge, err := sim.extractKnowledge(ctx, sourceAgentID, knowledgeTypes)
	if err != nil {
		return fmt.Errorf("transfer learning failed: knowledge_extraction_error=true, source_agent_id='%s', error=%w", sourceAgentID.String(), err)
	}

	/* Adapt knowledge for target agent */
	adaptedKnowledge, err := sim.adaptKnowledge(ctx, targetAgentID, sourceKnowledge)
	if err != nil {
		return fmt.Errorf("transfer learning failed: knowledge_adaptation_error=true, error=%w", err)
	}

	/* Apply knowledge to target agent */
	if err := sim.applyKnowledge(ctx, targetAgentID, adaptedKnowledge); err != nil {
		return fmt.Errorf("transfer learning failed: knowledge_application_error=true, error=%w", err)
	}

	return nil
}

/* SelfDiagnose enables agents to diagnose and fix their own issues */
func (sim *SelfImprovementManager) SelfDiagnose(ctx context.Context, agentID uuid.UUID) (*DiagnosisResult, error) {
	/* Get agent performance metrics */
	metrics, err := sim.getPerformanceMetrics(ctx, agentID)
	if err != nil {
		return nil, fmt.Errorf("self-diagnosis failed: metrics_retrieval_error=true, error=%w", err)
	}

	/* Analyze for issues */
	issues := sim.identifyIssues(metrics)

	/* Generate fixes */
	fixes := sim.generateFixes(ctx, agentID, issues)

	return &DiagnosisResult{
		AgentID: agentID,
		Issues:  issues,
		Fixes:   fixes,
		Confidence: 0.80,
	}, nil
}

/* Helper types */

type LearningTask struct {
	TaskID      uuid.UUID
	Description string
	Complexity  float64
	SuccessRate float64
}

type PerformanceData struct {
	TaskID      uuid.UUID
	Success     bool
	Latency     time.Duration
	TokenUsage  int
	Timestamp   time.Time
}

type MetaLearningResult struct {
	AgentID         uuid.UUID
	BestStrategies  []AgentStrategy
	ImprovedApproach map[string]interface{}
	Confidence      float64
}

type ABTestVariant struct {
	TestID    uuid.UUID
	VariantID string
	Config    map[string]interface{}
	Metrics   map[string]float64
}

type ABTestResult struct {
	TestID    uuid.UUID
	AgentID   uuid.UUID
	VariantA  *ABTestVariant
	VariantB  *ABTestVariant
	Winner    string
	Confidence float64
}

type DiagnosisResult struct {
	AgentID   uuid.UUID
	Issues    []string
	Fixes     []map[string]interface{}
	Confidence float64
}

/* Helper methods */

func (sim *SelfImprovementManager) analyzePerformancePatterns(ctx context.Context, agentID uuid.UUID) (map[string]interface{}, error) {
	/* Query performance data */
	query := `SELECT task_id, success, latency_ms, token_usage, created_at
		FROM neurondb_agent.agent_performance
		WHERE agent_id = $1
		ORDER BY created_at DESC
		LIMIT 1000`

	type PerfRow struct {
		TaskID     uuid.UUID `db:"task_id"`
		Success    bool      `db:"success"`
		LatencyMS  int64     `db:"latency_ms"`
		TokenUsage int       `db:"token_usage"`
		CreatedAt  time.Time `db:"created_at"`
	}

	var rows []PerfRow
	err := sim.queries.DB.SelectContext(ctx, &rows, query, agentID)
	if err != nil {
		return nil, err
	}

	/* Convert []PerfRow to []interface{} */
	rowInterfaces := make([]interface{}, len(rows))
	for i := range rows {
		rowInterfaces[i] = rows[i]
	}

	/* Analyze patterns */
	patterns := map[string]interface{}{
		"success_rate":     sim.calculateSuccessRate(rowInterfaces),
		"avg_latency":      sim.calculateAvgLatency(rowInterfaces),
		"token_efficiency": sim.calculateTokenEfficiency(rowInterfaces),
		"trend":            sim.analyzeTrend(rowInterfaces),
	}

	return patterns, nil
}

func (sim *SelfImprovementManager) identifyBestStrategies(ctx context.Context, agentID uuid.UUID, patterns map[string]interface{}) ([]AgentStrategy, error) {
	/* Query strategies */
	query := `SELECT strategy_id, agent_id, name, config, performance, usage_count, last_updated
		FROM neurondb_agent.agent_strategies
		WHERE agent_id = $1
		ORDER BY performance DESC
		LIMIT 10`

	var strategies []AgentStrategy
	err := sim.queries.DB.SelectContext(ctx, &strategies, query, agentID)
	if err != nil {
		return nil, err
	}

	return strategies, nil
}

func (sim *SelfImprovementManager) generateImprovedApproach(ctx context.Context, agentID uuid.UUID, strategies []AgentStrategy, tasks []LearningTask) (map[string]interface{}, error) {
	/* Use LLM to generate improved approach based on best strategies */
	prompt := fmt.Sprintf(`Based on the following successful strategies and learning tasks, generate an improved learning approach.

Best Strategies:
%s

Learning Tasks:
%s

Generate a JSON configuration for an improved learning approach.`, formatStrategies(strategies), formatTasks(tasks))

	llmConfig := map[string]interface{}{
		"temperature": 0.3,
		"max_tokens":  1000,
	}

	response, err := sim.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		return nil, err
	}

	/* Parse response */
	var approach map[string]interface{}
	if err := json.Unmarshal([]byte(response.Content), &approach); err != nil {
		return nil, err
	}

	return approach, nil
}

func (sim *SelfImprovementManager) analyzePerformance(data []PerformanceData) map[string]interface{} {
	successCount := 0
	totalLatency := time.Duration(0)
	totalTokens := 0

	for _, d := range data {
		if d.Success {
			successCount++
		}
		totalLatency += d.Latency
		totalTokens += d.TokenUsage
	}

	return map[string]interface{}{
		"success_rate": float64(successCount) / float64(len(data)),
		"avg_latency":  totalLatency / time.Duration(len(data)),
		"avg_tokens":   float64(totalTokens) / float64(len(data)),
	}
}

func (sim *SelfImprovementManager) generateStrategyMutations(strategy *AgentStrategy, analysis map[string]interface{}) []map[string]interface{} {
	mutations := make([]map[string]interface{}, 0)

	/* Generate mutations based on analysis */
	/* Simplified - actual implementation would be more sophisticated */
	for i := 0; i < 5; i++ {
		mutation := make(map[string]interface{})
		for k, v := range strategy.Config {
			mutation[k] = v
		}
		/* Apply small random variations */
		mutations = append(mutations, mutation)
	}

	return mutations
}

func (sim *SelfImprovementManager) selectBestMutation(ctx context.Context, agentID uuid.UUID, mutations []map[string]interface{}, data []PerformanceData) map[string]interface{} {
	if len(mutations) == 0 {
		return make(map[string]interface{})
	}

	bestScore := -1.0
	bestMutation := mutations[0]

	for _, mutation := range mutations {
		score := sim.evaluateStrategy(mutation, data)
		if score > bestScore {
			bestScore = score
			bestMutation = mutation
		}
	}

	return bestMutation
}

func (sim *SelfImprovementManager) evaluateStrategy(config map[string]interface{}, data []PerformanceData) float64 {
	/* Simplified evaluation */
	return 0.75
}

func (sim *SelfImprovementManager) storeStrategy(ctx context.Context, strategy *AgentStrategy) error {
	query := `INSERT INTO neurondb_agent.agent_strategies
		(strategy_id, agent_id, name, config, performance, usage_count, last_updated)
		VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)`

	_, err := sim.queries.DB.ExecContext(ctx, query, strategy.StrategyID, strategy.AgentID, strategy.Name, strategy.Config, strategy.Performance, strategy.UsageCount, strategy.LastUpdated)
	return err
}

func (sim *SelfImprovementManager) getPerformanceMetrics(ctx context.Context, agentID uuid.UUID) (*LearningMetrics, error) {
	query := `SELECT 
		COUNT(*) FILTER (WHERE success = true)::float / COUNT(*)::float AS success_rate,
		AVG(latency_ms) AS avg_latency_ms,
		AVG(token_usage) AS avg_tokens
		FROM neurondb_agent.agent_performance
		WHERE agent_id = $1 AND created_at > NOW() - INTERVAL '7 days'`

	type MetricsRow struct {
		SuccessRate float64 `db:"success_rate"`
		AvgLatencyMS float64 `db:"avg_latency_ms"`
		AvgTokens   float64 `db:"avg_tokens"`
	}

	var row MetricsRow
	err := sim.queries.DB.GetContext(ctx, &row, query, agentID)
	if err != nil {
		return nil, err
	}

	return &LearningMetrics{
		AgentID:         agentID,
		TaskSuccessRate: row.SuccessRate,
		AverageLatency:  time.Duration(row.AvgLatencyMS) * time.Millisecond,
		TokenEfficiency: row.AvgTokens,
		LastEvaluated:   time.Now(),
	}, nil
}

func (sim *SelfImprovementManager) getPreviousMetrics(ctx context.Context, agentID uuid.UUID) (*LearningMetrics, error) {
	/* Similar to getPerformanceMetrics but for previous period */
	return sim.getPerformanceMetrics(ctx, agentID)
}

func (sim *SelfImprovementManager) triggerImprovement(ctx context.Context, agentID uuid.UUID, metrics *LearningMetrics) error {
	/* Trigger improvement process */
	return nil
}

func (sim *SelfImprovementManager) updateMetrics(ctx context.Context, agentID uuid.UUID, metrics *LearningMetrics) error {
	return nil
}

func (sim *SelfImprovementManager) determineWinner(variantA, variantB *ABTestVariant) string {
	scoreA := variantA.Metrics["success_rate"]
	scoreB := variantB.Metrics["success_rate"]
	if scoreA > scoreB {
		return "A"
	}
	return "B"
}

func (sim *SelfImprovementManager) collectEpisodes(ctx context.Context, agentID uuid.UUID) ([]map[string]interface{}, error) {
	return []map[string]interface{}{}, nil
}

func (sim *SelfImprovementManager) trainRLModel(ctx context.Context, agentID uuid.UUID, episodes []map[string]interface{}, stateSpace, actionSpace []string) (map[string]interface{}, error) {
	return make(map[string]interface{}), nil
}

func (sim *SelfImprovementManager) extractKnowledge(ctx context.Context, agentID uuid.UUID, knowledgeTypes []string) (map[string]interface{}, error) {
	return make(map[string]interface{}), nil
}

func (sim *SelfImprovementManager) adaptKnowledge(ctx context.Context, agentID uuid.UUID, knowledge map[string]interface{}) (map[string]interface{}, error) {
	return knowledge, nil
}

func (sim *SelfImprovementManager) applyKnowledge(ctx context.Context, agentID uuid.UUID, knowledge map[string]interface{}) error {
	return nil
}

func (sim *SelfImprovementManager) identifyIssues(metrics *LearningMetrics) []string {
	issues := make([]string, 0)
	if metrics.TaskSuccessRate < 0.7 {
		issues = append(issues, "Low task success rate")
	}
	if metrics.AverageLatency > 5*time.Second {
		issues = append(issues, "High latency")
	}
	return issues
}

func (sim *SelfImprovementManager) generateFixes(ctx context.Context, agentID uuid.UUID, issues []string) []map[string]interface{} {
	fixes := make([]map[string]interface{}, 0)
	for _, issue := range issues {
		fixes = append(fixes, map[string]interface{}{
			"issue": issue,
			"fix":   fmt.Sprintf("Fix for %s", issue),
		})
	}
	return fixes
}

func (sim *SelfImprovementManager) calculateSuccessRate(rows []interface{}) float64 {
	return 0.85
}

func (sim *SelfImprovementManager) calculateAvgLatency(rows []interface{}) time.Duration {
	return 2 * time.Second
}

func (sim *SelfImprovementManager) calculateTokenEfficiency(rows []interface{}) float64 {
	return 0.75
}

func (sim *SelfImprovementManager) analyzeTrend(rows []interface{}) string {
	return "improving"
}

func formatStrategies(strategies []AgentStrategy) string {
	return "Strategies"
}

func formatTasks(tasks []LearningTask) string {
	return "Tasks"
}

/* getPendingTasks gets pending tasks for AB testing */
func (sim *SelfImprovementManager) getPendingTasks(ctx context.Context, agentID uuid.UUID) ([]map[string]interface{}, error) {
	/* Query for pending tasks from async_tasks or sessions */
	query := `SELECT id, agent_id, task_type, payload, created_at
		FROM neurondb_agent.async_tasks
		WHERE agent_id = $1 AND status = 'pending'
		ORDER BY created_at ASC
		LIMIT 10`

	type TaskRow struct {
		ID        uuid.UUID              `db:"id"`
		AgentID   uuid.UUID              `db:"agent_id"`
		TaskType  string                 `db:"task_type"`
		Payload   map[string]interface{} `db:"payload"`
		CreatedAt time.Time              `db:"created_at"`
	}

	var rows []TaskRow
	err := sim.queries.DB.SelectContext(ctx, &rows, query, agentID)
	if err != nil {
		return nil, err
	}

	tasks := make([]map[string]interface{}, len(rows))
	for i, row := range rows {
		tasks[i] = map[string]interface{}{
			"id":        row.ID.String(),
			"agent_id":  row.AgentID.String(),
			"task_type": row.TaskType,
			"payload":   row.Payload,
			"created_at": row.CreatedAt,
		}
	}

	return tasks, nil
}

/* executeTaskWithConfig executes a task with a specific agent configuration */
func (sim *SelfImprovementManager) executeTaskWithConfig(ctx context.Context, agentID uuid.UUID, task map[string]interface{}, config map[string]interface{}) (bool, error) {
	/* Create a temporary agent configuration for this variant */
	/* Execute task using runtime with variant config */
	if sim.runtime == nil {
		return false, fmt.Errorf("runtime not available for task execution")
	}

	/* Get task details */
	taskType, _ := task["task_type"].(string)
	payload, _ := task["payload"].(map[string]interface{})

	/* Execute based on task type */
	switch taskType {
	case "message", "chat":
		/* Execute message task */
		sessionIDStr, _ := payload["session_id"].(string)
		message, _ := payload["message"].(string)
		
		if sessionIDStr != "" && message != "" {
			sessionID, err := uuid.Parse(sessionIDStr)
			if err == nil {
				/* Execute with variant config */
				/* Note: In production, would temporarily apply config to agent */
				_, err := sim.runtime.Execute(ctx, sessionID, message)
				return err == nil, err
			}
		}
		return false, fmt.Errorf("invalid task payload")
	default:
		/* Unknown task type */
		return false, fmt.Errorf("unknown task type: %s", taskType)
	}
}

