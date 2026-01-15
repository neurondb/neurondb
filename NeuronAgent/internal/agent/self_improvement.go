/*-------------------------------------------------------------------------
 *
 * self_improvement.go
 *    Self-improvement and meta-learning system
 *
 * Provides meta-learning, strategy evolution, performance feedback loops,
 * A/B testing, and reinforcement learning integration.
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
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* ExecutionRow represents a row from execution history */
type ExecutionRow struct {
	ID            uuid.UUID     `db:"id"`
	SessionID     uuid.UUID     `db:"session_id"`
	UserMessage   string        `db:"user_message"`
	FinalAnswer   string        `db:"final_answer"`
	Success       bool          `db:"success"`
	QualityScore  float64       `db:"quality_score"`
	TokensUsed    int           `db:"tokens_used"`
	ExecutionTime time.Duration `db:"execution_time"`
}

/* SelfImprovementManager manages agent self-improvement */
type SelfImprovementManager struct {
	queries     *db.Queries
	llm         *LLMClient
	mlClient    *neurondb.MLClient
	feedbackLoop *FeedbackLoop
}

/* NewSelfImprovementManager creates a new self-improvement manager */
func NewSelfImprovementManager(queries *db.Queries, llm *LLMClient, mlClient *neurondb.MLClient) *SelfImprovementManager {
	return &SelfImprovementManager{
		queries:      queries,
		llm:          llm,
		mlClient:     mlClient,
		feedbackLoop: NewFeedbackLoop(queries),
	}
}

/* LearnFromExperience learns from past experiences */
func (sim *SelfImprovementManager) LearnFromExperience(ctx context.Context, agentID uuid.UUID) error {
	/* Get recent execution results */
	query := `SELECT id, session_id, user_message, final_answer, success, quality_score, tokens_used, execution_time
		FROM neurondb_agent.execution_results
		WHERE agent_id = $1
		  AND created_at > NOW() - INTERVAL '7 days'
		ORDER BY created_at DESC
		LIMIT 100`

	var rows []ExecutionRow
	err := sim.queries.DB.SelectContext(ctx, &rows, query, agentID)
	if err != nil {
		return fmt.Errorf("self-improvement learning failed: query_error=true, error=%w", err)
	}

	/* Analyze patterns using ML */
	if sim.mlClient != nil && len(rows) >= 10 {
		/* Use NeuronDB ML functions to identify patterns */
		/* Create a temporary table with execution features */
		featuresTable := fmt.Sprintf("temp_execution_features_%s", agentID.String()[:8])
		
		/* Create table with features for ML analysis */
		createTableQuery := fmt.Sprintf(`
			CREATE TEMP TABLE %s (
				id SERIAL PRIMARY KEY,
				success BOOLEAN,
				quality_score FLOAT,
				tokens_used INT,
				execution_time_ms FLOAT,
				message_length INT,
				answer_length INT,
				features FLOAT[]
			)
		`, featuresTable)

		_, err := sim.queries.DB.ExecContext(ctx, createTableQuery)
		if err != nil {
			metrics.WarnWithContext(ctx, "Failed to create features table for ML analysis", map[string]interface{}{
				"agent_id": agentID.String(),
				"error":    err.Error(),
			})
			/* Fall back to simple analysis */
			successRate := sim.calculateSuccessRate(rows)
			avgQuality := sim.calculateAverageQuality(rows)
			sim.updateStrategy(ctx, agentID, successRate, avgQuality)
			return nil
		}

		/* Insert features into table */
		for _, row := range rows {
			features := []float32{
				float32(row.QualityScore),
				float32(row.TokensUsed),
				float32(row.ExecutionTime.Milliseconds()),
				float32(len(row.UserMessage)),
				float32(len(row.FinalAnswer)),
			}

			insertQuery := fmt.Sprintf(`
				INSERT INTO %s (success, quality_score, tokens_used, execution_time_ms, message_length, answer_length, features)
				VALUES ($1, $2, $3, $4, $5, $6, $7::float[])
			`, featuresTable)

			_, err := sim.queries.DB.ExecContext(ctx, insertQuery,
				row.Success,
				row.QualityScore,
				row.TokensUsed,
				float64(row.ExecutionTime.Milliseconds()),
				len(row.UserMessage),
				len(row.FinalAnswer),
				features,
			)
			if err != nil {
				continue
			}
		}

		/* Train a clustering model to identify patterns */
		projectName := fmt.Sprintf("agent_%s_patterns", agentID.String()[:8])
		modelID, err := sim.mlClient.TrainModel(ctx, projectName, "kmeans", featuresTable, "", []string{"features"}, map[string]interface{}{
			"n_clusters": 3, /* Cluster into successful, mixed, and failed patterns */
		})
		if err != nil {
			metrics.WarnWithContext(ctx, "ML pattern identification failed", map[string]interface{}{
				"agent_id": agentID.String(),
				"error":    err.Error(),
			})
			/* Fall back to simple analysis */
			successRate := sim.calculateSuccessRate(rows)
			avgQuality := sim.calculateAverageQuality(rows)
			sim.updateStrategy(ctx, agentID, successRate, avgQuality)
			return nil
		}

		/* Analyze clusters to identify successful patterns */
		/* For kmeans, we need to predict cluster assignments and analyze them */
		/* First, get predictions for all rows */
		clusterAnalysisQuery := fmt.Sprintf(`
			WITH predictions AS (
				SELECT 
					id,
					success,
					quality_score,
					tokens_used,
					execution_time_ms,
					ROUND(neurondb.predict($1, features)::numeric, 0)::int as cluster_id
				FROM %s
			)
			SELECT 
				cluster_id,
				AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
				AVG(quality_score) as avg_quality,
				AVG(tokens_used) as avg_tokens,
				AVG(execution_time_ms) as avg_execution_time,
				COUNT(*) as cluster_size
			FROM predictions
			GROUP BY cluster_id
			ORDER BY success_rate DESC
		`, featuresTable)

		type ClusterAnalysis struct {
			ClusterID        int     `db:"cluster_id"`
			SuccessRate      float64 `db:"success_rate"`
			AvgQuality       float64 `db:"avg_quality"`
			AvgTokens        float64 `db:"avg_tokens"`
			AvgExecutionTime float64 `db:"avg_execution_time"`
			ClusterSize      int     `db:"cluster_size"`
		}

		var clusters []ClusterAnalysis
		err = sim.queries.DB.SelectContext(ctx, &clusters, clusterAnalysisQuery, modelID)
		if err == nil && len(clusters) > 0 {
			/* Find the most successful cluster */
			bestCluster := clusters[0]
			for _, cluster := range clusters {
				if cluster.SuccessRate > bestCluster.SuccessRate {
					bestCluster = cluster
				}
			}

			/* Update strategy based on successful cluster patterns */
			successRate := bestCluster.SuccessRate
			avgQuality := bestCluster.AvgQuality

			metrics.InfoWithContext(ctx, "ML pattern identification completed", map[string]interface{}{
				"agent_id":        agentID.String(),
				"model_id":        modelID,
				"clusters_found":  len(clusters),
				"best_success_rate": successRate,
				"best_avg_quality":  avgQuality,
			})

			/* Update agent strategy based on learnings */
			if err := sim.updateStrategy(ctx, agentID, successRate, avgQuality); err != nil {
				metrics.WarnWithContext(ctx, "Failed to update strategy", map[string]interface{}{
					"agent_id": agentID.String(),
					"error":    err.Error(),
				})
			}
		} else {
			/* Fall back to simple analysis */
			successRate := sim.calculateSuccessRate(rows)
			avgQuality := sim.calculateAverageQuality(rows)
			sim.updateStrategy(ctx, agentID, successRate, avgQuality)
		}

		/* Clean up temp table */
		sim.queries.DB.ExecContext(ctx, fmt.Sprintf("DROP TABLE IF EXISTS %s", featuresTable))
	} else {
		/* Use simple analysis if ML client not available or insufficient data */
		successRate := sim.calculateSuccessRate(rows)
		avgQuality := sim.calculateAverageQuality(rows)

		/* Update agent strategy based on learnings */
		if err := sim.updateStrategy(ctx, agentID, successRate, avgQuality); err != nil {
			metrics.WarnWithContext(ctx, "Failed to update strategy", map[string]interface{}{
				"agent_id": agentID.String(),
				"error":    err.Error(),
			})
		}
	}

	return nil
}

/* calculateSuccessRate calculates success rate from executions */
func (sim *SelfImprovementManager) calculateSuccessRate(rows []ExecutionRow) float64 {
	if len(rows) == 0 {
		return 0.0
	}

	successCount := 0
	for _, row := range rows {
		if row.Success {
			successCount++
		}
	}

	return float64(successCount) / float64(len(rows))
}

/* calculateAverageQuality calculates average quality score */
func (sim *SelfImprovementManager) calculateAverageQuality(rows []ExecutionRow) float64 {
	if len(rows) == 0 {
		return 0.0
	}

	total := 0.0
	for _, row := range rows {
		total += row.QualityScore
	}

	return total / float64(len(rows))
}

/* updateStrategy updates agent strategy based on learnings */
func (sim *SelfImprovementManager) updateStrategy(ctx context.Context, agentID uuid.UUID, successRate, avgQuality float64) error {
	/* Get current agent config */
	agent, err := sim.queries.GetAgentByID(ctx, agentID)
	if err != nil {
		return err
	}

	/* Adjust strategy based on performance */
	config := agent.Config
	if config == nil {
		config = make(map[string]interface{})
	}

	/* Adjust temperature based on quality */
	if avgQuality < 0.7 {
		/* Lower quality - reduce temperature for more focused responses */
		if temp, ok := config["temperature"].(float64); ok && temp > 0.3 {
			config["temperature"] = temp - 0.1
		}
	} else if avgQuality > 0.9 {
		/* High quality - can increase temperature for more creativity */
		if temp, ok := config["temperature"].(float64); ok && temp < 0.9 {
			config["temperature"] = temp + 0.05
		}
	}

	/* Update agent config */
	updateQuery := `UPDATE neurondb_agent.agents
		SET config = $1::jsonb, updated_at = NOW()
		WHERE id = $2`

	_, err = sim.queries.DB.ExecContext(ctx, updateQuery, config, agentID)
	return err
}

/* FeedbackLoop manages performance feedback */
type FeedbackLoop struct {
	queries *db.Queries
}

/* NewFeedbackLoop creates a new feedback loop */
func NewFeedbackLoop(queries *db.Queries) *FeedbackLoop {
	return &FeedbackLoop{
		queries: queries,
	}
}

/* RecordFeedback records performance feedback */
func (fl *FeedbackLoop) RecordFeedback(ctx context.Context, agentID uuid.UUID, executionID uuid.UUID, success bool, qualityScore float64, feedback string) error {
	query := `INSERT INTO neurondb_agent.performance_feedback
		(id, agent_id, execution_id, success, quality_score, feedback, created_at)
		VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, NOW())`

	_, err := fl.queries.DB.ExecContext(ctx, query, agentID, executionID, success, qualityScore, feedback)
	return err
}

/* ABTestManager manages A/B testing for agent configurations */
type ABTestManager struct {
	queries *db.Queries
}

/* NewABTestManager creates a new A/B test manager */
func NewABTestManager(queries *db.Queries) *ABTestManager {
	return &ABTestManager{
		queries: queries,
	}
}

/* CreateABTest creates an A/B test */
func (abm *ABTestManager) CreateABTest(ctx context.Context, agentID uuid.UUID, variantA, variantB map[string]interface{}) (uuid.UUID, error) {
	testID := uuid.New()

	query := `INSERT INTO neurondb_agent.ab_tests
		(id, agent_id, variant_a, variant_b, status, created_at)
		VALUES ($1, $2, $3::jsonb, $4::jsonb, 'active', NOW())`

	_, err := abm.queries.DB.ExecContext(ctx, query, testID, agentID, variantA, variantB)
	return testID, err
}

/* RecordABTestResult records A/B test result */
func (abm *ABTestManager) RecordABTestResult(ctx context.Context, testID uuid.UUID, variant string, success bool, qualityScore float64) error {
	query := `INSERT INTO neurondb_agent.ab_test_results
		(id, test_id, variant, success, quality_score, created_at)
		VALUES (gen_random_uuid(), $1, $2, $3, $4, NOW())`

	_, err := abm.queries.DB.ExecContext(ctx, query, testID, variant, success, qualityScore)
	return err
}
