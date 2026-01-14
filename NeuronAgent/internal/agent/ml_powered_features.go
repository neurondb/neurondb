/*-------------------------------------------------------------------------
 *
 * ml_powered_features.go
 *    ML-powered features for agents using NeuronDB ML capabilities
 *
 * Implements ML-powered memory organization, anomaly detection, predictive planning,
 * recommendation systems, time series analysis, classification, regression, and
 * feature engineering.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/ml_powered_features.go
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
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* MLPoweredFeatures provides ML-powered capabilities for agents */
type MLPoweredFeatures struct {
	db         *db.DB
	queries    *db.Queries
	mlClient   *neurondb.MLClient
	embed      *neurondb.EmbeddingClient
	llm        *LLMClient
}

/* NewMLPoweredFeatures creates ML-powered features manager */
func NewMLPoweredFeatures(database *db.DB, queries *db.Queries, mlClient *neurondb.MLClient, embedClient *neurondb.EmbeddingClient, llmClient *LLMClient) *MLPoweredFeatures {
	return &MLPoweredFeatures{
		db:       database,
		queries:  queries,
		mlClient: mlClient,
		embed:    embedClient,
		llm:      llmClient,
	}
}

/* MLPoweredMemory organizes memory using clustering */
func (m *MLPoweredFeatures) MLPoweredMemory(ctx context.Context, agentID uuid.UUID, memoryEmbeddings [][]float32, numClusters int) ([]MemoryCluster, error) {
	if len(memoryEmbeddings) < numClusters {
		return nil, fmt.Errorf("ML-powered memory failed: not_enough_memories=true, count=%d, clusters=%d", len(memoryEmbeddings), numClusters)
	}

	/* Create temporary table for embeddings */
	tableName := fmt.Sprintf("temp_memory_embeddings_%s", agentID.String()[:8])
	createTable := fmt.Sprintf(`CREATE TEMP TABLE %s (
		id SERIAL PRIMARY KEY,
		embedding vector(%d)
	)`, tableName, len(memoryEmbeddings[0]))

	_, err := m.db.DB.ExecContext(ctx, createTable)
	if err != nil {
		return nil, fmt.Errorf("ML-powered memory failed: table_creation_error=true, error=%w", err)
	}
	defer m.db.DB.ExecContext(ctx, fmt.Sprintf("DROP TABLE IF EXISTS %s", tableName))

	/* Insert embeddings */
	for _, embedding := range memoryEmbeddings {
		embeddingJSON, _ := json.Marshal(embedding)
		insertQuery := fmt.Sprintf("INSERT INTO %s (embedding) VALUES ($1::vector)", tableName)
		_, err = m.db.DB.ExecContext(ctx, insertQuery, embeddingJSON)
		if err != nil {
			continue /* Skip on error */
		}
	}

	/* Train clustering model */
	params := map[string]interface{}{
		"n_clusters": numClusters,
		"algorithm":  "kmeans",
	}
	modelID, err := m.mlClient.TrainModel(ctx, "memory_clustering", "kmeans", tableName, "", []string{"embedding"}, params)
	if err != nil {
		return nil, fmt.Errorf("ML-powered memory failed: clustering_training_error=true, error=%w", err)
	}

	/* Predict clusters */
	query := fmt.Sprintf(`SELECT id, neurondb.predict($1, embedding) AS cluster_id
		FROM %s`, tableName)

	type ClusterRow struct {
		ID        int     `db:"id"`
		ClusterID float64 `db:"cluster_id"`
	}

	var rows []ClusterRow
	err = m.db.DB.SelectContext(ctx, &rows, query, modelID)
	if err != nil {
		return nil, fmt.Errorf("ML-powered memory failed: clustering_prediction_error=true, error=%w", err)
	}

	/* Group by cluster */
	clusters := make(map[int][]int)
	for _, row := range rows {
		clusterID := int(row.ClusterID)
		clusters[clusterID] = append(clusters[clusterID], row.ID)
	}

	var result []MemoryCluster
	for clusterID, memoryIDs := range clusters {
		result = append(result, MemoryCluster{
			ClusterID: clusterID,
			MemoryIDs: memoryIDs,
			Size:      len(memoryIDs),
		})
	}

	return result, nil
}

/* AnomalyDetection detects unusual patterns in agent behavior */
func (m *MLPoweredFeatures) AnomalyDetection(ctx context.Context, agentID uuid.UUID, behaviorData []BehaviorMetric) ([]Anomaly, error) {
	if len(behaviorData) < 10 {
		return nil, fmt.Errorf("anomaly detection failed: insufficient_data=true, count=%d", len(behaviorData))
	}

	/* Create temporary table */
	tableName := fmt.Sprintf("temp_behavior_%s", agentID.String()[:8])
	createTable := fmt.Sprintf(`CREATE TEMP TABLE %s (
		id SERIAL PRIMARY KEY,
		timestamp TIMESTAMP,
		metric_value DOUBLE PRECISION,
		features vector(10)
	)`, tableName)

	_, err := m.db.DB.ExecContext(ctx, createTable)
	if err != nil {
		return nil, fmt.Errorf("anomaly detection failed: table_creation_error=true, error=%w", err)
	}
	defer m.db.DB.ExecContext(ctx, fmt.Sprintf("DROP TABLE IF EXISTS %s", tableName))

	/* Insert behavior data */
	for _, metric := range behaviorData {
		features := m.extractFeatures(metric)
		featuresJSON, _ := json.Marshal(features)
		insertQuery := fmt.Sprintf("INSERT INTO %s (timestamp, metric_value, features) VALUES ($1, $2, $3::vector)", tableName)
		_, err = m.db.DB.ExecContext(ctx, insertQuery, metric.Timestamp, metric.Value, featuresJSON)
		if err != nil {
			continue
		}
	}

	/* Use outlier detection (z-score method via SQL) */
	query := fmt.Sprintf(`SELECT id, timestamp, metric_value,
		ABS(metric_value - AVG(metric_value) OVER()) / NULLIF(STDDEV(metric_value) OVER(), 0) AS z_score
		FROM %s
		WHERE ABS(metric_value - AVG(metric_value) OVER()) / NULLIF(STDDEV(metric_value) OVER(), 0) > 3`, tableName)

	type AnomalyRow struct {
		ID          int       `db:"id"`
		Timestamp   time.Time `db:"timestamp"`
		MetricValue float64   `db:"metric_value"`
		ZScore      float64   `db:"z_score"`
	}

	var rows []AnomalyRow
	err = m.db.DB.SelectContext(ctx, &rows, query)
	if err != nil {
		return nil, fmt.Errorf("anomaly detection failed: query_error=true, error=%w", err)
	}

	var anomalies []Anomaly
	for _, row := range rows {
		anomalies = append(anomalies, Anomaly{
			AgentID:    agentID,
			Timestamp:  row.Timestamp,
			Metric:     "behavior",
			Value:      row.MetricValue,
			ZScore:     row.ZScore,
			Severity:   m.calculateSeverity(row.ZScore),
		})
	}

	return anomalies, nil
}

/* PredictivePlanning predicts task success using ML models */
func (m *MLPoweredFeatures) PredictivePlanning(ctx context.Context, agentID uuid.UUID, task TaskFeatures) (float64, error) {
	/* Train or load model for task success prediction */
	modelID, err := m.getOrTrainSuccessModel(ctx, agentID)
	if err != nil {
		return 0.5, err /* Default probability */
	}

	/* Extract features */
	features := m.extractTaskFeatures(task)

	/* Predict success probability */
	probability, err := m.mlClient.Predict(ctx, modelID, features)
	if err != nil {
		return 0.5, fmt.Errorf("predictive planning failed: prediction_error=true, error=%w", err)
	}

	/* Normalize to 0-1 range */
	if probability < 0 {
		probability = 0
	}
	if probability > 1 {
		probability = 1
	}

	return probability, nil
}

/* RecommendationSystem suggests relevant tools and strategies */
func (m *MLPoweredFeatures) RecommendationSystem(ctx context.Context, agentID uuid.UUID, task string, availableTools []string) ([]ToolRecommendation, error) {
	/* Get historical tool usage for similar tasks */
	taskEmbedding, err := m.embed.Embed(ctx, task, "all-MiniLM-L6-v2")
	if err != nil {
		return nil, fmt.Errorf("recommendation system failed: embedding_error=true, error=%w", err)
	}

	/* Find similar past tasks and their successful tools */
	query := `SELECT tool_name, COUNT(*) as usage_count, AVG(success_rate) as avg_success
		FROM neurondb_agent.tool_usage_history
		WHERE agent_id = $1
		AND task_embedding <=> $2::vector < 0.3
		GROUP BY tool_name
		ORDER BY avg_success DESC, usage_count DESC
		LIMIT 5`

	type ToolUsageRow struct {
		ToolName     string  `db:"tool_name"`
		UsageCount   int     `db:"usage_count"`
		AvgSuccess   float64 `db:"avg_success"`
	}

	var rows []ToolUsageRow
	err = m.db.DB.SelectContext(ctx, &rows, query, agentID, taskEmbedding)
	if err != nil {
		/* If table doesn't exist, return default recommendations */
		return m.defaultToolRecommendations(availableTools), nil
	}

	var recommendations []ToolRecommendation
	for _, row := range rows {
		recommendations = append(recommendations, ToolRecommendation{
			ToolName:   row.ToolName,
			Confidence: row.AvgSuccess,
			Reason:     fmt.Sprintf("Used successfully in similar tasks (%d times)", row.UsageCount),
		})
	}

	return recommendations, nil
}

/* TimeSeriesAnalysis analyzes agent performance over time */
func (m *MLPoweredFeatures) TimeSeriesAnalysis(ctx context.Context, agentID uuid.UUID, metric string, startTime, endTime time.Time) (*TimeSeriesAnalysis, error) {
	query := `SELECT timestamp, value
		FROM neurondb_agent.performance_metrics
		WHERE agent_id = $1 AND metric_name = $2
		AND timestamp BETWEEN $3 AND $4
		ORDER BY timestamp`


	var rows []MetricRow
	err := m.db.DB.SelectContext(ctx, &rows, query, agentID, metric, startTime, endTime)
	if err != nil {
		return nil, fmt.Errorf("time series analysis failed: query_error=true, error=%w", err)
	}

	if len(rows) == 0 {
		return nil, fmt.Errorf("time series analysis failed: no_data=true")
	}

	/* Calculate statistics */
	var sum, min, max float64
	min = rows[0].Value
	max = rows[0].Value
	for _, row := range rows {
		sum += row.Value
		if row.Value < min {
			min = row.Value
		}
		if row.Value > max {
			max = row.Value
		}
	}

	avg := sum / float64(len(rows))

	/* Calculate trend (simple linear regression) */
	trend := m.calculateTrend(rows)

	return &TimeSeriesAnalysis{
		Metric:    metric,
		StartTime: startTime,
		EndTime:   endTime,
		Count:     len(rows),
		Average:   avg,
		Min:       min,
		Max:       max,
		Trend:     trend,
	}, nil
}

/* ClassifyTask categorizes tasks and messages automatically */
func (m *MLPoweredFeatures) ClassifyTask(ctx context.Context, task string) (string, error) {
	/* Get or train classification model */
	modelID, err := m.getOrTrainClassificationModel(ctx)
	if err != nil {
		/* Fallback to simple keyword-based classification */
		return m.simpleClassification(task), nil
	}

	/* Extract features */
	taskEmbedding, err := m.embed.Embed(ctx, task, "all-MiniLM-L6-v2")
	if err != nil {
		return m.simpleClassification(task), nil
	}

	/* Predict category */
	categoryID, err := m.mlClient.Predict(ctx, modelID, taskEmbedding)
	if err != nil {
		return m.simpleClassification(task), nil
	}

	/* Map category ID to name */
	category := m.mapCategoryID(int(categoryID))
	return category, nil
}

/* PredictResourceUsage predicts resource usage and costs */
func (m *MLPoweredFeatures) PredictResourceUsage(ctx context.Context, agentID uuid.UUID, task TaskFeatures) (*ResourcePrediction, error) {
	/* Get or train regression model */
	modelID, err := m.getOrTrainResourceModel(ctx, agentID)
	if err != nil {
		return &ResourcePrediction{
			Tokens:       1000,
			Latency:      1.0,
			Cost:         0.01,
			Confidence:   0.5,
		}, nil
	}

	/* Extract features */
	features := m.extractTaskFeatures(task)

	/* Predict tokens (regression) */
	tokens, err := m.mlClient.Predict(ctx, modelID, features)
	if err != nil {
		return &ResourcePrediction{
			Tokens:       1000,
			Latency:      1.0,
			Cost:         0.01,
			Confidence:   0.5,
		}, nil
	}

	/* Estimate latency and cost from tokens */
	latency := tokens / 1000.0 /* Assume 1000 tokens/second */
	cost := tokens * 0.00001   /* Assume $0.00001 per token */

	return &ResourcePrediction{
		Tokens:     int(tokens),
		Latency:    latency,
		Cost:       cost,
		Confidence: 0.7,
	}, nil
}

/* FeatureEngineering automatically extracts features from agent data */
func (m *MLPoweredFeatures) FeatureEngineering(ctx context.Context, data interface{}) ([]float32, error) {
	/* Extract features based on data type */
	switch v := data.(type) {
	case string:
		/* Text feature extraction */
		embedding, err := m.embed.Embed(ctx, v, "all-MiniLM-L6-v2")
		if err != nil {
			return nil, err
		}
		/* Reduce dimensions if needed */
		return m.reduceDimensions(embedding, 50), nil
	case map[string]interface{}:
		/* Structured data feature extraction */
		return m.extractStructuredFeatures(v), nil
	default:
		return nil, fmt.Errorf("feature engineering failed: unsupported_data_type=true")
	}
}

/* Helper types */

type MemoryCluster struct {
	ClusterID int
	MemoryIDs []int
	Size      int
}

type BehaviorMetric struct {
	Timestamp time.Time
	Value     float64
	Type      string
}

type Anomaly struct {
	AgentID   uuid.UUID
	Timestamp time.Time
	Metric    string
	Value     float64
	ZScore    float64
	Severity  string
}

type TaskFeatures struct {
	TaskLength    int
	NumTools      int
	Complexity    float64
	SimilarTasks  int
}

type ToolRecommendation struct {
	ToolName   string
	Confidence float64
	Reason     string
}

type TimeSeriesAnalysis struct {
	Metric    string
	StartTime time.Time
	EndTime   time.Time
	Count     int
	Average   float64
	Min       float64
	Max       float64
	Trend     float64 /* Positive = increasing, Negative = decreasing */
}

type ResourcePrediction struct {
	Tokens     int
	Latency    float64 /* seconds */
	Cost       float64 /* dollars */
	Confidence float64
}

/* Helper methods */

func (m *MLPoweredFeatures) extractFeatures(metric BehaviorMetric) []float32 {
	/* Extract features from metric */
	features := make([]float32, 10)
	features[0] = float32(metric.Value)
	features[1] = float32(metric.Timestamp.Hour())
	features[2] = float32(metric.Timestamp.Weekday())
	/* Add more features as needed */
	return features
}

func (m *MLPoweredFeatures) calculateSeverity(zScore float64) string {
	if zScore > 5 {
		return "critical"
	} else if zScore > 3 {
		return "high"
	} else if zScore > 2 {
		return "medium"
	}
	return "low"
}

func (m *MLPoweredFeatures) getOrTrainSuccessModel(ctx context.Context, agentID uuid.UUID) (int, error) {
	/* Check if model exists */
	models, err := m.mlClient.ListModels(ctx, "task_success")
	if err == nil && len(models) > 0 {
		return models[0].ModelID, nil
	}

	/* Model doesn't exist - would need training data */
	/* For now, return error to indicate model needs training */
	return 0, fmt.Errorf("success model not found: model_needs_training=true")
}

func (m *MLPoweredFeatures) extractTaskFeatures(task TaskFeatures) []float32 {
	return []float32{
		float32(task.TaskLength),
		float32(task.NumTools),
		float32(task.Complexity),
		float32(task.SimilarTasks),
	}
}

func (m *MLPoweredFeatures) defaultToolRecommendations(availableTools []string) []ToolRecommendation {
	var recommendations []ToolRecommendation
	for _, tool := range availableTools {
		recommendations = append(recommendations, ToolRecommendation{
			ToolName:   tool,
			Confidence: 0.5,
			Reason:     "Default recommendation",
		})
	}
	return recommendations
}

type MetricRow struct {
	Timestamp time.Time `db:"timestamp"`
	Value     float64   `db:"value"`
}

func (m *MLPoweredFeatures) calculateTrend(rows []MetricRow) float64 {
	if len(rows) < 2 {
		return 0
	}

	/* Simple linear trend: (last - first) / count */
	first := rows[0].Value
	last := rows[len(rows)-1].Value
	return (last - first) / float64(len(rows))
}

func (m *MLPoweredFeatures) getOrTrainClassificationModel(ctx context.Context) (int, error) {
	/* Check if model exists */
	models, err := m.mlClient.ListModels(ctx, "task_classification")
	if err == nil && len(models) > 0 {
		return models[0].ModelID, nil
	}

	return 0, fmt.Errorf("classification model not found: model_needs_training=true")
}

func (m *MLPoweredFeatures) simpleClassification(task string) string {
	taskLower := task
	if len(taskLower) > 500 {
		taskLower = taskLower[:500]
	}

	if stringContains(taskLower, "search") || stringContains(taskLower, "find") {
		return "search"
	}
	if stringContains(taskLower, "analyze") || stringContains(taskLower, "analyze") {
		return "analysis"
	}
	if stringContains(taskLower, "create") || stringContains(taskLower, "generate") {
		return "generation"
	}

	return "general"
}

func (m *MLPoweredFeatures) mapCategoryID(categoryID int) string {
	categories := map[int]string{
		0: "general",
		1: "search",
		2: "analysis",
		3: "generation",
		4: "code",
		5: "data",
	}
	if category, ok := categories[categoryID]; ok {
		return category
	}
	return "general"
}

func (m *MLPoweredFeatures) getOrTrainResourceModel(ctx context.Context, agentID uuid.UUID) (int, error) {
	models, err := m.mlClient.ListModels(ctx, "resource_prediction")
	if err == nil && len(models) > 0 {
		return models[0].ModelID, nil
	}
	return 0, fmt.Errorf("resource model not found: model_needs_training=true")
}

func (m *MLPoweredFeatures) reduceDimensions(embedding []float32, targetDim int) []float32 {
	if len(embedding) <= targetDim {
		return embedding
	}

	/* Simple reduction: take first targetDim dimensions */
	return embedding[:targetDim]
}

func (m *MLPoweredFeatures) extractStructuredFeatures(data map[string]interface{}) []float32 {
	/* Extract numeric features from structured data */
	features := make([]float32, 0, 10)

	for _, value := range data {
		switch v := value.(type) {
		case float64:
			features = append(features, float32(v))
		case int:
			features = append(features, float32(v))
		case bool:
			if v {
				features = append(features, 1.0)
			} else {
				features = append(features, 0.0)
			}
		}
		if len(features) >= 10 {
			break
		}
	}

	/* Pad to 10 features */
	for len(features) < 10 {
		features = append(features, 0.0)
	}

	return features
}

