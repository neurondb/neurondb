/*-------------------------------------------------------------------------
 *
 * analytics_client.go
 *    Analytics operations via NeuronDB
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/pkg/neurondb/analytics_client.go
 *
 *-------------------------------------------------------------------------
 */

package neurondb

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/jmoiron/sqlx"
)

/* AnalyticsClient handles analytics operations via NeuronDB */
type AnalyticsClient struct {
	db *sqlx.DB
}

/* NewAnalyticsClient creates a new analytics client */
func NewAnalyticsClient(db *sqlx.DB) *AnalyticsClient {
	return &AnalyticsClient{db: db}
}

/* ClusterData performs clustering on data */
func (c *AnalyticsClient) ClusterData(ctx context.Context, tableName, featureCol string, algorithm string, params map[string]interface{}) ([]ClusterResult, error) {
	paramsJSON, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("clustering failed: table_name='%s', feature_col='%s', algorithm='%s', parameter_marshaling_error=true, error=%w",
			tableName, featureCol, algorithm, err)
	}

	var resultsJSON string
	query := `SELECT neurondb_cluster($1, $2, $3, $4::jsonb) AS clusters`

	err = c.db.GetContext(ctx, &resultsJSON, query, tableName, featureCol, algorithm, paramsJSON)
	if err != nil {
		return nil, fmt.Errorf("clustering failed via NeuronDB: table_name='%s', feature_col='%s', algorithm='%s', function='neurondb_cluster', error=%w",
			tableName, featureCol, algorithm, err)
	}

	var results []ClusterResult
	if err := json.Unmarshal([]byte(resultsJSON), &results); err != nil {
		return nil, fmt.Errorf("clustering result parsing failed: table_name='%s', algorithm='%s', results_json_length=%d, error=%w",
			tableName, algorithm, len(resultsJSON), err)
	}

	return results, nil
}

/* DetectOutliers detects outliers in data */
func (c *AnalyticsClient) DetectOutliers(ctx context.Context, tableName, featureCol string, method string, params map[string]interface{}) ([]OutlierResult, error) {
	paramsJSON, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("outlier detection failed: table_name='%s', feature_col='%s', method='%s', parameter_marshaling_error=true, error=%w",
			tableName, featureCol, method, err)
	}

	var resultsJSON string
	query := `SELECT neurondb_detect_outliers($1, $2, $3, $4::jsonb) AS outliers`

	err = c.db.GetContext(ctx, &resultsJSON, query, tableName, featureCol, method, paramsJSON)
	if err != nil {
		return nil, fmt.Errorf("outlier detection failed via NeuronDB: table_name='%s', feature_col='%s', method='%s', function='neurondb_detect_outliers', error=%w",
			tableName, featureCol, method, err)
	}

	var results []OutlierResult
	if err := json.Unmarshal([]byte(resultsJSON), &results); err != nil {
		return nil, fmt.Errorf("outlier detection result parsing failed: table_name='%s', method='%s', results_json_length=%d, error=%w",
			tableName, method, len(resultsJSON), err)
	}

	return results, nil
}

/* ReduceDimensionality performs dimensionality reduction */
func (c *AnalyticsClient) ReduceDimensionality(ctx context.Context, tableName, featureCol string, method string, targetDim int, params map[string]interface{}) ([]DimensionalityResult, error) {
	paramsJSON, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("dimensionality reduction failed: table_name='%s', feature_col='%s', method='%s', target_dim=%d, parameter_marshaling_error=true, error=%w",
			tableName, featureCol, method, targetDim, err)
	}

	var resultsJSON string
	query := `SELECT neurondb_reduce_dimensionality($1, $2, $3, $4, $5::jsonb) AS reduced`

	err = c.db.GetContext(ctx, &resultsJSON, query, tableName, featureCol, method, targetDim, paramsJSON)
	if err != nil {
		return nil, fmt.Errorf("dimensionality reduction failed via NeuronDB: table_name='%s', feature_col='%s', method='%s', target_dim=%d, function='neurondb_reduce_dimensionality', error=%w",
			tableName, featureCol, method, targetDim, err)
	}

	var results []DimensionalityResult
	if err := json.Unmarshal([]byte(resultsJSON), &results); err != nil {
		return nil, fmt.Errorf("dimensionality reduction result parsing failed: table_name='%s', method='%s', results_json_length=%d, error=%w",
			tableName, method, len(resultsJSON), err)
	}

	return results, nil
}

/* AnalyzeData performs general data analysis */
func (c *AnalyticsClient) AnalyzeData(ctx context.Context, tableName string, columns []string) (map[string]interface{}, error) {
	var resultsJSON string
	query := `SELECT neurondb_analyze_data($1, $2::text[]) AS analysis`

	err := c.db.GetContext(ctx, &resultsJSON, query, tableName, columns)
	if err != nil {
		return nil, fmt.Errorf("data analysis failed via NeuronDB: table_name='%s', columns_count=%d, function='neurondb_analyze_data', error=%w",
			tableName, len(columns), err)
	}

	var analysis map[string]interface{}
	if err := json.Unmarshal([]byte(resultsJSON), &analysis); err != nil {
		return nil, fmt.Errorf("data analysis result parsing failed: table_name='%s', analysis_json_length=%d, error=%w",
			tableName, len(resultsJSON), err)
	}

	return analysis, nil
}

/* ClusterResult represents a clustering result */
type ClusterResult struct {
	ID        interface{} `json:"id"`
	ClusterID int         `json:"cluster_id"`
	Distance  float64     `json:"distance"`
}

/* OutlierResult represents an outlier detection result */
type OutlierResult struct {
	ID        interface{} `json:"id"`
	IsOutlier bool        `json:"is_outlier"`
	Score     float64     `json:"score"`
}

/* DimensionalityResult represents a dimensionality reduction result */
type DimensionalityResult struct {
	ID      interface{} `json:"id"`
	Reduced Vector      `json:"reduced"`
}
