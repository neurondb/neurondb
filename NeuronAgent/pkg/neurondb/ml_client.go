/*-------------------------------------------------------------------------
 *
 * ml_client.go
 *    ML operations via NeuronDB
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/pkg/neurondb/ml_client.go
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

/* MLClient handles ML operations via NeuronDB */
type MLClient struct {
	db *sqlx.DB
}

/* NewMLClient creates a new ML client */
func NewMLClient(db *sqlx.DB) *MLClient {
	return &MLClient{db: db}
}

/* TrainModel trains an ML model using NeuronDB */
func (c *MLClient) TrainModel(ctx context.Context, project, algorithm, tableName, labelCol string, featureCols []string, params map[string]interface{}) (int, error) {
	/* Validate inputs */
	if algorithm == "" {
		return 0, fmt.Errorf("ML training failed: algorithm_empty=true")
	}
	if tableName == "" {
		return 0, fmt.Errorf("ML training failed: table_name_empty=true")
	}
	if labelCol == "" {
		return 0, fmt.Errorf("ML training failed: label_col_empty=true")
	}
	if len(featureCols) == 0 {
		return 0, fmt.Errorf("ML training failed: feature_cols_empty=true")
	}
	if project == "" {
		project = "default"
	}

	paramsJSON, err := json.Marshal(params)
	if err != nil {
		return 0, fmt.Errorf("ML training failed: project='%s', algorithm='%s', table_name='%s', parameter_marshaling_error=true, error=%w",
			project, algorithm, tableName, err)
	}

	var modelID int
	query := `SELECT neurondb.train($1, $2, $3, $4, $5::text[], $6::jsonb) AS model_id`

	err = c.db.GetContext(ctx, &modelID, query, project, algorithm, tableName, labelCol, featureCols, paramsJSON)
	if err != nil {
		return 0, fmt.Errorf("ML training failed via NeuronDB: project='%s', algorithm='%s', table_name='%s', label_col='%s', feature_cols_count=%d, function='neurondb.train', error=%w",
			project, algorithm, tableName, labelCol, len(featureCols), err)
	}

	return modelID, nil
}

/* Predict makes a prediction using a trained model */
func (c *MLClient) Predict(ctx context.Context, modelID int, features []float32) (float64, error) {
	var prediction float64
	query := `SELECT neurondb.predict($1, $2::real[]) AS prediction`

	err := c.db.GetContext(ctx, &prediction, query, modelID, features)
	if err != nil {
		return 0, fmt.Errorf("ML prediction failed via NeuronDB: model_id=%d, features_dimension=%d, function='neurondb.predict', error=%w",
			modelID, len(features), err)
	}

	return prediction, nil
}

/* PredictBatch makes batch predictions */
func (c *MLClient) PredictBatch(ctx context.Context, modelID int, features [][]float32) ([]float64, error) {
	/* Convert to array format for PostgreSQL */
	query := `SELECT array_agg(prediction) AS predictions
		FROM (
			SELECT neurondb.predict($1, unnest($2::real[][])::real[]) AS prediction
		) sub`

	var predictions []float64
	err := c.db.GetContext(ctx, &predictions, query, modelID, features)
	if err != nil {
		return nil, fmt.Errorf("ML batch prediction failed via NeuronDB: model_id=%d, batch_size=%d, function='neurondb.predict', error=%w",
			modelID, len(features), err)
	}

	return predictions, nil
}

/* EvaluateModel evaluates a model's performance */
func (c *MLClient) EvaluateModel(ctx context.Context, modelID int, testTable, labelCol string, featureCols []string) (map[string]interface{}, error) {
	query := `SELECT neurondb.evaluate($1, $2, $3, $4::text[]) AS metrics`

	var metricsJSON string
	err := c.db.GetContext(ctx, &metricsJSON, query, modelID, testTable, labelCol, featureCols)
	if err != nil {
		return nil, fmt.Errorf("ML evaluation failed via NeuronDB: model_id=%d, test_table='%s', label_col='%s', feature_cols_count=%d, function='neurondb.evaluate', error=%w",
			modelID, testTable, labelCol, len(featureCols), err)
	}

	var metrics map[string]interface{}
	if err := json.Unmarshal([]byte(metricsJSON), &metrics); err != nil {
		return nil, fmt.Errorf("ML evaluation metrics parsing failed: model_id=%d, metrics_json_length=%d, error=%w",
			modelID, len(metricsJSON), err)
	}

	return metrics, nil
}

/* ListModels lists all trained models */
func (c *MLClient) ListModels(ctx context.Context, project string) ([]ModelInfo, error) {
	query := `SELECT model_id, project_name, algorithm, table_name, label_col, feature_cols, 
		created_at, updated_at, status
		FROM neurondb.ml_models
		WHERE project_name = $1
		ORDER BY created_at DESC`

	var models []ModelInfo
	err := c.db.SelectContext(ctx, &models, query, project)
	if err != nil {
		return nil, fmt.Errorf("ML model listing failed via NeuronDB: project='%s', error=%w", project, err)
	}

	return models, nil
}

/* GetModelInfo gets information about a specific model */
func (c *MLClient) GetModelInfo(ctx context.Context, modelID int) (*ModelInfo, error) {
	query := `SELECT model_id, project_name, algorithm, table_name, label_col, feature_cols,
		created_at, updated_at, status
		FROM neurondb.ml_models
		WHERE model_id = $1`

	var model ModelInfo
	err := c.db.GetContext(ctx, &model, query, modelID)
	if err != nil {
		return nil, fmt.Errorf("ML model info retrieval failed via NeuronDB: model_id=%d, error=%w", modelID, err)
	}

	return &model, nil
}

/* DeleteModel deletes a trained model */
func (c *MLClient) DeleteModel(ctx context.Context, modelID int) error {
	query := `DELETE FROM neurondb.ml_models WHERE model_id = $1`

	result, err := c.db.ExecContext(ctx, query, modelID)
	if err != nil {
		return fmt.Errorf("ML model deletion failed via NeuronDB: model_id=%d, error=%w", modelID, err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("ML model deletion failed: model_id=%d, rows_affected_check_error=%w", modelID, err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("ML model deletion failed: model_id=%d, model_not_found=true", modelID)
	}

	return nil
}

/* ModelInfo represents information about a trained model */
type ModelInfo struct {
	ModelID     int      `db:"model_id"`
	ProjectName string   `db:"project_name"`
	Algorithm   string   `db:"algorithm"`
	TableName   string   `db:"table_name"`
	LabelCol    string   `db:"label_col"`
	FeatureCols []string `db:"feature_cols"`
	CreatedAt   string   `db:"created_at"`
	UpdatedAt   string   `db:"updated_at"`
	Status      string   `db:"status"`
}
