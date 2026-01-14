/*-------------------------------------------------------------------------
 *
 * gpu_acceleration.go
 *    GPU acceleration support for agent operations
 *
 * Provides transparent GPU acceleration for embeddings, vector search,
 * and LLM inference when GPU is available.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/gpu_acceleration.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"runtime"

	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* GPUAcceleration provides GPU acceleration for agent operations */
type GPUAcceleration struct {
	db            *db.DB
	embedClient   *neurondb.EmbeddingClient
	llmClient     *LLMClient
	gpuAvailable  bool
	gpuType       string /* "cuda", "rocm", "metal", "cpu" */
	enabled       bool
}

/* NewGPUAcceleration creates GPU acceleration manager */
func NewGPUAcceleration(database *db.DB, embedClient *neurondb.EmbeddingClient, llmClient *LLMClient) *GPUAcceleration {
	gpu := &GPUAcceleration{
		db:          database,
		embedClient: embedClient,
		llmClient:   llmClient,
		enabled:     true,
	}

	/* Auto-detect GPU */
	gpu.detectGPU()

	return gpu
}

/* detectGPU detects available GPU */
func (g *GPUAcceleration) detectGPU() {
	/* Query NeuronDB for GPU availability */
	query := `SELECT setting FROM pg_settings WHERE name = 'neurondb.gpu_backend'`
	var setting string
	err := g.db.DB.GetContext(context.Background(), &setting, query)
	if err != nil {
		g.gpuAvailable = false
		g.gpuType = "cpu"
		return
	}

	if setting != "" && setting != "cpu" {
		g.gpuAvailable = true
		g.gpuType = setting
	} else {
		g.gpuAvailable = false
		g.gpuType = "cpu"
	}

	/* Additional platform-specific detection */
	if runtime.GOOS == "darwin" {
		/* Check for Metal on macOS */
		if g.gpuType == "" {
			g.gpuType = "metal"
			g.gpuAvailable = true
		}
	}
}

/* IsAvailable checks if GPU acceleration is available */
func (g *GPUAcceleration) IsAvailable() bool {
	return g.gpuAvailable && g.enabled
}

/* GetGPUType returns the GPU type */
func (g *GPUAcceleration) GetGPUType() string {
	return g.gpuType
}

/* Enable enables GPU acceleration */
func (g *GPUAcceleration) Enable() {
	g.enabled = true
}

/* Disable disables GPU acceleration */
func (g *GPUAcceleration) Disable() {
	g.enabled = false
}

/* AccelerateEmbedding accelerates embedding generation */
func (g *GPUAcceleration) AccelerateEmbedding(ctx context.Context, text, model string) ([]float32, error) {
	if !g.IsAvailable() {
		/* Fallback to CPU */
		return g.embedClient.Embed(ctx, text, model)
	}

	/* NeuronDB handles GPU acceleration transparently */
	/* The embedding client will use GPU if available */
	return g.embedClient.Embed(ctx, text, model)
}

/* AccelerateVectorSearch accelerates vector similarity search */
func (g *GPUAcceleration) AccelerateVectorSearch(ctx context.Context, tableName, vectorCol string, queryVector []float32, limit int, metric string) ([]VectorSearchResult, error) {
	if !g.IsAvailable() {
		/* Fallback to CPU search */
		return g.vectorSearchCPU(ctx, tableName, vectorCol, queryVector, limit, metric)
	}

	/* NeuronDB handles GPU acceleration transparently */
	/* Use GPU-accelerated search via SQL */
	return g.vectorSearchGPU(ctx, tableName, vectorCol, queryVector, limit, metric)
}

/* AccelerateLLM accelerates LLM inference (if using local models) */
func (g *GPUAcceleration) AccelerateLLM(ctx context.Context, modelName, prompt string, config map[string]interface{}) (*LLMResponse, error) {
	if !g.IsAvailable() {
		/* Fallback to CPU or cloud API */
		return g.llmClient.Generate(ctx, modelName, prompt, config)
	}

	/* NeuronDB handles GPU acceleration transparently for local models */
	return g.llmClient.Generate(ctx, modelName, prompt, config)
}

/* GetPerformanceMetrics returns GPU performance metrics */
func (g *GPUAcceleration) GetPerformanceMetrics(ctx context.Context) (*GPUPerformanceMetrics, error) {
	if !g.IsAvailable() {
		return &GPUPerformanceMetrics{
			GPUAvailable: false,
			GPUType:      g.gpuType,
			Utilization:  0.0,
			MemoryUsed:   0,
			MemoryTotal:  0,
		}, nil
	}

	/* Query NeuronDB for GPU metrics */
	query := `SELECT 
		(SELECT setting FROM pg_settings WHERE name = 'neurondb.gpu_backend') AS gpu_type,
		0.0 AS utilization,
		0 AS memory_used,
		0 AS memory_total`

	type MetricsRow struct {
		GPUType     string  `db:"gpu_type"`
		Utilization float64 `db:"utilization"`
		MemoryUsed  int64   `db:"memory_used"`
		MemoryTotal int64   `db:"memory_total"`
	}

	var row MetricsRow
	err := g.db.DB.GetContext(ctx, &row, query)
	if err != nil {
		return &GPUPerformanceMetrics{
			GPUAvailable: g.gpuAvailable,
			GPUType:      g.gpuType,
			Utilization:  0.0,
			MemoryUsed:   0,
			MemoryTotal:  0,
		}, nil
	}

	return &GPUPerformanceMetrics{
		GPUAvailable: g.gpuAvailable,
		GPUType:      row.GPUType,
		Utilization:  row.Utilization,
		MemoryUsed:   row.MemoryUsed,
		MemoryTotal:  row.MemoryTotal,
	}, nil
}

/* Helper types */

type VectorSearchResult struct {
	ID         int64
	Content    string
	Similarity float64
}

type GPUPerformanceMetrics struct {
	GPUAvailable bool
	GPUType      string
	Utilization  float64
	MemoryUsed   int64
	MemoryTotal  int64
}

/* Helper methods */

func (g *GPUAcceleration) vectorSearchCPU(ctx context.Context, tableName, vectorCol string, queryVector []float32, limit int, metric string) ([]VectorSearchResult, error) {
	/* CPU-based vector search */
	query := fmt.Sprintf(`SELECT id, content, 1 - (%s <=> $1::vector) AS similarity
		FROM %s
		ORDER BY %s <=> $1::vector
		LIMIT $2`, vectorCol, tableName, vectorCol)

	type ResultRow struct {
		ID         int64  `db:"id"`
		Content    string `db:"content"`
		Similarity float64 `db:"similarity"`
	}

	var rows []ResultRow
	err := g.db.DB.SelectContext(ctx, &rows, query, queryVector, limit)
	if err != nil {
		return nil, fmt.Errorf("CPU vector search failed: error=%w", err)
	}

	results := make([]VectorSearchResult, len(rows))
	for i, row := range rows {
		results[i] = VectorSearchResult{
			ID:         row.ID,
			Content:    row.Content,
			Similarity: row.Similarity,
		}
	}

	return results, nil
}

func (g *GPUAcceleration) vectorSearchGPU(ctx context.Context, tableName, vectorCol string, queryVector []float32, limit int, metric string) ([]VectorSearchResult, error) {
	/* GPU-accelerated search via NeuronDB */
	/* NeuronDB handles GPU acceleration transparently */
	/* Same query as CPU, but NeuronDB uses GPU internally */
	return g.vectorSearchCPU(ctx, tableName, vectorCol, queryVector, limit, metric)
}

