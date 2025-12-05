package test

import (
	"context"
	"testing"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/tools"
)

// TestAllToolsRegistered tests that all tools are properly registered
func TestAllToolsRegistered(t *testing.T) {
	db := database.NewDatabase()
	logger := logging.NewLogger(logging.Config{
		Level:  "info",
		Format: "text",
		Output: "stderr",
	})

	registry := tools.NewToolRegistry(db, logger)
	tools.RegisterAllTools(registry, db, logger)

	// Expected tool categories and counts
	expectedTools := []string{
		// Vector operations
		"vector_search", "vector_search_l2", "vector_search_cosine", "vector_search_inner_product",
		"vector_similarity", "vector_arithmetic", "vector_distance", "vector_similarity_unified",
		// Quantization
		"vector_quantize", "quantization_analyze",
		// Embeddings
		"generate_embedding", "batch_embedding", "embed_image", "embed_multimodal", "embed_cached",
		"configure_embedding_model", "get_embedding_model_config", "list_embedding_model_configs", "delete_embedding_model_config",
		// Hybrid search
		"hybrid_search", "reciprocal_rank_fusion", "semantic_keyword_search", "multi_vector_search",
		"faceted_vector_search", "temporal_vector_search", "diverse_vector_search",
		// Reranking
		"rerank_cross_encoder", "rerank_llm", "rerank_cohere", "rerank_colbert", "rerank_ltr", "rerank_ensemble",
		// ML
		"train_model", "predict", "predict_batch", "evaluate_model", "list_models", "get_model_info", "delete_model", "export_model",
		// Analytics
		"analyze_data", "cluster_data", "reduce_dimensionality", "detect_outliers", "quality_metrics", "detect_drift", "topic_discovery",
		// Time series
		"timeseries_analysis",
		// AutoML
		"automl",
		// ONNX
		"onnx_model",
		// Indexing
		"create_hnsw_index", "create_ivf_index", "index_status", "drop_index", "tune_hnsw_index", "tune_ivf_index",
		// RAG
		"process_document", "retrieve_context", "generate_response", "chunk_document",
		// Workers & GPU
		"worker_management", "gpu_info",
		// PostgreSQL
		"postgresql_version", "postgresql_stats", "postgresql_databases", "postgresql_connections",
		"postgresql_locks", "postgresql_replication", "postgresql_settings", "postgresql_extensions",
	}

	for _, toolName := range expectedTools {
		tool := registry.GetTool(toolName)
		if tool == nil {
			t.Errorf("Expected tool '%s' not found in registry", toolName)
		} else {
			if tool.Name() != toolName {
				t.Errorf("Tool name mismatch: expected '%s', got '%s'", toolName, tool.Name())
			}
		}
	}
}

// TestToolValidation tests parameter validation
func TestToolValidation(t *testing.T) {
	db := database.NewDatabase()
	logger := logging.NewLogger(logging.Config{
		Level:  "info",
		Format: "text",
		Output: "stderr",
	})

	registry := tools.NewToolRegistry(db, logger)
	tools.RegisterAllTools(registry, db, logger)

	// Test vector_search validation
	tool := registry.GetTool("vector_search")
	if tool == nil {
		t.Fatal("vector_search tool not found")
	}

	// Test with missing required parameter
	ctx := context.Background()
	result, err := tool.Execute(ctx, map[string]interface{}{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if result.Success {
		t.Error("Expected validation error for missing parameters")
	}
	if result.Error == nil {
		t.Error("Expected error result for missing parameters")
	}
}

// TestPostgreSQLTools tests PostgreSQL tools (no DB connection required for version)
func TestPostgreSQLTools(t *testing.T) {
	db := database.NewDatabase()
	logger := logging.NewLogger(logging.Config{
		Level:  "info",
		Format: "text",
		Output: "stderr",
	})

	// Note: These tests require a database connection
	// They are integration tests and should be run with a test database
	t.Skip("Skipping integration tests - requires database connection")
}

