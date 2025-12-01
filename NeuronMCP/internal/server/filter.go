package server

import (
	"github.com/neurondb/NeuronMCP/internal/config"
	"github.com/neurondb/NeuronMCP/internal/tools"
)

// filterToolsByFeatures filters tools based on feature flags
func (s *Server) filterToolsByFeatures(definitions []tools.ToolDefinition) []tools.ToolDefinition {
	features := s.config.GetFeaturesConfig()
	filtered := make([]tools.ToolDefinition, 0, len(definitions))
	
	for _, def := range definitions {
		if shouldIncludeTool(def.Name, features) {
			filtered = append(filtered, def)
		}
	}
	
	return filtered
}

// shouldIncludeTool determines if a tool should be included based on feature flags
func shouldIncludeTool(toolName string, features *config.FeaturesConfig) bool {
	// Vector tools
	if isVectorTool(toolName) {
		return features.Vector != nil && features.Vector.Enabled
	}
	
	// ML tools
	if isMLTool(toolName) {
		return features.ML != nil && features.ML.Enabled
	}
	
	// Analytics tools
	if isAnalyticsTool(toolName) {
		return features.Analytics != nil && features.Analytics.Enabled
	}
	
	// RAG tools
	if isRAGTool(toolName) {
		return features.RAG != nil && features.RAG.Enabled
	}
	
	// Project tools
	if isProjectTool(toolName) {
		return features.Projects != nil && features.Projects.Enabled
	}
	
	// GPU tools
	if isGPUTool(toolName) {
		return features.GPU != nil && features.GPU.Enabled
	}
	
	// Default: include if no specific feature flag
	return true
}

// Tool category checkers
func isVectorTool(name string) bool {
	vectorPrefixes := []string{"vector_", "embed_", "generate_embedding", "batch_embedding", "create_hnsw_index", "drop_index"}
	for _, prefix := range vectorPrefixes {
		if len(name) >= len(prefix) && name[:len(prefix)] == prefix {
			return true
		}
	}
	return false
}

func isMLTool(name string) bool {
	mlPrefixes := []string{"train_", "predict_", "get_model_info", "list_models", "delete_model", "model_metrics"}
	for _, prefix := range mlPrefixes {
		if len(name) >= len(prefix) && name[:len(prefix)] == prefix {
			return true
		}
	}
	return false
}

func isAnalyticsTool(name string) bool {
	analyticsPrefixes := []string{"cluster_", "detect_"}
	for _, prefix := range analyticsPrefixes {
		if len(name) >= len(prefix) && name[:len(prefix)] == prefix {
			return true
		}
	}
	return false
}

func isRAGTool(name string) bool {
	ragPrefixes := []string{"rag_", "chunk_"}
	for _, prefix := range ragPrefixes {
		if len(name) >= len(prefix) && name[:len(prefix)] == prefix {
			return true
		}
	}
	return false
}

func isProjectTool(name string) bool {
	projectPrefixes := []string{"create_ml_project", "list_ml_projects", "project_"}
	for _, prefix := range projectPrefixes {
		if len(name) >= len(prefix) && name[:len(prefix)] == prefix {
			return true
		}
	}
	return false
}

func isGPUTool(name string) bool {
	gpuPrefixes := []string{"gpu_"}
	for _, prefix := range gpuPrefixes {
		if len(name) >= len(prefix) && name[:len(prefix)] == prefix {
			return true
		}
	}
	return false
}

