package tools

import (
	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// RegisterAllTools registers all available tools with the registry
func RegisterAllTools(registry *ToolRegistry, db *database.Database, logger *logging.Logger) {
	// Vector search tools
	registry.Register(NewVectorSearchTool(db, logger))
	registry.Register(NewVectorSearchL2Tool(db, logger))
	registry.Register(NewVectorSearchCosineTool(db, logger))
	registry.Register(NewVectorSearchInnerProductTool(db, logger))

	// Embedding tools
	registry.Register(NewGenerateEmbeddingTool(db, logger))
	registry.Register(NewBatchEmbeddingTool(db, logger))

	// Note: Additional tools (L1, Hamming, Chebyshev, Minkowski, operations, indexing, ML, analytics, etc.)
	// would be registered here. For brevity, implementing core functionality first.
	// Full implementation would include all tools from the TypeScript version.
}

