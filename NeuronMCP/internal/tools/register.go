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

	// Additional vector tools
	registry.Register(NewVectorSimilarityTool(db, logger))
	registry.Register(NewCreateVectorIndexTool(db, logger))

	// ML tools
	registry.Register(NewTrainModelTool(db, logger))
	registry.Register(NewPredictTool(db, logger))
	registry.Register(NewEvaluateModelTool(db, logger))
	registry.Register(NewListModelsTool(db, logger))
	registry.Register(NewGetModelInfoTool(db, logger))
	registry.Register(NewDeleteModelTool(db, logger))

	// Analytics tools
	registry.Register(NewClusterDataTool(db, logger))
	registry.Register(NewDetectOutliersTool(db, logger))
	registry.Register(NewReduceDimensionalityTool(db, logger))

	// RAG tools
	registry.Register(NewProcessDocumentTool(db, logger))
	registry.Register(NewRetrieveContextTool(db, logger))
	registry.Register(NewGenerateResponseTool(db, logger))
	registry.Register(NewChunkDocumentTool(db, logger))

	// Indexing tools
	registry.Register(NewCreateHNSWIndexTool(db, logger))
	registry.Register(NewCreateIVFIndexTool(db, logger))
	registry.Register(NewIndexStatusTool(db, logger))
	registry.Register(NewDropIndexTool(db, logger))
	registry.Register(NewTuneHNSWIndexTool(db, logger))
	registry.Register(NewTuneIVFIndexTool(db, logger))

	// Additional ML tools
	registry.Register(NewPredictBatchTool(db, logger))
	registry.Register(NewExportModelTool(db, logger))

	// Analytics tools
	registry.Register(NewAnalyzeDataTool(db, logger))

	// Hybrid search tools
	registry.Register(NewHybridSearchTool(db, logger))
	registry.Register(NewReciprocalRankFusionTool(db, logger))
	registry.Register(NewSemanticKeywordSearchTool(db, logger))
	registry.Register(NewMultiVectorSearchTool(db, logger))
	registry.Register(NewFacetedVectorSearchTool(db, logger))
	registry.Register(NewTemporalVectorSearchTool(db, logger))
	registry.Register(NewDiverseVectorSearchTool(db, logger))

	// Reranking tools
	registry.Register(NewRerankCrossEncoderTool(db, logger))
	registry.Register(NewRerankLLMTool(db, logger))
	registry.Register(NewRerankCohereTool(db, logger))
	registry.Register(NewRerankColBERTTool(db, logger))
	registry.Register(NewRerankLTRTool(db, logger))
	registry.Register(NewRerankEnsembleTool(db, logger))

	// Advanced vector operations
	registry.Register(NewVectorArithmeticTool(db, logger))
	registry.Register(NewVectorDistanceTool(db, logger))
	registry.Register(NewVectorSimilarityUnifiedTool(db, logger))

	// Quantization tools
	registry.Register(NewVectorQuantizationTool(db, logger))
	registry.Register(NewQuantizationAnalysisTool(db, logger))

	// Complete embedding tools
	registry.Register(NewEmbedImageTool(db, logger))
	registry.Register(NewEmbedMultimodalTool(db, logger))
	registry.Register(NewEmbedCachedTool(db, logger))
	registry.Register(NewConfigureEmbeddingModelTool(db, logger))
	registry.Register(NewGetEmbeddingModelConfigTool(db, logger))
	registry.Register(NewListEmbeddingModelConfigsTool(db, logger))
	registry.Register(NewDeleteEmbeddingModelConfigTool(db, logger))

	// Quality metrics, drift detection, topic discovery
	registry.Register(NewQualityMetricsTool(db, logger))
	registry.Register(NewDriftDetectionTool(db, logger))
	registry.Register(NewTopicDiscoveryTool(db, logger))

	// Time series, AutoML, ONNX
	registry.Register(NewTimeSeriesTool(db, logger))
	registry.Register(NewAutoMLTool(db, logger))
	registry.Register(NewONNXTool(db, logger))

	// Vector graph operations
	registry.Register(NewVectorGraphTool(db, logger))

	// Vecmap operations
	registry.Register(NewVecmapOperationsTool(db, logger))

	// Dataset loading
	registry.Register(NewDatasetLoadingTool(db, logger))

	// Workers and GPU
	registry.Register(NewWorkerManagementTool(db, logger))
	registry.Register(NewGPUMonitoringTool(db, logger))

	// PostgreSQL tools
	registry.Register(NewPostgreSQLVersionTool(db, logger))
	registry.Register(NewPostgreSQLStatsTool(db, logger))
	registry.Register(NewPostgreSQLDatabaseListTool(db, logger))
	registry.Register(NewPostgreSQLConnectionsTool(db, logger))
	registry.Register(NewPostgreSQLLocksTool(db, logger))
	registry.Register(NewPostgreSQLReplicationTool(db, logger))
	registry.Register(NewPostgreSQLSettingsTool(db, logger))
	registry.Register(NewPostgreSQLExtensionsTool(db, logger))
}

