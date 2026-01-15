/*-------------------------------------------------------------------------
 *
 * register.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/register.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"strings"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* RegisterAllTools registers all available tools with the registry */
func RegisterAllTools(registry *ToolRegistry, db *database.Database, logger *logging.Logger) {
	/* Vector search tools */
	registry.Register(NewVectorSearchTool(db, logger))
	registry.Register(NewVectorSearchL2Tool(db, logger))
	registry.Register(NewVectorSearchCosineTool(db, logger))
	registry.Register(NewVectorSearchInnerProductTool(db, logger))
	registry.Register(NewVectorSearchL1Tool(db, logger))
	registry.Register(NewVectorSearchHammingTool(db, logger))
	registry.Register(NewVectorSearchChebyshevTool(db, logger))
	registry.Register(NewVectorSearchMinkowskiTool(db, logger))

	/* Embedding tools */
	registry.Register(NewGenerateEmbeddingTool(db, logger))
	registry.Register(NewBatchEmbeddingTool(db, logger))

	/* Additional vector tools */
	registry.Register(NewVectorSimilarityTool(db, logger))
	registry.Register(NewCreateVectorIndexTool(db, logger))

	/* ML tools */
	registry.Register(NewTrainModelTool(db, logger))
	registry.Register(NewPredictTool(db, logger))
	registry.Register(NewEvaluateModelTool(db, logger))
	registry.Register(NewListModelsTool(db, logger))
	registry.Register(NewGetModelInfoTool(db, logger))
	registry.Register(NewDeleteModelTool(db, logger))

	/* Analytics tools */
	registry.Register(NewClusterDataTool(db, logger))
	registry.Register(NewDetectOutliersTool(db, logger))
	registry.Register(NewReduceDimensionalityTool(db, logger))

	/* RAG tools */
	registry.Register(NewProcessDocumentTool(db, logger))
	registry.Register(NewRetrieveContextTool(db, logger))
	registry.Register(NewGenerateResponseTool(db, logger))

	/* Composite RAG tools */
	registry.Register(NewIngestDocumentsTool(db, logger))
	registry.Register(NewAnswerWithCitationsTool(db, logger))
	registry.Register(NewChunkDocumentTool(db, logger))

	/* Indexing tools */
	registry.Register(NewCreateHNSWIndexTool(db, logger))
	registry.Register(NewCreateIVFIndexTool(db, logger))
	registry.Register(NewIndexStatusTool(db, logger))
	registry.Register(NewDropIndexTool(db, logger))
	registry.Register(NewTuneHNSWIndexTool(db, logger))
	registry.Register(NewTuneIVFIndexTool(db, logger))

	/* Additional ML tools */
	registry.Register(NewPredictBatchTool(db, logger))
	registry.Register(NewExportModelTool(db, logger))

	/* Analytics tools */
	registry.Register(NewAnalyzeDataTool(db, logger))

	/* Hybrid search tools */
	registry.Register(NewHybridSearchTool(db, logger))
	registry.Register(NewTextSearchTool(db, logger))
	registry.Register(NewReciprocalRankFusionTool(db, logger))
	registry.Register(NewSemanticKeywordSearchTool(db, logger))
	registry.Register(NewMultiVectorSearchTool(db, logger))
	registry.Register(NewFacetedVectorSearchTool(db, logger))
	registry.Register(NewTemporalVectorSearchTool(db, logger))
	registry.Register(NewDiverseVectorSearchTool(db, logger))

	/* Reranking tools */
	registry.Register(NewRerankCrossEncoderTool(db, logger))
	registry.Register(NewRerankLLMTool(db, logger))
	registry.Register(NewRerankCohereTool(db, logger))
	registry.Register(NewRerankColBERTTool(db, logger))
	registry.Register(NewRerankLTRTool(db, logger))
	registry.Register(NewRerankEnsembleTool(db, logger))

	/* Advanced vector operations */
	registry.Register(NewVectorArithmeticTool(db, logger))
	registry.Register(NewVectorDistanceTool(db, logger))
	registry.Register(NewVectorSimilarityUnifiedTool(db, logger))

	/* Quantization tools */
	registry.Register(NewVectorQuantizationTool(db, logger))
	registry.Register(NewQuantizationAnalysisTool(db, logger))

	/* Complete embedding tools */
	registry.Register(NewEmbedImageTool(db, logger))
	registry.Register(NewEmbedMultimodalTool(db, logger))
	registry.Register(NewEmbedCachedTool(db, logger))
	registry.Register(NewConfigureEmbeddingModelTool(db, logger))
	registry.Register(NewGetEmbeddingModelConfigTool(db, logger))
	registry.Register(NewListEmbeddingModelConfigsTool(db, logger))
	registry.Register(NewDeleteEmbeddingModelConfigTool(db, logger))

	/* Quality metrics, drift detection, topic discovery */
	registry.Register(NewQualityMetricsTool(db, logger))
	registry.Register(NewDriftDetectionTool(db, logger))
	registry.Register(NewTopicDiscoveryTool(db, logger))

	/* Time series, AutoML, ONNX */
	registry.Register(NewTimeSeriesTool(db, logger))
	registry.Register(NewAutoMLTool(db, logger))
	registry.Register(NewONNXTool(db, logger))

	/* Vector graph operations */
	registry.Register(NewVectorGraphTool(db, logger))

	/* Vecmap operations */
	registry.Register(NewVecmapOperationsTool(db, logger))

	/* Dataset loading */
	registry.Register(NewDatasetLoadingTool(db, logger))

	/* Workers and GPU */
	registry.Register(NewWorkerManagementTool(db, logger))
	registry.Register(NewGPUMonitoringTool(db, logger))

	/* AI Intelligence Layer tools */
	registry.Register(NewAIModelOrchestrationTool(db, logger))
	registry.Register(NewAICostTrackingTool(db, logger))
	registry.Register(NewAIEmbeddingQualityTool(db, logger))
	registry.Register(NewAIModelComparisonTool(db, logger))
	registry.Register(NewAIRAGEvaluationTool(db, logger))
	registry.Register(NewAIEmbeddingDriftDetectionTool(db, logger))
	registry.Register(NewAIModelFinetuningTool(db, logger))
	registry.Register(NewAIPromptVersioningTool(db, logger))
	registry.Register(NewAITokenOptimizationTool(db, logger))
	registry.Register(NewAIMultiModelEnsembleTool(db, logger))

	/* PostgreSQL Optimization tools */
	registry.Register(NewPostgreSQLQueryOptimizerTool(db, logger))
	registry.Register(NewPostgreSQLPerformanceInsightsTool(db, logger))
	registry.Register(NewPostgreSQLIndexAdvisorTool(db, logger))
	registry.Register(NewPostgreSQLQueryPlanAnalyzerTool(db, logger))
	registry.Register(NewPostgreSQLSchemaEvolutionTool(db, logger))
	registry.Register(NewPostgreSQLMigrationTool(db, logger))
	registry.Register(NewPostgreSQLConnectionPoolOptimizerTool(db, logger))
	registry.Register(NewPostgreSQLVacuumAnalyzerTool(db, logger))
	registry.Register(NewPostgreSQLReplicationLagMonitorTool(db, logger))
	registry.Register(NewPostgreSQLWaitEventAnalyzerTool(db, logger))

	/* Developer Experience tools */
	registry.Register(NewNLToSQLTool(db, logger))
	registry.Register(NewSQLToNLTool(db, logger))
	registry.Register(NewQueryBuilderTool(db, logger))
	registry.Register(NewCodeGeneratorTool(db, logger))
	registry.Register(NewTestDataGeneratorTool(db, logger))
	registry.Register(NewSchemaVisualizerTool(db, logger))
	registry.Register(NewQueryExplainerTool(db, logger))
	registry.Register(NewSchemaDocumentationTool(db, logger))
	registry.Register(NewMigrationGeneratorTool(db, logger))

	/* Enterprise Features tools */
	registry.Register(NewMultiTenantManagementTool(db, logger))
	registry.Register(NewDataGovernanceTool(db, logger))
	registry.Register(NewDataLineageTool(db, logger))
	registry.Register(NewComplianceReporterTool(db, logger))
	registry.Register(NewAuditAnalyzerTool(db, logger))
	registry.Register(NewBackupAutomationTool(db, logger))

	/* Performance & Scalability tools */
	registry.Register(NewQueryResultCacheTool(db, logger))
	registry.Register(NewCacheOptimizerTool(db, logger))
	registry.Register(NewPerformanceBenchmarkTool(db, logger))
	registry.Register(NewAutoScalingAdvisorTool(db, logger))
	registry.Register(NewSlowQueryAnalyzerTool(db, logger))

	/* Integration Ecosystem tools */
	registry.Register(NewSDKGeneratorTool(db, logger, registry))

	/* Monitoring & Analytics tools */
	registry.Register(NewRealTimeDashboardTool(db, logger))
	registry.Register(NewAnomalyDetectionTool(db, logger))
	registry.Register(NewPredictiveAnalyticsTool(db, logger))
	registry.Register(NewCostForecastingTool(db, logger))
	registry.Register(NewUsageAnalyticsTool(db, logger))
	registry.Register(NewAlertManagerTool(db, logger))

	/* Plugin Enhancement tools */
	registry.Register(NewPluginMarketplaceTool(db, logger))
	registry.Register(NewPluginHotReloadTool(db, logger))
	registry.Register(NewPluginVersioningTool(db, logger))
	registry.Register(NewPluginSandboxTool(db, logger))
	registry.Register(NewPluginTestingTool(db, logger))
	registry.Register(NewPluginBuilderTool(db, logger))

	/* PostgreSQL tools - Server Information (8 tools) */
	registry.Register(NewPostgreSQLVersionTool(db, logger))
	registry.Register(NewPostgreSQLStatsTool(db, logger))
	registry.Register(NewPostgreSQLDatabaseListTool(db, logger))
	registry.Register(NewPostgreSQLConnectionsTool(db, logger))
	registry.Register(NewPostgreSQLLocksTool(db, logger))
	registry.Register(NewPostgreSQLReplicationTool(db, logger))
	registry.Register(NewPostgreSQLSettingsTool(db, logger))
	registry.Register(NewPostgreSQLExtensionsTool(db, logger))

	/* PostgreSQL tools - Database Object Management (8 tools) */
	registry.Register(NewPostgreSQLTablesTool(db, logger))
	registry.Register(NewPostgreSQLIndexesTool(db, logger))
	registry.Register(NewPostgreSQLSchemasTool(db, logger))
	registry.Register(NewPostgreSQLViewsTool(db, logger))
	registry.Register(NewPostgreSQLSequencesTool(db, logger))
	registry.Register(NewPostgreSQLFunctionsTool(db, logger))
	registry.Register(NewPostgreSQLTriggersTool(db, logger))
	registry.Register(NewPostgreSQLConstraintsTool(db, logger))

	/* PostgreSQL tools - User and Role Management (3 tools) */
	registry.Register(NewPostgreSQLUsersTool(db, logger))
	registry.Register(NewPostgreSQLRolesTool(db, logger))
	registry.Register(NewPostgreSQLPermissionsTool(db, logger))

	/* PostgreSQL tools - Performance and Statistics (4 tools) */
	registry.Register(NewPostgreSQLTableStatsTool(db, logger))
	registry.Register(NewPostgreSQLIndexStatsTool(db, logger))
	registry.Register(NewPostgreSQLActiveQueriesTool(db, logger))
	registry.Register(NewPostgreSQLWaitEventsTool(db, logger))

	/* PostgreSQL tools - Size and Storage (4 tools) */
	registry.Register(NewPostgreSQLTableSizeTool(db, logger))
	registry.Register(NewPostgreSQLIndexSizeTool(db, logger))
	registry.Register(NewPostgreSQLBloatTool(db, logger))
	registry.Register(NewPostgreSQLVacuumStatsTool(db, logger))

	/* PostgreSQL tools - Administration (16 tools) */
	registry.Register(NewPostgreSQLExplainTool(db, logger))
	registry.Register(NewPostgreSQLExplainAnalyzeTool(db, logger))
	registry.Register(NewPostgreSQLVacuumTool(db, logger))
	registry.Register(NewPostgreSQLAnalyzeTool(db, logger))
	registry.Register(NewPostgreSQLReindexTool(db, logger))
	registry.Register(NewPostgreSQLTransactionsTool(db, logger))
	registry.Register(NewPostgreSQLTerminateQueryTool(db, logger))
	registry.Register(NewPostgreSQLSetConfigTool(db, logger))
	registry.Register(NewPostgreSQLReloadConfigTool(db, logger))
	registry.Register(NewPostgreSQLSlowQueriesTool(db, logger))
	registry.Register(NewPostgreSQLCacheHitRatioTool(db, logger))
	registry.Register(NewPostgreSQLBufferStatsTool(db, logger))
	registry.Register(NewPostgreSQLPartitionsTool(db, logger))
	registry.Register(NewPostgreSQLPartitionStatsTool(db, logger))
	registry.Register(NewPostgreSQLFDWServersTool(db, logger))
	registry.Register(NewPostgreSQLFDWTablesTool(db, logger))
	registry.Register(NewPostgreSQLLogicalReplicationSlotsTool(db, logger))

	/* PostgreSQL tools - Query Execution & Management (6 tools) */
	registry.Register(NewPostgreSQLExecuteQueryTool(db, logger))
	registry.Register(NewPostgreSQLQueryPlanTool(db, logger))
	registry.Register(NewPostgreSQLCancelQueryTool(db, logger))
	registry.Register(NewPostgreSQLKillQueryTool(db, logger))
	registry.Register(NewPostgreSQLQueryHistoryTool(db, logger))
	registry.Register(NewPostgreSQLQueryOptimizationTool(db, logger))

	/* PostgreSQL tools - Database & Schema Management (6 tools) */
	registry.Register(NewPostgreSQLCreateDatabaseTool(db, logger))
	registry.Register(NewPostgreSQLDropDatabaseTool(db, logger))
	registry.Register(NewPostgreSQLAlterDatabaseTool(db, logger))
	registry.Register(NewPostgreSQLCreateSchemaTool(db, logger))
	registry.Register(NewPostgreSQLDropSchemaTool(db, logger))
	registry.Register(NewPostgreSQLAlterSchemaTool(db, logger))

	/* PostgreSQL tools - User & Role Management (6 tools) */
	registry.Register(NewPostgreSQLCreateUserTool(db, logger))
	registry.Register(NewPostgreSQLAlterUserTool(db, logger))
	registry.Register(NewPostgreSQLDropUserTool(db, logger))
	registry.Register(NewPostgreSQLCreateRoleTool(db, logger))
	registry.Register(NewPostgreSQLAlterRoleTool(db, logger))
	registry.Register(NewPostgreSQLDropRoleTool(db, logger))

	/* PostgreSQL tools - Permission Management (4 tools) */
	registry.Register(NewPostgreSQLGrantTool(db, logger))
	registry.Register(NewPostgreSQLRevokeTool(db, logger))
	registry.Register(NewPostgreSQLGrantRoleTool(db, logger))
	registry.Register(NewPostgreSQLRevokeRoleTool(db, logger))

	/* PostgreSQL tools - Backup & Recovery (6 tools) */
	registry.Register(NewPostgreSQLBackupDatabaseTool(db, logger))
	registry.Register(NewPostgreSQLRestoreDatabaseTool(db, logger))
	registry.Register(NewPostgreSQLBackupTableTool(db, logger))
	registry.Register(NewPostgreSQLListBackupsTool(db, logger))
	registry.Register(NewPostgreSQLVerifyBackupTool(db, logger))
	registry.Register(NewPostgreSQLBackupScheduleTool(db, logger))

	/* PostgreSQL tools - Schema Modification (7 tools) */
	registry.Register(NewPostgreSQLCreateTableTool(db, logger))
	registry.Register(NewPostgreSQLAlterTableTool(db, logger))
	registry.Register(NewPostgreSQLDropTableTool(db, logger))
	registry.Register(NewPostgreSQLCreateIndexTool(db, logger))
	registry.Register(NewPostgreSQLCreateViewTool(db, logger))
	registry.Register(NewPostgreSQLCreateFunctionTool(db, logger))
	registry.Register(NewPostgreSQLCreateTriggerTool(db, logger))

	/* PostgreSQL tools - Object Management (17 tools) */
	registry.Register(NewPostgreSQLAlterIndexTool(db, logger))
	registry.Register(NewPostgreSQLDropIndexTool(db, logger))
	registry.Register(NewPostgreSQLAlterViewTool(db, logger))
	registry.Register(NewPostgreSQLDropViewTool(db, logger))
	registry.Register(NewPostgreSQLAlterFunctionTool(db, logger))
	registry.Register(NewPostgreSQLDropFunctionTool(db, logger))
	registry.Register(NewPostgreSQLAlterTriggerTool(db, logger))
	registry.Register(NewPostgreSQLDropTriggerTool(db, logger))
	registry.Register(NewPostgreSQLCreateSequenceTool(db, logger))
	registry.Register(NewPostgreSQLAlterSequenceTool(db, logger))
	registry.Register(NewPostgreSQLDropSequenceTool(db, logger))
	registry.Register(NewPostgreSQLCreateTypeTool(db, logger))
	registry.Register(NewPostgreSQLAlterTypeTool(db, logger))
	registry.Register(NewPostgreSQLDropTypeTool(db, logger))
	registry.Register(NewPostgreSQLCreateDomainTool(db, logger))
	registry.Register(NewPostgreSQLAlterDomainTool(db, logger))
	registry.Register(NewPostgreSQLDropDomainTool(db, logger))

	/* PostgreSQL tools - Data Manipulation (5 tools) */
	registry.Register(NewPostgreSQLInsertTool(db, logger))
	registry.Register(NewPostgreSQLUpdateTool(db, logger))
	registry.Register(NewPostgreSQLDeleteTool(db, logger))
	registry.Register(NewPostgreSQLTruncateTool(db, logger))
	registry.Register(NewPostgreSQLCopyTool(db, logger))

	/* PostgreSQL tools - Advanced DDL (10 tools) */
	registry.Register(NewPostgreSQLCreateMaterializedViewTool(db, logger))
	registry.Register(NewPostgreSQLRefreshMaterializedViewTool(db, logger))
	registry.Register(NewPostgreSQLDropMaterializedViewTool(db, logger))
	registry.Register(NewPostgreSQLCreatePartitionTool(db, logger))
	registry.Register(NewPostgreSQLAttachPartitionTool(db, logger))
	registry.Register(NewPostgreSQLDetachPartitionTool(db, logger))
	registry.Register(NewPostgreSQLCreateForeignTableTool(db, logger))
	registry.Register(NewPostgreSQLDropPartitionTool(db, logger))
	registry.Register(NewPostgreSQLDropForeignTableTool(db, logger))
	registry.Register(NewPostgreSQLAlterTableAdvancedTool(db, logger))

	/* PostgreSQL tools - High Availability (5 tools) */
	registry.Register(NewPostgreSQLReplicationLagTool(db, logger))
	registry.Register(NewPostgreSQLPromoteReplicaTool(db, logger))
	registry.Register(NewPostgreSQLSyncStatusTool(db, logger))
	registry.Register(NewPostgreSQLClusterTool(db, logger))
	registry.Register(NewPostgreSQLFailoverTool(db, logger))

	/* PostgreSQL tools - Security & Compliance (7 tools) */
	registry.Register(NewPostgreSQLAuditLogTool(db, logger))
	registry.Register(NewPostgreSQLSecurityScanTool(db, logger))
	registry.Register(NewPostgreSQLComplianceCheckTool(db, logger))
	registry.Register(NewPostgreSQLEncryptionStatusTool(db, logger))
	registry.Register(NewPostgreSQLValidateSQLTool(db, logger))
	registry.Register(NewPostgreSQLCheckPermissionsTool(db, logger))
	registry.Register(NewPostgreSQLAuditOperationTool(db, logger))

	/* PostgreSQL tools - Maintenance Operations (1 tool) */
	registry.Register(NewPostgreSQLMaintenanceWindowTool(db, logger))

	/* Advanced Vector Operations (10 tools) */
	registry.Register(NewVectorAggregateTool(db, logger))
	registry.Register(NewVectorNormalizeBatchTool(db, logger))
	registry.Register(NewVectorSimilarityMatrixTool(db, logger))
	registry.Register(NewVectorBatchDistanceTool(db, logger))
	registry.Register(NewVectorIndexStatisticsTool(db, logger))
	registry.Register(NewVectorDimensionReductionTool(db, logger))
	registry.Register(NewVectorClusterAnalysisTool(db, logger))
	registry.Register(NewVectorAnomalyDetectionTool(db, logger))
	registry.Register(NewVectorQuantizationAdvancedTool(db, logger))
	registry.Register(NewVectorCacheManagementTool(db, logger))

	/* Advanced ML Features (8 tools) */
	registry.Register(NewMLModelVersioningTool(db, logger))
	registry.Register(NewMLModelABTestingTool(db, logger))
	registry.Register(NewMLModelExplainabilityTool(db, logger))
	registry.Register(NewMLModelMonitoringTool(db, logger))
	registry.Register(NewMLModelRollbackTool(db, logger))
	registry.Register(NewMLModelRetrainingTool(db, logger))
	registry.Register(NewMLEnsembleModelsTool(db, logger))
	registry.Register(NewMLModelExportFormatsTool(db, logger))

	/* Advanced Graph Operations (6 tools) */
	registry.Register(NewVectorGraphShortestPathTool(db, logger))
	registry.Register(NewVectorGraphCentralityTool(db, logger))
	registry.Register(NewVectorGraphAnalysisTool(db, logger))
	registry.Register(NewVectorGraphCommunityDetectionAdvancedTool(db, logger))
	registry.Register(NewVectorGraphClusteringTool(db, logger))
	registry.Register(NewVectorGraphVisualizationTool(db, logger))

	/* Multi-Modal Operations (5 tools) */
	registry.Register(NewMultimodalEmbedTool(db, logger))
	registry.Register(NewMultimodalSearchTool(db, logger))
	registry.Register(NewMultimodalRetrievalTool(db, logger))
	registry.Register(NewImageEmbedBatchTool(db, logger))
	registry.Register(NewAudioEmbedTool(db, logger))
}

/* RegisterEssentialTools registers only the most essential tools (default for Claude Desktop compatibility) */
func RegisterEssentialTools(registry *ToolRegistry, db *database.Database, logger *logging.Logger) {
	/* Essential PostgreSQL tools */
	registry.Register(NewPostgreSQLVersionTool(db, logger))
	registry.Register(NewPostgreSQLExecuteQueryTool(db, logger))
	registry.Register(NewPostgreSQLTablesTool(db, logger))

	/* Essential Vector tools */
	registry.Register(NewGenerateEmbeddingTool(db, logger))
	registry.Register(NewVectorSearchTool(db, logger))

	/* Essential RAG tools */
	registry.Register(NewRetrieveContextTool(db, logger))
}

/* RegisterPostgreSQLOnlyTools registers ALL PostgreSQL tools only (no neurondb_ prefix) */
func RegisterPostgreSQLOnlyTools(registry *ToolRegistry, db *database.Database, logger *logging.Logger) {

	/* PostgreSQL tools - Server Information (8 tools) */
	registry.Register(NewPostgreSQLVersionTool(db, logger))
	registry.Register(NewPostgreSQLStatsTool(db, logger))
	registry.Register(NewPostgreSQLDatabaseListTool(db, logger))
	registry.Register(NewPostgreSQLConnectionsTool(db, logger))
	registry.Register(NewPostgreSQLLocksTool(db, logger))
	registry.Register(NewPostgreSQLReplicationTool(db, logger))
	registry.Register(NewPostgreSQLSettingsTool(db, logger))
	registry.Register(NewPostgreSQLExtensionsTool(db, logger))

	/* PostgreSQL tools - Database Object Management (8 tools) */
	registry.Register(NewPostgreSQLTablesTool(db, logger))
	registry.Register(NewPostgreSQLIndexesTool(db, logger))
	registry.Register(NewPostgreSQLSchemasTool(db, logger))
	registry.Register(NewPostgreSQLViewsTool(db, logger))
	registry.Register(NewPostgreSQLSequencesTool(db, logger))
	registry.Register(NewPostgreSQLFunctionsTool(db, logger))
	registry.Register(NewPostgreSQLTriggersTool(db, logger))
	registry.Register(NewPostgreSQLConstraintsTool(db, logger))

	/* PostgreSQL tools - User and Role Management (3 tools) */
	registry.Register(NewPostgreSQLUsersTool(db, logger))
	registry.Register(NewPostgreSQLRolesTool(db, logger))
	registry.Register(NewPostgreSQLPermissionsTool(db, logger))

	/* PostgreSQL tools - Performance and Statistics (4 tools) */
	registry.Register(NewPostgreSQLTableStatsTool(db, logger))
	registry.Register(NewPostgreSQLIndexStatsTool(db, logger))
	registry.Register(NewPostgreSQLActiveQueriesTool(db, logger))
	registry.Register(NewPostgreSQLWaitEventsTool(db, logger))

	/* PostgreSQL tools - Size and Storage (4 tools) */
	registry.Register(NewPostgreSQLTableSizeTool(db, logger))
	registry.Register(NewPostgreSQLIndexSizeTool(db, logger))
	registry.Register(NewPostgreSQLBloatTool(db, logger))
	registry.Register(NewPostgreSQLVacuumStatsTool(db, logger))

	/* PostgreSQL tools - Administration (16 tools) */
	registry.Register(NewPostgreSQLExplainTool(db, logger))
	registry.Register(NewPostgreSQLExplainAnalyzeTool(db, logger))
	registry.Register(NewPostgreSQLVacuumTool(db, logger))
	registry.Register(NewPostgreSQLAnalyzeTool(db, logger))
	registry.Register(NewPostgreSQLReindexTool(db, logger))
	registry.Register(NewPostgreSQLTransactionsTool(db, logger))
	registry.Register(NewPostgreSQLTerminateQueryTool(db, logger))
	registry.Register(NewPostgreSQLSetConfigTool(db, logger))
	registry.Register(NewPostgreSQLReloadConfigTool(db, logger))
	registry.Register(NewPostgreSQLSlowQueriesTool(db, logger))
	registry.Register(NewPostgreSQLCacheHitRatioTool(db, logger))
	registry.Register(NewPostgreSQLBufferStatsTool(db, logger))
	registry.Register(NewPostgreSQLPartitionsTool(db, logger))
	registry.Register(NewPostgreSQLPartitionStatsTool(db, logger))
	registry.Register(NewPostgreSQLFDWServersTool(db, logger))
	registry.Register(NewPostgreSQLFDWTablesTool(db, logger))
	registry.Register(NewPostgreSQLLogicalReplicationSlotsTool(db, logger))

	/* PostgreSQL tools - Query Execution & Management (6 tools) */
	registry.Register(NewPostgreSQLExecuteQueryTool(db, logger))
	registry.Register(NewPostgreSQLQueryPlanTool(db, logger))
	registry.Register(NewPostgreSQLCancelQueryTool(db, logger))
	registry.Register(NewPostgreSQLKillQueryTool(db, logger))
	registry.Register(NewPostgreSQLQueryHistoryTool(db, logger))
	registry.Register(NewPostgreSQLQueryOptimizationTool(db, logger))

	/* PostgreSQL tools - Database & Schema Management (6 tools) */
	registry.Register(NewPostgreSQLCreateDatabaseTool(db, logger))
	registry.Register(NewPostgreSQLDropDatabaseTool(db, logger))
	registry.Register(NewPostgreSQLAlterDatabaseTool(db, logger))
	registry.Register(NewPostgreSQLCreateSchemaTool(db, logger))
	registry.Register(NewPostgreSQLDropSchemaTool(db, logger))
	registry.Register(NewPostgreSQLAlterSchemaTool(db, logger))

	/* PostgreSQL tools - User & Role Management (6 tools) */
	registry.Register(NewPostgreSQLCreateUserTool(db, logger))
	registry.Register(NewPostgreSQLAlterUserTool(db, logger))
	registry.Register(NewPostgreSQLDropUserTool(db, logger))
	registry.Register(NewPostgreSQLCreateRoleTool(db, logger))
	registry.Register(NewPostgreSQLAlterRoleTool(db, logger))
	registry.Register(NewPostgreSQLDropRoleTool(db, logger))

	/* PostgreSQL tools - Permission Management (4 tools) */
	registry.Register(NewPostgreSQLGrantTool(db, logger))
	registry.Register(NewPostgreSQLRevokeTool(db, logger))
	registry.Register(NewPostgreSQLGrantRoleTool(db, logger))
	registry.Register(NewPostgreSQLRevokeRoleTool(db, logger))

	/* PostgreSQL tools - Backup & Recovery (6 tools) */
	registry.Register(NewPostgreSQLBackupDatabaseTool(db, logger))
	registry.Register(NewPostgreSQLRestoreDatabaseTool(db, logger))
	registry.Register(NewPostgreSQLBackupTableTool(db, logger))
	registry.Register(NewPostgreSQLListBackupsTool(db, logger))
	registry.Register(NewPostgreSQLVerifyBackupTool(db, logger))
	registry.Register(NewPostgreSQLBackupScheduleTool(db, logger))

	/* PostgreSQL tools - Schema Modification (7 tools) */
	registry.Register(NewPostgreSQLCreateTableTool(db, logger))
	registry.Register(NewPostgreSQLAlterTableTool(db, logger))
	registry.Register(NewPostgreSQLDropTableTool(db, logger))
	registry.Register(NewPostgreSQLCreateIndexTool(db, logger))
	registry.Register(NewPostgreSQLCreateViewTool(db, logger))
	registry.Register(NewPostgreSQLCreateFunctionTool(db, logger))
	registry.Register(NewPostgreSQLCreateTriggerTool(db, logger))

	/* PostgreSQL tools - Object Management (17 tools) */
	registry.Register(NewPostgreSQLAlterIndexTool(db, logger))
	registry.Register(NewPostgreSQLDropIndexTool(db, logger))
	registry.Register(NewPostgreSQLAlterViewTool(db, logger))
	registry.Register(NewPostgreSQLDropViewTool(db, logger))
	registry.Register(NewPostgreSQLAlterFunctionTool(db, logger))
	registry.Register(NewPostgreSQLDropFunctionTool(db, logger))
	registry.Register(NewPostgreSQLAlterTriggerTool(db, logger))
	registry.Register(NewPostgreSQLDropTriggerTool(db, logger))
	registry.Register(NewPostgreSQLCreateSequenceTool(db, logger))
	registry.Register(NewPostgreSQLAlterSequenceTool(db, logger))
	registry.Register(NewPostgreSQLDropSequenceTool(db, logger))
	registry.Register(NewPostgreSQLCreateTypeTool(db, logger))
	registry.Register(NewPostgreSQLAlterTypeTool(db, logger))
	registry.Register(NewPostgreSQLDropTypeTool(db, logger))
	registry.Register(NewPostgreSQLCreateDomainTool(db, logger))
	registry.Register(NewPostgreSQLAlterDomainTool(db, logger))
	registry.Register(NewPostgreSQLDropDomainTool(db, logger))

	/* PostgreSQL tools - Data Manipulation (5 tools) */
	registry.Register(NewPostgreSQLInsertTool(db, logger))
	registry.Register(NewPostgreSQLUpdateTool(db, logger))
	registry.Register(NewPostgreSQLDeleteTool(db, logger))
	registry.Register(NewPostgreSQLTruncateTool(db, logger))
	registry.Register(NewPostgreSQLCopyTool(db, logger))

	/* PostgreSQL tools - Advanced DDL (10 tools) */
	registry.Register(NewPostgreSQLCreateMaterializedViewTool(db, logger))
	registry.Register(NewPostgreSQLRefreshMaterializedViewTool(db, logger))
	registry.Register(NewPostgreSQLDropMaterializedViewTool(db, logger))
	registry.Register(NewPostgreSQLCreatePartitionTool(db, logger))
	registry.Register(NewPostgreSQLAttachPartitionTool(db, logger))
	registry.Register(NewPostgreSQLDetachPartitionTool(db, logger))
	registry.Register(NewPostgreSQLCreateForeignTableTool(db, logger))
	registry.Register(NewPostgreSQLDropPartitionTool(db, logger))
	registry.Register(NewPostgreSQLDropForeignTableTool(db, logger))
	registry.Register(NewPostgreSQLAlterTableAdvancedTool(db, logger))

	/* PostgreSQL tools - High Availability (5 tools) */
	registry.Register(NewPostgreSQLReplicationLagTool(db, logger))
	registry.Register(NewPostgreSQLPromoteReplicaTool(db, logger))
	registry.Register(NewPostgreSQLSyncStatusTool(db, logger))
	registry.Register(NewPostgreSQLClusterTool(db, logger))
	registry.Register(NewPostgreSQLFailoverTool(db, logger))

	/* PostgreSQL tools - Security & Compliance (7 tools) */
	registry.Register(NewPostgreSQLAuditLogTool(db, logger))
	registry.Register(NewPostgreSQLSecurityScanTool(db, logger))
	registry.Register(NewPostgreSQLComplianceCheckTool(db, logger))
	registry.Register(NewPostgreSQLEncryptionStatusTool(db, logger))
	registry.Register(NewPostgreSQLValidateSQLTool(db, logger))
	registry.Register(NewPostgreSQLCheckPermissionsTool(db, logger))
	registry.Register(NewPostgreSQLAuditOperationTool(db, logger))

	/* PostgreSQL tools - Maintenance Operations (1 tool) */
	registry.Register(NewPostgreSQLMaintenanceWindowTool(db, logger))

	/* TEST: Add ONE neurondb_ tool to see if it causes issues */
	registry.Register(NewVectorSearchTool(db, logger))
}

/* RegisterMinimalTools registers 5 essential PostgreSQL tools only (no neurondb_ prefix) */
func RegisterMinimalTools(registry *ToolRegistry, db *database.Database, logger *logging.Logger) {
	/* Only PostgreSQL tools - no neurondb_ prefix to test Claude Desktop compatibility */
	registry.Register(NewPostgreSQLVersionTool(db, logger))
	registry.Register(NewPostgreSQLExecuteQueryTool(db, logger))
	registry.Register(NewPostgreSQLTablesTool(db, logger))
	registry.Register(NewPostgreSQLQueryPlanTool(db, logger))
	registry.Register(NewPostgreSQLCancelQueryTool(db, logger))
}

/* RegisterToolsByCategory registers tools based on category selection */
func RegisterToolsByCategory(registry *ToolRegistry, db *database.Database, logger *logging.Logger, categories string) {
	categoryList := strings.Split(categories, ",")
	categoryMap := make(map[string]bool)
	for _, cat := range categoryList {
		categoryMap[strings.TrimSpace(strings.ToLower(cat))] = true
	}

	/* Always register basic tools */
	RegisterBasicTools(registry, db, logger)

	/* Register PostgreSQL tools */
	if categoryMap["postgresql"] || categoryMap["all"] {
		RegisterPostgreSQLTools(registry, db, logger)
	}

	/* Register Vector tools */
	if categoryMap["vector"] || categoryMap["all"] {
		RegisterVectorTools(registry, db, logger)
	}

	/* Register ML tools */
	if categoryMap["ml"] || categoryMap["all"] {
		RegisterMLTools(registry, db, logger)
	}

	/* Register RAG tools */
	if categoryMap["rag"] || categoryMap["all"] {
		RegisterRAGTools(registry, db, logger)
	}
}

/* RegisterBasicTools registers essential tools that are always available */
func RegisterBasicTools(registry *ToolRegistry, db *database.Database, logger *logging.Logger) {
	/* Basic PostgreSQL tools */
	registry.Register(NewPostgreSQLVersionTool(db, logger))
	registry.Register(NewPostgreSQLExecuteQueryTool(db, logger))
	registry.Register(NewPostgreSQLTablesTool(db, logger))

	/* Basic Vector tools */
	registry.Register(NewGenerateEmbeddingTool(db, logger))
	registry.Register(NewVectorSearchTool(db, logger))
}

/* RegisterPostgreSQLTools registers ALL PostgreSQL-related tools */
func RegisterPostgreSQLTools(registry *ToolRegistry, db *database.Database, logger *logging.Logger) {
	/* PostgreSQL tools - Server Information (8 tools) */
	registry.Register(NewPostgreSQLVersionTool(db, logger))
	registry.Register(NewPostgreSQLStatsTool(db, logger))
	registry.Register(NewPostgreSQLDatabaseListTool(db, logger))
	registry.Register(NewPostgreSQLConnectionsTool(db, logger))
	registry.Register(NewPostgreSQLLocksTool(db, logger))
	registry.Register(NewPostgreSQLReplicationTool(db, logger))
	registry.Register(NewPostgreSQLSettingsTool(db, logger))
	registry.Register(NewPostgreSQLExtensionsTool(db, logger))

	/* PostgreSQL tools - Database Object Management (8 tools) */
	registry.Register(NewPostgreSQLTablesTool(db, logger))
	registry.Register(NewPostgreSQLIndexesTool(db, logger))
	registry.Register(NewPostgreSQLSchemasTool(db, logger))
	registry.Register(NewPostgreSQLViewsTool(db, logger))
	registry.Register(NewPostgreSQLSequencesTool(db, logger))
	registry.Register(NewPostgreSQLFunctionsTool(db, logger))
	registry.Register(NewPostgreSQLTriggersTool(db, logger))
	registry.Register(NewPostgreSQLConstraintsTool(db, logger))

	/* PostgreSQL tools - User and Role Management (3 tools) */
	registry.Register(NewPostgreSQLUsersTool(db, logger))
	registry.Register(NewPostgreSQLRolesTool(db, logger))
	registry.Register(NewPostgreSQLPermissionsTool(db, logger))

	/* PostgreSQL tools - Performance and Statistics (4 tools) */
	registry.Register(NewPostgreSQLTableStatsTool(db, logger))
	registry.Register(NewPostgreSQLIndexStatsTool(db, logger))
	registry.Register(NewPostgreSQLActiveQueriesTool(db, logger))
	registry.Register(NewPostgreSQLWaitEventsTool(db, logger))

	/* PostgreSQL tools - Size and Storage (4 tools) */
	registry.Register(NewPostgreSQLTableSizeTool(db, logger))
	registry.Register(NewPostgreSQLIndexSizeTool(db, logger))
	registry.Register(NewPostgreSQLBloatTool(db, logger))
	registry.Register(NewPostgreSQLVacuumStatsTool(db, logger))

	/* PostgreSQL tools - Administration (16 tools) */
	registry.Register(NewPostgreSQLExplainTool(db, logger))
	registry.Register(NewPostgreSQLExplainAnalyzeTool(db, logger))
	registry.Register(NewPostgreSQLVacuumTool(db, logger))
	registry.Register(NewPostgreSQLAnalyzeTool(db, logger))
	registry.Register(NewPostgreSQLReindexTool(db, logger))
	registry.Register(NewPostgreSQLTransactionsTool(db, logger))
	registry.Register(NewPostgreSQLTerminateQueryTool(db, logger))
	registry.Register(NewPostgreSQLSetConfigTool(db, logger))
	registry.Register(NewPostgreSQLReloadConfigTool(db, logger))
	registry.Register(NewPostgreSQLSlowQueriesTool(db, logger))
	registry.Register(NewPostgreSQLCacheHitRatioTool(db, logger))
	registry.Register(NewPostgreSQLBufferStatsTool(db, logger))
	registry.Register(NewPostgreSQLPartitionsTool(db, logger))
	registry.Register(NewPostgreSQLPartitionStatsTool(db, logger))
	registry.Register(NewPostgreSQLFDWServersTool(db, logger))
	registry.Register(NewPostgreSQLFDWTablesTool(db, logger))
	registry.Register(NewPostgreSQLLogicalReplicationSlotsTool(db, logger))

	/* PostgreSQL tools - Query Execution & Management (6 tools) */
	registry.Register(NewPostgreSQLExecuteQueryTool(db, logger))
	registry.Register(NewPostgreSQLQueryPlanTool(db, logger))
	registry.Register(NewPostgreSQLCancelQueryTool(db, logger))
	registry.Register(NewPostgreSQLKillQueryTool(db, logger))
	registry.Register(NewPostgreSQLQueryHistoryTool(db, logger))
	registry.Register(NewPostgreSQLQueryOptimizationTool(db, logger))

	/* PostgreSQL tools - Database & Schema Management (6 tools) */
	registry.Register(NewPostgreSQLCreateDatabaseTool(db, logger))
	registry.Register(NewPostgreSQLDropDatabaseTool(db, logger))
	registry.Register(NewPostgreSQLAlterDatabaseTool(db, logger))
	registry.Register(NewPostgreSQLCreateSchemaTool(db, logger))
	registry.Register(NewPostgreSQLDropSchemaTool(db, logger))
	registry.Register(NewPostgreSQLAlterSchemaTool(db, logger))

	/* PostgreSQL tools - User & Role Management (6 tools) */
	registry.Register(NewPostgreSQLCreateUserTool(db, logger))
	registry.Register(NewPostgreSQLAlterUserTool(db, logger))
	registry.Register(NewPostgreSQLDropUserTool(db, logger))
	registry.Register(NewPostgreSQLCreateRoleTool(db, logger))
	registry.Register(NewPostgreSQLAlterRoleTool(db, logger))
	registry.Register(NewPostgreSQLDropRoleTool(db, logger))

	/* PostgreSQL tools - Permission Management (4 tools) */
	registry.Register(NewPostgreSQLGrantTool(db, logger))
	registry.Register(NewPostgreSQLRevokeTool(db, logger))
	registry.Register(NewPostgreSQLGrantRoleTool(db, logger))
	registry.Register(NewPostgreSQLRevokeRoleTool(db, logger))

	/* PostgreSQL tools - Backup & Recovery (6 tools) */
	registry.Register(NewPostgreSQLBackupDatabaseTool(db, logger))
	registry.Register(NewPostgreSQLRestoreDatabaseTool(db, logger))
	registry.Register(NewPostgreSQLBackupTableTool(db, logger))
	registry.Register(NewPostgreSQLListBackupsTool(db, logger))
	registry.Register(NewPostgreSQLVerifyBackupTool(db, logger))
	registry.Register(NewPostgreSQLBackupScheduleTool(db, logger))

	/* PostgreSQL tools - Schema Modification (7 tools) */
	registry.Register(NewPostgreSQLCreateTableTool(db, logger))
	registry.Register(NewPostgreSQLAlterTableTool(db, logger))
	registry.Register(NewPostgreSQLDropTableTool(db, logger))
	registry.Register(NewPostgreSQLCreateIndexTool(db, logger))
	registry.Register(NewPostgreSQLCreateViewTool(db, logger))
	registry.Register(NewPostgreSQLCreateFunctionTool(db, logger))
	registry.Register(NewPostgreSQLCreateTriggerTool(db, logger))

	/* PostgreSQL tools - Object Management (17 tools) */
	registry.Register(NewPostgreSQLAlterIndexTool(db, logger))
	registry.Register(NewPostgreSQLDropIndexTool(db, logger))
	registry.Register(NewPostgreSQLAlterViewTool(db, logger))
	registry.Register(NewPostgreSQLDropViewTool(db, logger))
	registry.Register(NewPostgreSQLAlterFunctionTool(db, logger))
	registry.Register(NewPostgreSQLDropFunctionTool(db, logger))
	registry.Register(NewPostgreSQLAlterTriggerTool(db, logger))
	registry.Register(NewPostgreSQLDropTriggerTool(db, logger))
	registry.Register(NewPostgreSQLCreateSequenceTool(db, logger))
	registry.Register(NewPostgreSQLAlterSequenceTool(db, logger))
	registry.Register(NewPostgreSQLDropSequenceTool(db, logger))
	registry.Register(NewPostgreSQLCreateTypeTool(db, logger))
	registry.Register(NewPostgreSQLAlterTypeTool(db, logger))
	registry.Register(NewPostgreSQLDropTypeTool(db, logger))
	registry.Register(NewPostgreSQLCreateDomainTool(db, logger))
	registry.Register(NewPostgreSQLAlterDomainTool(db, logger))
	registry.Register(NewPostgreSQLDropDomainTool(db, logger))

	/* PostgreSQL tools - Data Manipulation (5 tools) */
	registry.Register(NewPostgreSQLInsertTool(db, logger))
	registry.Register(NewPostgreSQLUpdateTool(db, logger))
	registry.Register(NewPostgreSQLDeleteTool(db, logger))
	registry.Register(NewPostgreSQLTruncateTool(db, logger))
	registry.Register(NewPostgreSQLCopyTool(db, logger))

	/* PostgreSQL tools - Advanced DDL (10 tools) */
	registry.Register(NewPostgreSQLCreateMaterializedViewTool(db, logger))
	registry.Register(NewPostgreSQLRefreshMaterializedViewTool(db, logger))
	registry.Register(NewPostgreSQLDropMaterializedViewTool(db, logger))
	registry.Register(NewPostgreSQLCreatePartitionTool(db, logger))
	registry.Register(NewPostgreSQLAttachPartitionTool(db, logger))
	registry.Register(NewPostgreSQLDetachPartitionTool(db, logger))
	registry.Register(NewPostgreSQLCreateForeignTableTool(db, logger))
	registry.Register(NewPostgreSQLDropPartitionTool(db, logger))
	registry.Register(NewPostgreSQLDropForeignTableTool(db, logger))
	registry.Register(NewPostgreSQLAlterTableAdvancedTool(db, logger))

	/* PostgreSQL tools - High Availability (5 tools) */
	registry.Register(NewPostgreSQLReplicationLagTool(db, logger))
	registry.Register(NewPostgreSQLPromoteReplicaTool(db, logger))
	registry.Register(NewPostgreSQLSyncStatusTool(db, logger))
	registry.Register(NewPostgreSQLClusterTool(db, logger))
	registry.Register(NewPostgreSQLFailoverTool(db, logger))

	/* PostgreSQL tools - Security (7 tools) */
	registry.Register(NewPostgreSQLAuditLogTool(db, logger))
	registry.Register(NewPostgreSQLSecurityScanTool(db, logger))
	registry.Register(NewPostgreSQLComplianceCheckTool(db, logger))
	registry.Register(NewPostgreSQLEncryptionStatusTool(db, logger))
	registry.Register(NewPostgreSQLValidateSQLTool(db, logger))
	registry.Register(NewPostgreSQLCheckPermissionsTool(db, logger))
	registry.Register(NewPostgreSQLAuditOperationTool(db, logger))

	/* PostgreSQL tools - Maintenance (1 tool) */
	registry.Register(NewPostgreSQLMaintenanceWindowTool(db, logger))

	/* NeuronDB Vector Search Tools */
	registry.Register(NewVectorSearchTool(db, logger))
	registry.Register(NewVectorSearchL2Tool(db, logger))
	registry.Register(NewVectorSearchCosineTool(db, logger))
	registry.Register(NewVectorSearchInnerProductTool(db, logger))
	registry.Register(NewVectorSearchL1Tool(db, logger))
	registry.Register(NewVectorSearchHammingTool(db, logger))
	registry.Register(NewVectorSearchChebyshevTool(db, logger))
	registry.Register(NewVectorSearchMinkowskiTool(db, logger))

	/* NeuronDB Embedding Tools */
	registry.Register(NewGenerateEmbeddingTool(db, logger))
	registry.Register(NewBatchEmbeddingTool(db, logger))

	/* NeuronDB Additional Vector Tools */
	registry.Register(NewVectorSimilarityTool(db, logger))
	registry.Register(NewCreateVectorIndexTool(db, logger))

	/* NeuronDB ML Tools */
	registry.Register(NewTrainModelTool(db, logger))
	registry.Register(NewPredictTool(db, logger))
	registry.Register(NewEvaluateModelTool(db, logger))
	registry.Register(NewListModelsTool(db, logger))
	registry.Register(NewGetModelInfoTool(db, logger))
	registry.Register(NewDeleteModelTool(db, logger))

	/* NeuronDB Analytics Tools */
	registry.Register(NewClusterDataTool(db, logger))
	registry.Register(NewDetectOutliersTool(db, logger))
	registry.Register(NewReduceDimensionalityTool(db, logger))

	/* NeuronDB RAG Tools */
	registry.Register(NewProcessDocumentTool(db, logger))
	registry.Register(NewRetrieveContextTool(db, logger))
	registry.Register(NewGenerateResponseTool(db, logger))

	/* NeuronDB Composite RAG Tools */
	registry.Register(NewIngestDocumentsTool(db, logger))
	registry.Register(NewAnswerWithCitationsTool(db, logger))
	registry.Register(NewChunkDocumentTool(db, logger))

	/* NeuronDB Indexing Tools */
	registry.Register(NewCreateHNSWIndexTool(db, logger))
	registry.Register(NewCreateIVFIndexTool(db, logger))
	registry.Register(NewIndexStatusTool(db, logger))
	registry.Register(NewDropIndexTool(db, logger))
	registry.Register(NewTuneHNSWIndexTool(db, logger))
	registry.Register(NewTuneIVFIndexTool(db, logger))

	/* NeuronDB Additional ML Tools */
	registry.Register(NewPredictBatchTool(db, logger))
	registry.Register(NewExportModelTool(db, logger))

	/* NeuronDB Analytics Tools */
	registry.Register(NewAnalyzeDataTool(db, logger))

	/* NeuronDB Hybrid Search Tools */
	registry.Register(NewHybridSearchTool(db, logger))
	registry.Register(NewTextSearchTool(db, logger))
	registry.Register(NewReciprocalRankFusionTool(db, logger))
	registry.Register(NewSemanticKeywordSearchTool(db, logger))
	registry.Register(NewMultiVectorSearchTool(db, logger))
	registry.Register(NewFacetedVectorSearchTool(db, logger))
	registry.Register(NewTemporalVectorSearchTool(db, logger))
	registry.Register(NewDiverseVectorSearchTool(db, logger))

	/* NeuronDB Reranking Tools */
	registry.Register(NewRerankCrossEncoderTool(db, logger))
	registry.Register(NewRerankLLMTool(db, logger))
	registry.Register(NewRerankCohereTool(db, logger))
	registry.Register(NewRerankColBERTTool(db, logger))
	registry.Register(NewRerankLTRTool(db, logger))
	registry.Register(NewRerankEnsembleTool(db, logger))

	/* NeuronDB Advanced Vector Operations */
	registry.Register(NewVectorArithmeticTool(db, logger))
	registry.Register(NewVectorDistanceTool(db, logger))
	registry.Register(NewVectorSimilarityUnifiedTool(db, logger))

	/* NeuronDB Quantization Tools */
	registry.Register(NewVectorQuantizationTool(db, logger))
	registry.Register(NewQuantizationAnalysisTool(db, logger))

	/* NeuronDB Complete Embedding Tools */
	registry.Register(NewEmbedImageTool(db, logger))
	registry.Register(NewEmbedMultimodalTool(db, logger))
	registry.Register(NewEmbedCachedTool(db, logger))
	registry.Register(NewConfigureEmbeddingModelTool(db, logger))
	registry.Register(NewGetEmbeddingModelConfigTool(db, logger))
	registry.Register(NewListEmbeddingModelConfigsTool(db, logger))
	registry.Register(NewDeleteEmbeddingModelConfigTool(db, logger))

	/* NeuronDB Quality Metrics, Drift Detection, Topic Discovery */
	registry.Register(NewQualityMetricsTool(db, logger))
	registry.Register(NewDriftDetectionTool(db, logger))
	registry.Register(NewTopicDiscoveryTool(db, logger))

	/* NeuronDB Time Series, AutoML, ONNX */
	registry.Register(NewTimeSeriesTool(db, logger))
	registry.Register(NewAutoMLTool(db, logger))
}

/* RegisterVectorTools registers vector-related tools */
func RegisterVectorTools(registry *ToolRegistry, db *database.Database, logger *logging.Logger) {
	/* Core Vector tools that definitely exist */
	registry.Register(NewBatchEmbeddingTool(db, logger))
	registry.Register(NewVectorSearchCosineTool(db, logger))
	registry.Register(NewVectorSimilarityTool(db, logger))
	registry.Register(NewCreateVectorIndexTool(db, logger))
	registry.Register(NewIndexStatusTool(db, logger))
}

/* RegisterMLTools registers machine learning tools */
func RegisterMLTools(registry *ToolRegistry, db *database.Database, logger *logging.Logger) {
	/* Basic ML tools that exist */
	registry.Register(NewListModelsTool(db, logger))
	registry.Register(NewGetModelInfoTool(db, logger))
	registry.Register(NewTrainModelTool(db, logger))
}

/* RegisterRAGTools registers RAG (Retrieval-Augmented Generation) tools */
func RegisterRAGTools(registry *ToolRegistry, db *database.Database, logger *logging.Logger) {
	/* Basic RAG tools that exist */
	registry.Register(NewChunkDocumentTool(db, logger))
	registry.Register(NewIngestDocumentsTool(db, logger))
	registry.Register(NewRetrieveContextTool(db, logger))
}
