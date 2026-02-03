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
	"context"
	"strings"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/tools/composition"
	"github.com/neurondb/NeuronMCP/internal/tools/debugging"
	"github.com/neurondb/NeuronMCP/internal/tools/workflow"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* compositionRegistryAdapter adapts ToolRegistry to composition.ToolRegistryInterface */
type compositionRegistryAdapter struct {
	registry *ToolRegistry
}

/* GetTool retrieves a tool and adapts it */
func (a *compositionRegistryAdapter) GetTool(name string) composition.ToolInterface {
	if a.registry == nil {
		return nil
	}
	tool := a.registry.GetTool(name)
	if tool == nil {
		return nil
	}
	return &compositionToolAdapter{tool: tool}
}

/* compositionToolAdapter adapts Tool to composition.ToolInterface */
type compositionToolAdapter struct {
	tool Tool
}

/* Execute executes the tool */
func (a *compositionToolAdapter) Execute(ctx context.Context, arguments map[string]interface{}) (*composition.ToolResult, error) {
	result, err := a.tool.Execute(ctx, arguments)
	if err != nil {
		return nil, err
	}

	if result == nil {
		return &composition.ToolResult{Success: false}, nil
	}

	toolResult := &composition.ToolResult{
		Success:  result.Success,
		Data:     result.Data,
		Metadata: result.Metadata,
	}

	if result.Error != nil {
		toolResult.Error = &composition.ToolError{
			Message: result.Error.Message,
			Code:    result.Error.Code,
			Details: result.Error.Details,
		}
	}

	return toolResult, nil
}

/* debuggingToolAdapter wraps debugging tools to implement tools.Tool */
type debuggingToolAdapter struct {
	tool *debugging.DebugToolCallTool
}

func (a *debuggingToolAdapter) Name() string                         { return a.tool.Name() }
func (a *debuggingToolAdapter) Description() string                  { return a.tool.Description() }
func (a *debuggingToolAdapter) InputSchema() map[string]interface{}  { return a.tool.InputSchema() }
func (a *debuggingToolAdapter) OutputSchema() map[string]interface{} { return a.tool.OutputSchema() }
func (a *debuggingToolAdapter) Version() string                      { return a.tool.Version() }
func (a *debuggingToolAdapter) Deprecated() bool                     { return a.tool.Deprecated() }
func (a *debuggingToolAdapter) Deprecation() *mcp.DeprecationInfo    { return a.tool.Deprecation() }
func (a *debuggingToolAdapter) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	result, err := a.tool.Execute(ctx, params)
	if err != nil {
		return nil, err
	}
	return convertDebuggingResult(result), nil
}

func convertDebuggingResult(r *debugging.ToolResult) *ToolResult {
	if r == nil {
		return nil
	}
	result := &ToolResult{
		Success:  r.Success,
		Data:     r.Data,
		Metadata: r.Metadata,
	}
	if r.Error != nil {
		result.Error = &ToolError{
			Message: r.Error.Message,
			Code:    r.Error.Code,
			Details: r.Error.Details,
		}
	}
	return result
}

/* workflowRegistryAdapter adapts ToolRegistry to workflow.ToolRegistryInterface */
type workflowRegistryAdapter struct {
	registry *ToolRegistry
}

/* GetTool retrieves a tool and adapts it */
func (a *workflowRegistryAdapter) GetTool(name string) workflow.ToolInterface {
	if a.registry == nil {
		return nil
	}
	tool := a.registry.GetTool(name)
	if tool == nil {
		return nil
	}
	return &workflowToolAdapter{tool: tool}
}

/* workflowToolAdapter adapts Tool to workflow.ToolInterface */
type workflowToolAdapter struct {
	tool Tool
}

/* Execute executes the tool */
func (a *workflowToolAdapter) Execute(ctx context.Context, arguments map[string]interface{}) (*workflow.ToolResult, error) {
	result, err := a.tool.Execute(ctx, arguments)
	if err != nil {
		return nil, err
	}

	if result == nil {
		return &workflow.ToolResult{Success: false}, nil
	}

	toolResult := &workflow.ToolResult{
		Success: result.Success,
		Data:    result.Data,
	}

	if result.Error != nil {
		toolResult.Error = &workflow.ToolError{
			Message: result.Error.Message,
			Code:    result.Error.Code,
		}
	}

	return toolResult, nil
}

/* debuggingRegistryAdapter adapts ToolRegistry to debugging.ToolRegistryInterface */
type debuggingRegistryAdapter struct {
	registry *ToolRegistry
}

/* GetTool retrieves a tool and adapts it */
func (a *debuggingRegistryAdapter) GetTool(name string) debugging.ToolInterface {
	if a.registry == nil {
		return nil
	}
	tool := a.registry.GetTool(name)
	if tool == nil {
		return nil
	}
	return &debuggingToolAdapterForRegistry{tool: tool}
}

/* debuggingToolAdapterForRegistry adapts Tool to debugging.ToolInterface */
type debuggingToolAdapterForRegistry struct {
	tool Tool
}

/* Execute executes the tool */
func (a *debuggingToolAdapterForRegistry) Execute(ctx context.Context, arguments map[string]interface{}) (*debugging.ToolResult, error) {
	result, err := a.tool.Execute(ctx, arguments)
	if err != nil {
		return nil, err
	}

	if result == nil {
		return &debugging.ToolResult{Success: false}, nil
	}

	toolResult := &debugging.ToolResult{
		Success:  result.Success,
		Data:     result.Data,
		Metadata: result.Metadata,
	}

	if result.Error != nil {
		toolResult.Error = &debugging.ToolError{
			Message: result.Error.Message,
			Code:    result.Error.Code,
			Details: result.Error.Details,
		}
	}

	return toolResult, nil
}

/* debuggingQueryPlanAdapter wraps DebugQueryPlanTool to implement tools.Tool */
type debuggingQueryPlanAdapter struct {
	tool *debugging.DebugQueryPlanTool
}

func (a *debuggingQueryPlanAdapter) Name() string                        { return a.tool.Name() }
func (a *debuggingQueryPlanAdapter) Description() string                 { return a.tool.Description() }
func (a *debuggingQueryPlanAdapter) InputSchema() map[string]interface{} { return a.tool.InputSchema() }
func (a *debuggingQueryPlanAdapter) OutputSchema() map[string]interface{} {
	return a.tool.OutputSchema()
}
func (a *debuggingQueryPlanAdapter) Version() string                   { return a.tool.Version() }
func (a *debuggingQueryPlanAdapter) Deprecated() bool                  { return a.tool.Deprecated() }
func (a *debuggingQueryPlanAdapter) Deprecation() *mcp.DeprecationInfo { return a.tool.Deprecation() }
func (a *debuggingQueryPlanAdapter) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	result, err := a.tool.Execute(ctx, params)
	if err != nil {
		return nil, err
	}
	return convertDebuggingResult(result), nil
}

/* debuggingMonitorConnectionsAdapter wraps MonitorActiveConnectionsTool to implement tools.Tool */
type debuggingMonitorConnectionsAdapter struct {
	tool *debugging.MonitorActiveConnectionsTool
}

func (a *debuggingMonitorConnectionsAdapter) Name() string        { return a.tool.Name() }
func (a *debuggingMonitorConnectionsAdapter) Description() string { return a.tool.Description() }
func (a *debuggingMonitorConnectionsAdapter) InputSchema() map[string]interface{} {
	return a.tool.InputSchema()
}
func (a *debuggingMonitorConnectionsAdapter) OutputSchema() map[string]interface{} {
	return a.tool.OutputSchema()
}
func (a *debuggingMonitorConnectionsAdapter) Version() string  { return a.tool.Version() }
func (a *debuggingMonitorConnectionsAdapter) Deprecated() bool { return a.tool.Deprecated() }
func (a *debuggingMonitorConnectionsAdapter) Deprecation() *mcp.DeprecationInfo {
	return a.tool.Deprecation()
}
func (a *debuggingMonitorConnectionsAdapter) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	result, err := a.tool.Execute(ctx, params)
	if err != nil {
		return nil, err
	}
	return convertDebuggingResult(result), nil
}

/* debuggingMonitorPerformanceAdapter wraps MonitorQueryPerformanceTool to implement tools.Tool */
type debuggingMonitorPerformanceAdapter struct {
	tool *debugging.MonitorQueryPerformanceTool
}

func (a *debuggingMonitorPerformanceAdapter) Name() string        { return a.tool.Name() }
func (a *debuggingMonitorPerformanceAdapter) Description() string { return a.tool.Description() }
func (a *debuggingMonitorPerformanceAdapter) InputSchema() map[string]interface{} {
	return a.tool.InputSchema()
}
func (a *debuggingMonitorPerformanceAdapter) OutputSchema() map[string]interface{} {
	return a.tool.OutputSchema()
}
func (a *debuggingMonitorPerformanceAdapter) Version() string  { return a.tool.Version() }
func (a *debuggingMonitorPerformanceAdapter) Deprecated() bool { return a.tool.Deprecated() }
func (a *debuggingMonitorPerformanceAdapter) Deprecation() *mcp.DeprecationInfo {
	return a.tool.Deprecation()
}
func (a *debuggingMonitorPerformanceAdapter) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	result, err := a.tool.Execute(ctx, params)
	if err != nil {
		return nil, err
	}
	return convertDebuggingResult(result), nil
}

/* debuggingTraceAdapter wraps TraceRequestTool to implement tools.Tool */
type debuggingTraceAdapter struct {
	tool *debugging.TraceRequestTool
}

func (a *debuggingTraceAdapter) Name() string                         { return a.tool.Name() }
func (a *debuggingTraceAdapter) Description() string                  { return a.tool.Description() }
func (a *debuggingTraceAdapter) InputSchema() map[string]interface{}  { return a.tool.InputSchema() }
func (a *debuggingTraceAdapter) OutputSchema() map[string]interface{} { return a.tool.OutputSchema() }
func (a *debuggingTraceAdapter) Version() string                      { return a.tool.Version() }
func (a *debuggingTraceAdapter) Deprecated() bool                     { return a.tool.Deprecated() }
func (a *debuggingTraceAdapter) Deprecation() *mcp.DeprecationInfo    { return a.tool.Deprecation() }
func (a *debuggingTraceAdapter) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	result, err := a.tool.Execute(ctx, params)
	if err != nil {
		return nil, err
	}
	return convertDebuggingResult(result), nil
}

func convertCompositionResult(r *composition.ToolResult) *ToolResult {
	if r == nil {
		return nil
	}
	result := &ToolResult{
		Success:  r.Success,
		Data:     r.Data,
		Metadata: r.Metadata,
	}
	if r.Error != nil {
		result.Error = &ToolError{
			Message: r.Error.Message,
			Code:    r.Error.Code,
			Details: r.Error.Details,
		}
	}
	return result
}

/* compositionToolChainAdapter wraps ToolChainTool */
type compositionToolChainAdapter struct {
	tool *composition.ToolChainTool
}

func (a *compositionToolChainAdapter) Name() string        { return a.tool.Name() }
func (a *compositionToolChainAdapter) Description() string { return a.tool.Description() }
func (a *compositionToolChainAdapter) InputSchema() map[string]interface{} {
	return a.tool.InputSchema()
}
func (a *compositionToolChainAdapter) OutputSchema() map[string]interface{} {
	return a.tool.OutputSchema()
}
func (a *compositionToolChainAdapter) Version() string                   { return a.tool.Version() }
func (a *compositionToolChainAdapter) Deprecated() bool                  { return a.tool.Deprecated() }
func (a *compositionToolChainAdapter) Deprecation() *mcp.DeprecationInfo { return a.tool.Deprecation() }
func (a *compositionToolChainAdapter) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	result, err := a.tool.Execute(ctx, params)
	if err != nil {
		return nil, err
	}
	return convertCompositionResult(result), nil
}

/* compositionToolParallelAdapter wraps ToolParallelTool */
type compositionToolParallelAdapter struct {
	tool *composition.ToolParallelTool
}

func (a *compositionToolParallelAdapter) Name() string        { return a.tool.Name() }
func (a *compositionToolParallelAdapter) Description() string { return a.tool.Description() }
func (a *compositionToolParallelAdapter) InputSchema() map[string]interface{} {
	return a.tool.InputSchema()
}
func (a *compositionToolParallelAdapter) OutputSchema() map[string]interface{} {
	return a.tool.OutputSchema()
}
func (a *compositionToolParallelAdapter) Version() string  { return a.tool.Version() }
func (a *compositionToolParallelAdapter) Deprecated() bool { return a.tool.Deprecated() }
func (a *compositionToolParallelAdapter) Deprecation() *mcp.DeprecationInfo {
	return a.tool.Deprecation()
}
func (a *compositionToolParallelAdapter) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	result, err := a.tool.Execute(ctx, params)
	if err != nil {
		return nil, err
	}
	return convertCompositionResult(result), nil
}

/* compositionToolConditionalAdapter wraps ToolConditionalTool */
type compositionToolConditionalAdapter struct {
	tool *composition.ToolConditionalTool
}

func (a *compositionToolConditionalAdapter) Name() string        { return a.tool.Name() }
func (a *compositionToolConditionalAdapter) Description() string { return a.tool.Description() }
func (a *compositionToolConditionalAdapter) InputSchema() map[string]interface{} {
	return a.tool.InputSchema()
}
func (a *compositionToolConditionalAdapter) OutputSchema() map[string]interface{} {
	return a.tool.OutputSchema()
}
func (a *compositionToolConditionalAdapter) Version() string  { return a.tool.Version() }
func (a *compositionToolConditionalAdapter) Deprecated() bool { return a.tool.Deprecated() }
func (a *compositionToolConditionalAdapter) Deprecation() *mcp.DeprecationInfo {
	return a.tool.Deprecation()
}
func (a *compositionToolConditionalAdapter) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	result, err := a.tool.Execute(ctx, params)
	if err != nil {
		return nil, err
	}
	return convertCompositionResult(result), nil
}

/* compositionToolRetryAdapter wraps ToolRetryTool */
type compositionToolRetryAdapter struct {
	tool *composition.ToolRetryTool
}

func (a *compositionToolRetryAdapter) Name() string        { return a.tool.Name() }
func (a *compositionToolRetryAdapter) Description() string { return a.tool.Description() }
func (a *compositionToolRetryAdapter) InputSchema() map[string]interface{} {
	return a.tool.InputSchema()
}
func (a *compositionToolRetryAdapter) OutputSchema() map[string]interface{} {
	return a.tool.OutputSchema()
}
func (a *compositionToolRetryAdapter) Version() string                   { return a.tool.Version() }
func (a *compositionToolRetryAdapter) Deprecated() bool                  { return a.tool.Deprecated() }
func (a *compositionToolRetryAdapter) Deprecation() *mcp.DeprecationInfo { return a.tool.Deprecation() }
func (a *compositionToolRetryAdapter) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	result, err := a.tool.Execute(ctx, params)
	if err != nil {
		return nil, err
	}
	return convertCompositionResult(result), nil
}

func convertWorkflowResult(r *workflow.ToolResult) *ToolResult {
	if r == nil {
		return nil
	}
	result := &ToolResult{
		Success: r.Success,
		Data:    r.Data,
	}
	if r.Error != nil {
		result.Error = &ToolError{
			Message: r.Error.Message,
			Code:    r.Error.Code,
		}
	}
	return result
}

/* workflowToolCreateAdapter wraps CreateWorkflowTool */
type workflowToolCreateAdapter struct {
	tool *workflow.CreateWorkflowTool
}

func (a *workflowToolCreateAdapter) Name() string                        { return a.tool.Name() }
func (a *workflowToolCreateAdapter) Description() string                 { return a.tool.Description() }
func (a *workflowToolCreateAdapter) InputSchema() map[string]interface{} { return a.tool.InputSchema() }
func (a *workflowToolCreateAdapter) OutputSchema() map[string]interface{} {
	return a.tool.OutputSchema()
}
func (a *workflowToolCreateAdapter) Version() string                   { return a.tool.Version() }
func (a *workflowToolCreateAdapter) Deprecated() bool                  { return a.tool.Deprecated() }
func (a *workflowToolCreateAdapter) Deprecation() *mcp.DeprecationInfo { return a.tool.Deprecation() }
func (a *workflowToolCreateAdapter) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	result, err := a.tool.Execute(ctx, params)
	if err != nil {
		return nil, err
	}
	return convertWorkflowResult(result), nil
}

/* workflowToolExecuteAdapter wraps ExecuteWorkflowTool */
type workflowToolExecuteAdapter struct {
	tool *workflow.ExecuteWorkflowTool
}

func (a *workflowToolExecuteAdapter) Name() string        { return a.tool.Name() }
func (a *workflowToolExecuteAdapter) Description() string { return a.tool.Description() }
func (a *workflowToolExecuteAdapter) InputSchema() map[string]interface{} {
	return a.tool.InputSchema()
}
func (a *workflowToolExecuteAdapter) OutputSchema() map[string]interface{} {
	return a.tool.OutputSchema()
}
func (a *workflowToolExecuteAdapter) Version() string                   { return a.tool.Version() }
func (a *workflowToolExecuteAdapter) Deprecated() bool                  { return a.tool.Deprecated() }
func (a *workflowToolExecuteAdapter) Deprecation() *mcp.DeprecationInfo { return a.tool.Deprecation() }
func (a *workflowToolExecuteAdapter) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	result, err := a.tool.Execute(ctx, params)
	if err != nil {
		return nil, err
	}
	return convertWorkflowResult(result), nil
}

/* workflowToolStatusAdapter wraps WorkflowStatusTool */
type workflowToolStatusAdapter struct {
	tool *workflow.WorkflowStatusTool
}

func (a *workflowToolStatusAdapter) Name() string                        { return a.tool.Name() }
func (a *workflowToolStatusAdapter) Description() string                 { return a.tool.Description() }
func (a *workflowToolStatusAdapter) InputSchema() map[string]interface{} { return a.tool.InputSchema() }
func (a *workflowToolStatusAdapter) OutputSchema() map[string]interface{} {
	return a.tool.OutputSchema()
}
func (a *workflowToolStatusAdapter) Version() string                   { return a.tool.Version() }
func (a *workflowToolStatusAdapter) Deprecated() bool                  { return a.tool.Deprecated() }
func (a *workflowToolStatusAdapter) Deprecation() *mcp.DeprecationInfo { return a.tool.Deprecation() }
func (a *workflowToolStatusAdapter) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	result, err := a.tool.Execute(ctx, params)
	if err != nil {
		return nil, err
	}
	return convertWorkflowResult(result), nil
}

/* workflowToolListAdapter wraps ListWorkflowsTool */
type workflowToolListAdapter struct {
	tool *workflow.ListWorkflowsTool
}

func (a *workflowToolListAdapter) Name() string                         { return a.tool.Name() }
func (a *workflowToolListAdapter) Description() string                  { return a.tool.Description() }
func (a *workflowToolListAdapter) InputSchema() map[string]interface{}  { return a.tool.InputSchema() }
func (a *workflowToolListAdapter) OutputSchema() map[string]interface{} { return a.tool.OutputSchema() }
func (a *workflowToolListAdapter) Version() string                      { return a.tool.Version() }
func (a *workflowToolListAdapter) Deprecated() bool                     { return a.tool.Deprecated() }
func (a *workflowToolListAdapter) Deprecation() *mcp.DeprecationInfo    { return a.tool.Deprecation() }
func (a *workflowToolListAdapter) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	result, err := a.tool.Execute(ctx, params)
	if err != nil {
		return nil, err
	}
	return convertWorkflowResult(result), nil
}

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
	registry.Register(NewRAGEvaluateTool(db, logger))
	registry.Register(NewRAGChatTool(db, logger))
	registry.Register(NewRAGHybridTool(db, logger))
	registry.Register(NewRAGRerankTool(db, logger))
	registry.Register(NewRAGHyDETool(db, logger))
	registry.Register(NewRAGGraphTool(db, logger))
	registry.Register(NewRAGCorrectiveTool(db, logger))
	registry.Register(NewRAGAgenticTool(db, logger))
	registry.Register(NewRAGContextualTool(db, logger))
	registry.Register(NewRAGModularTool(db, logger))

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

	/* Enhanced Debugging & Monitoring tools */
	debuggingAdapter := &debuggingRegistryAdapter{registry: registry}
	registry.Register(&debuggingToolAdapter{tool: debugging.NewDebugToolCallTool(debuggingAdapter, logger)})
	registry.Register(&debuggingQueryPlanAdapter{tool: debugging.NewDebugQueryPlanTool(db, logger)})
	registry.Register(&debuggingMonitorConnectionsAdapter{tool: debugging.NewMonitorActiveConnectionsTool(db, logger)})
	registry.Register(&debuggingMonitorPerformanceAdapter{tool: debugging.NewMonitorQueryPerformanceTool(db, logger)})
	registry.Register(&debuggingTraceAdapter{tool: debugging.NewTraceRequestTool(logger)})

	/* Tool Composition tools */
	/* Create adapter inline to avoid import cycle */
	compositionAdapter := &compositionRegistryAdapter{registry: registry}
	/* Create tools using the New functions - they return pointers to the tool structs */
	chainTool := composition.NewToolChainTool(compositionAdapter, logger)
	registry.Register(&compositionToolChainAdapter{tool: chainTool})
	registry.Register(&compositionToolParallelAdapter{tool: composition.NewToolParallelTool(compositionAdapter, logger)})
	registry.Register(&compositionToolConditionalAdapter{tool: composition.NewToolConditionalTool(compositionAdapter, logger)})
	registry.Register(&compositionToolRetryAdapter{tool: composition.NewToolRetryTool(compositionAdapter, logger)})

	/* Workflow Orchestration tools */
	workflowManager := workflow.NewManager()
	workflowRegistryAdapter := &workflowRegistryAdapter{registry: registry}
	workflowExecutor := workflow.NewExecutor(workflowManager, &workflow.ToolExecutorAdapter{Registry: workflowRegistryAdapter}, logger)
	registry.Register(&workflowToolCreateAdapter{tool: workflow.NewCreateWorkflowTool(workflowManager, logger)})
	registry.Register(&workflowToolExecuteAdapter{tool: workflow.NewExecuteWorkflowTool(workflowManager, workflowExecutor, logger)})
	registry.Register(&workflowToolStatusAdapter{tool: workflow.NewWorkflowStatusTool(workflowManager, logger)})
	registry.Register(&workflowToolListAdapter{tool: workflow.NewListWorkflowsTool(workflowManager, logger)})

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
	registry.Register(NewRAGEvaluateTool(db, logger))
	registry.Register(NewRAGChatTool(db, logger))
	registry.Register(NewRAGHybridTool(db, logger))
	registry.Register(NewRAGRerankTool(db, logger))
	registry.Register(NewRAGHyDETool(db, logger))
	registry.Register(NewRAGGraphTool(db, logger))
	registry.Register(NewRAGCorrectiveTool(db, logger))
	registry.Register(NewRAGAgenticTool(db, logger))
	registry.Register(NewRAGContextualTool(db, logger))
	registry.Register(NewRAGModularTool(db, logger))

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
	/* Composite RAG tools - All 8 RAG architectures */
	/* 1. Naive RAG - Basic retrieval (via RetrieveContextTool) */
	/* 2. HyDE RAG - Hypothetical Document Embeddings */
	registry.Register(NewRAGHyDETool(db, logger))
	/* 3. Graph RAG - Knowledge graph traversal */
	registry.Register(NewRAGGraphTool(db, logger))
	/* 4. Corrective RAG - Iterative self-correction */
	registry.Register(NewRAGCorrectiveTool(db, logger))
	/* 5. Hybrid RAG - Vector + full-text fusion */
	registry.Register(NewRAGHybridTool(db, logger))
	/* 6. Agentic RAG - Autonomous planning */
	registry.Register(NewRAGAgenticTool(db, logger))
	/* 7. Contextual RAG - Context-aware query rewriting */
	registry.Register(NewRAGContextualTool(db, logger))
	/* 8. Modular RAG - Composable modules */
	registry.Register(NewRAGModularTool(db, logger))
	/* Additional RAG tools */
	registry.Register(NewRAGEvaluateTool(db, logger))
	registry.Register(NewRAGChatTool(db, logger))
	registry.Register(NewRAGRerankTool(db, logger))
	registry.Register(NewAnswerWithCitationsTool(db, logger))
}
