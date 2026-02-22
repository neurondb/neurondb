/*-------------------------------------------------------------------------
 *
 * modular_rag.go
 *    Modular RAG system with composable, plug-and-play modules
 *
 * Implements a composable RAG architecture where modules can be chained
 * together to create custom workflows.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/modular_rag.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"encoding/json"
	"fmt"
)

/* RAGModule interface defines the contract for RAG modules */
type RAGModule interface {
	/* Name returns the module name */
	Name() string
	
	/* Type returns the module type (retrieval, reranking, generation, filter) */
	Type() ModuleType
	
	/* Execute executes the module with given input */
	Execute(ctx context.Context, input ModuleInput) (ModuleOutput, error)
	
	/* Configure configures the module with parameters */
	Configure(params map[string]interface{}) error
	
	/* Validate validates the module configuration */
	Validate() error
}

/* ModuleType represents the type of RAG module */
type ModuleType string

const (
	ModuleTypeRetrieval  ModuleType = "retrieval"
	ModuleTypeReranking  ModuleType = "reranking"
	ModuleTypeGeneration ModuleType = "generation"
	ModuleTypeFilter     ModuleType = "filter"
	ModuleTypeTransform  ModuleType = "transform"
)

/* ModuleInput represents input to a module */
type ModuleInput struct {
	Query      string
	Documents  []string
	Embeddings [][]float32
	Metadata   map[string]interface{}
	Context    map[string]interface{}
}

/* ModuleOutput represents output from a module */
type ModuleOutput struct {
	Documents  []string
	Embeddings [][]float32
	Metadata   map[string]interface{}
	Scores     []float64
	NextInput  *ModuleInput
}

/* ModularRAGPipeline represents a composable RAG pipeline */
type ModularRAGPipeline struct {
	modules    []RAGModule
	config     PipelineConfig
	registry   *ModuleRegistry
}

/* PipelineConfig represents pipeline configuration */
type PipelineConfig struct {
	Name        string
	Description string
	Modules     []ModuleConfig
	Parameters  map[string]interface{}
}

/* ModuleConfig represents module configuration */
type ModuleConfig struct {
	Name       string
	Type       ModuleType
	Parameters map[string]interface{}
	Enabled    bool
}

/* ModuleRegistry manages available RAG modules */
type ModuleRegistry struct {
	modules map[string]RAGModule
}

/* NewModuleRegistry creates a new module registry */
func NewModuleRegistry() *ModuleRegistry {
	registry := &ModuleRegistry{
		modules: make(map[string]RAGModule),
	}
	
	/* Register built-in modules */
	registry.Register(NewVectorRetrievalModule())
	registry.Register(NewHybridRetrievalModule())
	registry.Register(NewRerankingModule())
	registry.Register(NewGenerationModule())
	registry.Register(NewFilterModule())
	
	return registry
}

/* Register registers a module in the registry */
func (r *ModuleRegistry) Register(module RAGModule) error {
	if module == nil {
		return fmt.Errorf("module registration failed: module_is_nil=true")
	}
	
	if module.Name() == "" {
		return fmt.Errorf("module registration failed: module_name_empty=true")
	}
	
	r.modules[module.Name()] = module
	return nil
}

/* Get retrieves a module by name */
func (r *ModuleRegistry) Get(name string) (RAGModule, error) {
	module, ok := r.modules[name]
	if !ok {
		return nil, fmt.Errorf("module not found: name='%s'", name)
	}
	return module, nil
}

/* List returns all registered modules */
func (r *ModuleRegistry) List() []RAGModule {
	modules := make([]RAGModule, 0, len(r.modules))
	for _, module := range r.modules {
		modules = append(modules, module)
	}
	return modules
}

/* NewModularRAGPipeline creates a new modular RAG pipeline */
func NewModularRAGPipeline(config PipelineConfig, registry *ModuleRegistry) (*ModularRAGPipeline, error) {
	if registry == nil {
		registry = NewModuleRegistry()
	}
	
	pipeline := &ModularRAGPipeline{
		modules:  make([]RAGModule, 0),
		config:   config,
		registry: registry,
	}
	
	/* Build pipeline from configuration */
	for _, moduleConfig := range config.Modules {
		if !moduleConfig.Enabled {
			continue
		}
		
		module, err := registry.Get(moduleConfig.Name)
		if err != nil {
			return nil, fmt.Errorf("pipeline creation failed: module='%s', error=%w", moduleConfig.Name, err)
		}
		
		/* Configure module */
		if err := module.Configure(moduleConfig.Parameters); err != nil {
			return nil, fmt.Errorf("pipeline creation failed: module='%s', configuration_error=true, error=%w", moduleConfig.Name, err)
		}
		
		/* Validate module */
		if err := module.Validate(); err != nil {
			return nil, fmt.Errorf("pipeline creation failed: module='%s', validation_error=true, error=%w", moduleConfig.Name, err)
		}
		
		pipeline.modules = append(pipeline.modules, module)
	}
	
	return pipeline, nil
}

/* Execute executes the modular pipeline */
func (p *ModularRAGPipeline) Execute(ctx context.Context, query string, initialContext map[string]interface{}) (*ModularRAGResult, error) {
	if len(p.modules) == 0 {
		return nil, fmt.Errorf("pipeline execution failed: no_modules_configured=true")
	}
	
	/* Initialize input */
	input := ModuleInput{
		Query:     query,
		Documents: []string{},
		Context:   initialContext,
	}
	
	var output ModuleOutput
	var err error
	
	/* Execute modules in sequence */
	for i, module := range p.modules {
		output, err = module.Execute(ctx, input)
		if err != nil {
			return nil, fmt.Errorf("pipeline execution failed: module='%s', step=%d, error=%w", module.Name(), i+1, err)
		}
		
		/* Prepare input for next module */
		if output.NextInput != nil {
			input = *output.NextInput
		} else {
			input = ModuleInput{
				Query:      input.Query,
				Documents:  output.Documents,
				Embeddings: output.Embeddings,
				Metadata:   output.Metadata,
				Context:    input.Context,
			}
		}
	}
	
	return &ModularRAGResult{
		Query:     query,
		Documents: output.Documents,
		Count:     len(output.Documents),
		Method:    "modular",
		Pipeline:  p.config.Name,
		Metadata:  output.Metadata,
	}, nil
}

/* ModularRAGResult represents the result of modular RAG execution */
type ModularRAGResult struct {
	Query     string
	Documents []string
	Count     int
	Method    string
	Pipeline  string
	Metadata  map[string]interface{}
}

/* modularRAGBackend is the interface used by modules to perform retrieval/rerank/generation. *AdvancedRAG implements it. */
type modularRAGBackend interface {
	VectorRetrieveDocuments(ctx context.Context, query, tableName, vectorCol, textCol string, topK int) ([]string, error)
	HybridRetrieveDocuments(ctx context.Context, query, tableName, vectorCol, textCol string, limit int, vectorWeight float64) ([]string, error)
	RerankDocuments(ctx context.Context, query string, documents []string, model string, topK int) ([]string, error)
	generateAnswer(ctx context.Context, query string, contexts []string) (string, error)
}

/* Built-in module implementations */

/* VectorRetrievalModule implements vector-based retrieval */
type VectorRetrievalModule struct {
	name       string
	topK       int
	threshold  float64
	embedModel string
}

func NewVectorRetrievalModule() *VectorRetrievalModule {
	return &VectorRetrievalModule{
		name:       "vector_retrieval",
		topK:       5,
		threshold:  0.0,
		embedModel: "all-MiniLM-L6-v2",
	}
}

func (m *VectorRetrievalModule) Name() string {
	return m.name
}

func (m *VectorRetrievalModule) Type() ModuleType {
	return ModuleTypeRetrieval
}

func (m *VectorRetrievalModule) Execute(ctx context.Context, input ModuleInput) (ModuleOutput, error) {
	backend, _ := input.Context["_rag_backend"].(modularRAGBackend)
	tableName, _ := input.Context["table_name"].(string)
	vectorCol, _ := input.Context["vector_col"].(string)
	textCol, _ := input.Context["text_col"].(string)
	if backend != nil && tableName != "" && vectorCol != "" {
		docs, err := backend.VectorRetrieveDocuments(ctx, input.Query, tableName, vectorCol, textCol, m.topK)
		if err != nil {
			return ModuleOutput{}, err
		}
		return ModuleOutput{
			Documents: docs,
			Metadata:  map[string]interface{}{"module": m.name},
		}, nil
	}
	return ModuleOutput{
		Documents: []string{},
		Metadata:  map[string]interface{}{"module": m.name},
	}, nil
}

func (m *VectorRetrievalModule) Configure(params map[string]interface{}) error {
	if topK, ok := params["top_k"].(float64); ok {
		m.topK = int(topK)
	}
	if threshold, ok := params["threshold"].(float64); ok {
		m.threshold = threshold
	}
	if embedModel, ok := params["embedding_model"].(string); ok {
		m.embedModel = embedModel
	}
	return nil
}

func (m *VectorRetrievalModule) Validate() error {
	if m.topK <= 0 {
		return fmt.Errorf("validation failed: top_k_must_be_positive=true")
	}
	return nil
}

/* HybridRetrievalModule implements hybrid retrieval */
type HybridRetrievalModule struct {
	name         string
	topK         int
	vectorWeight float64
}

func NewHybridRetrievalModule() *HybridRetrievalModule {
	return &HybridRetrievalModule{
		name:         "hybrid_retrieval",
		topK:         5,
		vectorWeight: 0.7,
	}
}

func (m *HybridRetrievalModule) Name() string {
	return m.name
}

func (m *HybridRetrievalModule) Type() ModuleType {
	return ModuleTypeRetrieval
}

func (m *HybridRetrievalModule) Execute(ctx context.Context, input ModuleInput) (ModuleOutput, error) {
	backend, _ := input.Context["_rag_backend"].(modularRAGBackend)
	tableName, _ := input.Context["table_name"].(string)
	vectorCol, _ := input.Context["vector_col"].(string)
	textCol, _ := input.Context["text_col"].(string)
	if backend != nil && tableName != "" && vectorCol != "" {
		docs, err := backend.HybridRetrieveDocuments(ctx, input.Query, tableName, vectorCol, textCol, m.topK, m.vectorWeight)
		if err != nil {
			return ModuleOutput{}, err
		}
		return ModuleOutput{
			Documents: docs,
			Metadata:  map[string]interface{}{"module": m.name},
		}, nil
	}
	return ModuleOutput{
		Documents: []string{},
		Metadata:  map[string]interface{}{"module": m.name},
	}, nil
}

func (m *HybridRetrievalModule) Configure(params map[string]interface{}) error {
	if topK, ok := params["top_k"].(float64); ok {
		m.topK = int(topK)
	}
	if vectorWeight, ok := params["vector_weight"].(float64); ok {
		m.vectorWeight = vectorWeight
	}
	return nil
}

func (m *HybridRetrievalModule) Validate() error {
	if m.vectorWeight < 0 || m.vectorWeight > 1 {
		return fmt.Errorf("validation failed: vector_weight_must_be_0_to_1=true")
	}
	return nil
}

/* RerankingModule implements reranking */
type RerankingModule struct {
	name  string
	topK  int
	model string
}

func NewRerankingModule() *RerankingModule {
	return &RerankingModule{
		name:  "reranking",
		topK:  3,
		model: "cross-encoder",
	}
}

func (m *RerankingModule) Name() string {
	return m.name
}

func (m *RerankingModule) Type() ModuleType {
	return ModuleTypeReranking
}

func (m *RerankingModule) Execute(ctx context.Context, input ModuleInput) (ModuleOutput, error) {
	backend, _ := input.Context["_rag_backend"].(modularRAGBackend)
	if backend != nil && len(input.Documents) > 0 {
		docs, err := backend.RerankDocuments(ctx, input.Query, input.Documents, m.model, m.topK)
		if err != nil {
			return ModuleOutput{}, err
		}
		return ModuleOutput{
			Documents: docs,
			Metadata:  map[string]interface{}{"module": m.name},
		}, nil
	}
	documents := input.Documents
	if len(documents) > m.topK {
		documents = documents[:m.topK]
	}
	return ModuleOutput{
		Documents: documents,
		Metadata:  map[string]interface{}{"module": m.name},
	}, nil
}

func (m *RerankingModule) Configure(params map[string]interface{}) error {
	if topK, ok := params["top_k"].(float64); ok {
		m.topK = int(topK)
	}
	if model, ok := params["model"].(string); ok {
		m.model = model
	}
	return nil
}

func (m *RerankingModule) Validate() error {
	return nil
}

/* GenerationModule implements answer generation */
type GenerationModule struct {
	name  string
	model string
}

func NewGenerationModule() *GenerationModule {
	return &GenerationModule{
		name:  "generation",
		model: "gpt-4",
	}
}

func (m *GenerationModule) Name() string {
	return m.name
}

func (m *GenerationModule) Type() ModuleType {
	return ModuleTypeGeneration
}

func (m *GenerationModule) Execute(ctx context.Context, input ModuleInput) (ModuleOutput, error) {
	backend, _ := input.Context["_rag_backend"].(modularRAGBackend)
	if backend != nil {
		answer, err := backend.generateAnswer(ctx, input.Query, input.Documents)
		if err != nil {
			return ModuleOutput{}, err
		}
		return ModuleOutput{
			Documents: input.Documents,
			Metadata:  map[string]interface{}{"module": m.name, "answer": answer},
		}, nil
	}
	return ModuleOutput{
		Documents: input.Documents,
		Metadata:  map[string]interface{}{"module": m.name, "answer": "Generated answer"},
	}, nil
}

func (m *GenerationModule) Configure(params map[string]interface{}) error {
	if model, ok := params["model"].(string); ok {
		m.model = model
	}
	return nil
}

func (m *GenerationModule) Validate() error {
	return nil
}

/* FilterModule implements document filtering */
type FilterModule struct {
	name      string
	minScore  float64
	maxDocs   int
	metadata  map[string]interface{}
}

func NewFilterModule() *FilterModule {
	return &FilterModule{
		name:     "filter",
		minScore: 0.5,
		maxDocs:  10,
		metadata: make(map[string]interface{}),
	}
}

func (m *FilterModule) Name() string {
	return m.name
}

func (m *FilterModule) Type() ModuleType {
	return ModuleTypeFilter
}

func (m *FilterModule) Execute(ctx context.Context, input ModuleInput) (ModuleOutput, error) {
	documents := input.Documents
	if len(documents) > m.maxDocs {
		documents = documents[:m.maxDocs]
	}
	return ModuleOutput{
		Documents: documents,
		Metadata:  map[string]interface{}{"module": m.name},
	}, nil
}

func (m *FilterModule) Configure(params map[string]interface{}) error {
	if minScore, ok := params["min_score"].(float64); ok {
		m.minScore = minScore
	}
	if maxDocs, ok := params["max_docs"].(float64); ok {
		m.maxDocs = int(maxDocs)
	}
	if metadata, ok := params["metadata"].(map[string]interface{}); ok {
		m.metadata = metadata
	}
	return nil
}

func (m *FilterModule) Validate() error {
	return nil
}

/* ModularRAG performs RAG using a modular pipeline */
func (r *AdvancedRAG) ModularRAG(ctx context.Context, query, tableName, vectorCol, textCol string, moduleConfigJSON string) (*ModularRAGResult, error) {
	/* Parse module configuration */
	var config PipelineConfig
	if err := json.Unmarshal([]byte(moduleConfigJSON), &config); err != nil {
		return nil, fmt.Errorf("modular RAG failed: config_parse_error=true, error=%w", err)
	}
	
	/* Create registry and pipeline */
	registry := NewModuleRegistry()
	pipeline, err := NewModularRAGPipeline(config, registry)
	if err != nil {
		return nil, fmt.Errorf("modular RAG failed: pipeline_creation_error=true, error=%w", err)
	}
	
	/* Execute pipeline with backend so modules can perform real retrieval/rerank/generation */
	initialContext := map[string]interface{}{
		"table_name":   tableName,
		"vector_col":   vectorCol,
		"text_col":     textCol,
		"_rag_backend": r,
	}
	
	result, err := pipeline.Execute(ctx, query, initialContext)
	if err != nil {
		return nil, fmt.Errorf("modular RAG failed: pipeline_execution_error=true, error=%w", err)
	}
	
	return result, nil
}
