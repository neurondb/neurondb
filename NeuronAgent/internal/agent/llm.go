package agent

import (
	"context"
	"io"

	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

type LLMClient struct {
	llmClient   *neurondb.LLMClient
	embedClient *neurondb.EmbeddingClient
}

func NewLLMClient(db *db.DB) *LLMClient {
	return &LLMClient{
		llmClient:   neurondb.NewLLMClient(db.DB),
		embedClient: neurondb.NewEmbeddingClient(db.DB),
	}
}

func (c *LLMClient) Generate(ctx context.Context, modelName string, prompt string, config map[string]interface{}) (*LLMResponse, error) {
	llmConfig := neurondb.LLMConfig{
		Model: modelName,
	}

	// Extract config values
	if temp, ok := config["temperature"].(float64); ok {
		llmConfig.Temperature = &temp
	}
	if maxTokens, ok := config["max_tokens"].(float64); ok {
		maxTokensInt := int(maxTokens)
		llmConfig.MaxTokens = &maxTokensInt
	}
	if topP, ok := config["top_p"].(float64); ok {
		llmConfig.TopP = &topP
	}

	result, err := c.llmClient.Generate(ctx, prompt, llmConfig)
	
	// Record metrics
	status := "success"
	if err != nil {
		status = "error"
	}
	metrics.RecordLLMCall(modelName, status, result.TokensUsed, 0) // Completion tokens not available
	
	if err != nil {
		return nil, err
	}

	// Estimate completion tokens if not provided
	completionTokens := EstimateTokens(result.Output)
	promptTokens := EstimateTokens(prompt)
	if result.TokensUsed == 0 {
		result.TokensUsed = promptTokens + completionTokens
	}

	return &LLMResponse{
		Content:   result.Output,
		ToolCalls: []ToolCall{}, // Will be parsed separately
		Usage: TokenUsage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      result.TokensUsed,
		},
	}, nil
}

func (c *LLMClient) GenerateStream(ctx context.Context, modelName string, prompt string, config map[string]interface{}, writer io.Writer) error {
	llmConfig := neurondb.LLMConfig{
		Model:  modelName,
		Stream: true,
	}

	// Extract config values
	if temp, ok := config["temperature"].(float64); ok {
		llmConfig.Temperature = &temp
	}
	if maxTokens, ok := config["max_tokens"].(float64); ok {
		maxTokensInt := int(maxTokens)
		llmConfig.MaxTokens = &maxTokensInt
	}
	if topP, ok := config["top_p"].(float64); ok {
		llmConfig.TopP = &topP
	}

	return c.llmClient.GenerateStream(ctx, prompt, llmConfig, writer)
}

func (c *LLMClient) Embed(ctx context.Context, model string, text string) ([]float32, error) {
	embedding, err := c.embedClient.Embed(ctx, text, model)
	if err != nil {
		return nil, err
	}
	return embedding, nil
}

