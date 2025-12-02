package agent

import (
	"context"
	"fmt"
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
		promptTokens := EstimateTokens(prompt)
		temperature := "default"
		if llmConfig.Temperature != nil {
			temperature = fmt.Sprintf("%.2f", *llmConfig.Temperature)
		}
		maxTokens := "default"
		if llmConfig.MaxTokens != nil {
			maxTokens = fmt.Sprintf("%d", *llmConfig.MaxTokens)
		}
		topP := "default"
		if llmConfig.TopP != nil {
			topP = fmt.Sprintf("%.2f", *llmConfig.TopP)
		}
		return nil, fmt.Errorf("LLM generation failed: model_name='%s', prompt_length=%d, prompt_tokens=%d, temperature=%s, max_tokens=%s, top_p=%s, streaming=false, error=%w",
			modelName, len(prompt), promptTokens, temperature, maxTokens, topP, err)
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

	err := c.llmClient.GenerateStream(ctx, prompt, llmConfig, writer)
	if err != nil {
		promptTokens := EstimateTokens(prompt)
		temperature := "default"
		if llmConfig.Temperature != nil {
			temperature = fmt.Sprintf("%.2f", *llmConfig.Temperature)
		}
		maxTokens := "default"
		if llmConfig.MaxTokens != nil {
			maxTokens = fmt.Sprintf("%d", *llmConfig.MaxTokens)
		}
		topP := "default"
		if llmConfig.TopP != nil {
			topP = fmt.Sprintf("%.2f", *llmConfig.TopP)
		}
		return fmt.Errorf("LLM streaming generation failed: model_name='%s', prompt_length=%d, prompt_tokens=%d, temperature=%s, max_tokens=%s, top_p=%s, streaming=true, error=%w",
			modelName, len(prompt), promptTokens, temperature, maxTokens, topP, err)
	}
	return nil
}

func (c *LLMClient) Embed(ctx context.Context, model string, text string) ([]float32, error) {
	embedding, err := c.embedClient.Embed(ctx, text, model)
	if err != nil {
		return nil, fmt.Errorf("embedding generation failed: model_name='%s', text_length=%d, error=%w",
			model, len(text), err)
	}
	if embedding == nil {
		return nil, fmt.Errorf("embedding generation returned nil: model_name='%s', text_length=%d", model, len(text))
	}
	return embedding, nil
}

