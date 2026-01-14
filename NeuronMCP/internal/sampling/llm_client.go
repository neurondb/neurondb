/*-------------------------------------------------------------------------
 *
 * llm_client.go
 *    LLM client for calling various LLM providers (Ollama, OpenAI, etc.)
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/sampling/llm_client.go
 *
 *-------------------------------------------------------------------------
 */

package sampling

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

/* LLMClient handles communication with LLM providers */
type LLMClient struct {
	httpClient *http.Client
}

/* NewLLMClient creates a new LLM client */
func NewLLMClient() *LLMClient {
	return &LLMClient{
		httpClient: &http.Client{
			Timeout: 120 * time.Second, // 2 minutes for LLM responses
		},
	}
}

/* OllamaRequest represents an Ollama API request */
type OllamaRequest struct {
	Model    string                 `json:"model"`
	Prompt   string                 `json:"prompt,omitempty"`
	Messages []Message              `json:"messages,omitempty"`
	Stream   bool                   `json:"stream"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

/* OllamaResponse represents an Ollama API response */
type OllamaResponse struct {
	Model     string                 `json:"model"`
	CreatedAt string                 `json:"created_at"`
	Response  string                 `json:"response"`
	Message   *Message               `json:"message,omitempty"`
	Done      bool                   `json:"done"`
	Context   []int                  `json:"context,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

/* CallOllama calls the Ollama API */
func (c *LLMClient) CallOllama(ctx context.Context, baseURL, model string, messages []Message, options map[string]interface{}) (string, error) {
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}

	/* Use chat API if messages are provided */
	endpoint := fmt.Sprintf("%s/api/chat", baseURL)
	
	req := OllamaRequest{
		Model:    model,
		Messages: messages,
		Stream:   false,
		Options:  options,
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("failed to call Ollama API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("Ollama API error (status %d): %s", resp.StatusCode, string(body))
	}

	var ollamaResp OllamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return "", fmt.Errorf("failed to decode Ollama response: %w", err)
	}

	/* Extract response text */
	if ollamaResp.Message != nil && ollamaResp.Message.Content != "" {
		return ollamaResp.Message.Content, nil
	}
	if ollamaResp.Response != "" {
		return ollamaResp.Response, nil
	}

	return "", fmt.Errorf("empty response from Ollama")
}

/* OpenAIRequest represents an OpenAI-compatible API request */
type OpenAIRequest struct {
	Model       string                 `json:"model"`
	Messages    []Message              `json:"messages"`
	Temperature float64                `json:"temperature,omitempty"`
	MaxTokens   int                    `json:"max_tokens,omitempty"`
	TopP        float64                `json:"top_p,omitempty"`
	Stream      bool                   `json:"stream"`
}

/* OpenAIResponse represents an OpenAI API response */
type OpenAIResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int     `json:"index"`
		Message      Message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

/* CallOpenAI calls the OpenAI API or compatible endpoint */
func (c *LLMClient) CallOpenAI(ctx context.Context, baseURL, apiKey, model string, messages []Message, temperature float64, maxTokens int) (string, error) {
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}

	endpoint := fmt.Sprintf("%s/chat/completions", baseURL)

	req := OpenAIRequest{
		Model:       model,
		Messages:    messages,
		Temperature: temperature,
		MaxTokens:   maxTokens,
		Stream:      false,
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("failed to call OpenAI API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("OpenAI API error (status %d): %s", resp.StatusCode, string(body))
	}

	var openAIResp OpenAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&openAIResp); err != nil {
		return "", fmt.Errorf("failed to decode OpenAI response: %w", err)
	}

	if len(openAIResp.Choices) == 0 {
		return "", fmt.Errorf("no choices in OpenAI response")
	}

	return openAIResp.Choices[0].Message.Content, nil
}








