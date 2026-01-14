/*-------------------------------------------------------------------------
 *
 * sampling.go
 *    Sampling engine for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/sampling/sampling.go
 *
 *-------------------------------------------------------------------------
 */

package sampling

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/tools"
)

const (
	DefaultMaxIterations     = 5
	DefaultLLMTimeout        = 120 * time.Second
	DefaultMaxRetries        = 3
	DefaultRetryBaseDelay    = 1 * time.Second
	MaxTemperature           = 2.0
	MinTemperature           = 0.0
	MaxMaxTokens             = 100000
	MinMaxTokens             = 1
	MaxTopP                  = 1.0
	MinTopP                  = 0.0
	MaxMessageCount          = 100
)

/* Message represents a chat message */
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

/* SamplingRequest represents a sampling request */
type SamplingRequest struct {
	Messages    []Message              `json:"messages"`
	Model       string                 `json:"model,omitempty"`
	Temperature *float64               `json:"temperature,omitempty"`
	MaxTokens   *int                   `json:"maxTokens,omitempty"`
	TopP        *float64               `json:"topP,omitempty"`
	Stream      bool                   `json:"stream,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

/* SamplingResponse represents a sampling response */
type SamplingResponse struct {
	Content   string                 `json:"content"`
	Model     string                 `json:"model"`
	StopReason string                `json:"stopReason,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

/* Manager manages sampling operations */
type Manager struct {
	db           *database.Database
	logger       *logging.Logger
	llmClient    *LLMClient
	toolRegistry *tools.ToolRegistry
}

/* NewManager creates a new sampling manager */
func NewManager(db *database.Database, logger *logging.Logger) *Manager {
	return &Manager{
		db:        db,
		logger:    logger,
		llmClient: NewLLMClient(),
	}
}

/* SetToolRegistry sets the tool registry for tool-aware sampling */
func (m *Manager) SetToolRegistry(registry *tools.ToolRegistry) {
	m.toolRegistry = registry
}

/* CreateMessage creates a completion from messages with tool calling support */
func (m *Manager) CreateMessage(ctx context.Context, req SamplingRequest) (*SamplingResponse, error) {
	/* Nil checks */
	if m == nil {
		return nil, fmt.Errorf("sampling manager instance is nil")
	}
	if ctx == nil {
		return nil, fmt.Errorf("sampling: context cannot be nil")
	}
	if m.db == nil {
		return nil, fmt.Errorf("sampling: database instance is not initialized")
	}
	if m.logger == nil {
		return nil, fmt.Errorf("sampling: logger is not initialized")
	}

	/* Validate messages */
	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("sampling: messages array cannot be empty")
	}

	if len(req.Messages) > MaxMessageCount {
		return nil, fmt.Errorf("sampling: messages array exceeds maximum size of %d: received %d messages", MaxMessageCount, len(req.Messages))
	}

	/* Validate individual messages */
	for i, msg := range req.Messages {
		if msg.Role == "" {
			return nil, fmt.Errorf("sampling: message at index %d has empty role: message_count=%d", i, len(req.Messages))
		}
		if msg.Content == "" {
			return nil, fmt.Errorf("sampling: message at index %d has empty content: message_count=%d, role='%s'", i, len(req.Messages), msg.Role)
		}
		/* Validate message content length */
		if len(msg.Content) > 10*1024*1024 { /* 10MB max */
			return nil, fmt.Errorf("sampling: message at index %d content too long: content_length=%d, max_length=%d", i, len(msg.Content), 10*1024*1024)
		}
	}

	/* Determine model to use */
	modelName := req.Model
	if modelName == "" {
		return nil, fmt.Errorf("sampling: model name is required and cannot be empty")
	}
	if len(modelName) > 200 {
		return nil, fmt.Errorf("sampling: model name too long: model_name_length=%d, max_length=200, model_name='%s'", len(modelName), modelName[:min(50, len(modelName))])
	}

	/* Validate and build LLM parameters */
	temperature := 0.7
	if req.Temperature != nil {
		temp := *req.Temperature
		if temp < MinTemperature || temp > MaxTemperature {
			return nil, fmt.Errorf("sampling: temperature out of range: temperature=%f, min=%f, max=%f", temp, MinTemperature, MaxTemperature)
		}
		temperature = temp
	}

	/* Increase default max_tokens to give model more room for complete responses */
	/* Tool calls need space for JSON formatting */
	/* Set to 8192 to allow for large prompts + complete tool call responses */
	maxTokens := 8192
	if req.MaxTokens != nil {
		tokens := *req.MaxTokens
		if tokens < MinMaxTokens || tokens > MaxMaxTokens {
			return nil, fmt.Errorf("sampling: maxTokens out of range: max_tokens=%d, min=%d, max=%d", tokens, MinMaxTokens, MaxMaxTokens)
		}
		maxTokens = tokens
	}

	/* Validate topP if provided */
	if req.TopP != nil {
		topP := *req.TopP
		if topP < MinTopP || topP > MaxTopP {
			return nil, fmt.Errorf("sampling: topP out of range: top_p=%f, min=%f, max=%f", topP, MinTopP, MaxTopP)
		}
	}

	/* Add timeout context */
	llmCtx, cancel := context.WithTimeout(ctx, DefaultLLMTimeout)
	defer cancel()

	/* Start agent loop with tool support */
	maxIterations := DefaultMaxIterations
	messages := make([]Message, len(req.Messages))
	copy(messages, req.Messages)

	for iteration := 0; iteration < maxIterations; iteration++ {
		/* Check context timeout */
		if llmCtx.Err() != nil {
			return nil, fmt.Errorf("sampling: timeout after %v: model='%s', iteration=%d/%d, error=%w", DefaultLLMTimeout, modelName, iteration+1, maxIterations, llmCtx.Err())
		}

		if m.logger != nil {
			m.logger.Debug(fmt.Sprintf("Agent iteration %d/%d", iteration+1, maxIterations), map[string]interface{}{
				"message_count": len(messages),
				"model":         modelName,
			})
		}

		/* Build prompt with tool awareness */
		prompt := m.buildPromptWithTools(messages)

		/* Call LLM with retry logic */
		var completion string
		var err error
		for retry := 0; retry < DefaultMaxRetries; retry++ {
			completion, err = m.callLLM(llmCtx, prompt, temperature, maxTokens)
			if err == nil {
				break
			}

			/* Check if context was cancelled */
			if llmCtx.Err() != nil {
				return nil, fmt.Errorf("sampling: LLM call cancelled: model='%s', iteration=%d/%d, retry=%d/%d, error=%w", modelName, iteration+1, maxIterations, retry+1, DefaultMaxRetries, llmCtx.Err())
			}

			/* Exponential backoff */
			if retry < DefaultMaxRetries-1 {
				delay := DefaultRetryBaseDelay * time.Duration(1<<uint(retry))
				if m.logger != nil {
					m.logger.Warn("LLM call failed, retrying", map[string]interface{}{
						"model":      modelName,
						"iteration":  iteration + 1,
						"retry":      retry + 1,
						"max_retries": DefaultMaxRetries,
						"delay":      delay,
						"error":      err.Error(),
					})
				}
				time.Sleep(delay)
			}
		}

		if err != nil {
			return nil, fmt.Errorf("sampling: failed to call LLM after %d retries: model='%s', iteration=%d/%d, error=%w", DefaultMaxRetries, modelName, iteration+1, maxIterations, err)
		}

		/* Check if LLM wants to call a tool */
		toolCall, err := m.parseToolCall(completion)
		if err != nil {
			/* Not a tool call, return as final response */
			if m.logger != nil {
				m.logger.Debug("No tool call detected, returning response", map[string]interface{}{
					"model":     modelName,
					"iteration": iteration + 1,
				})
			}
			return &SamplingResponse{
				Content:    completion,
				Model:      modelName,
				StopReason: "stop",
				Metadata:   req.Metadata,
			}, nil
		}

		/* Execute the tool */
		if m.logger != nil {
			m.logger.Info("Executing tool", map[string]interface{}{
				"tool_name": toolCall.Name,
				"arguments": toolCall.Arguments,
				"iteration": iteration + 1,
				"model":     modelName,
			})
		}

		toolResult, err := m.executeTool(llmCtx, toolCall.Name, toolCall.Arguments)
		if err != nil {
			if m.logger != nil {
				m.logger.Error("Tool execution failed", err, map[string]interface{}{
					"tool_name": toolCall.Name,
					"iteration": iteration + 1,
					"model":     modelName,
				})
			}
			/* Return error to user */
			return nil, fmt.Errorf("sampling: tool execution failed: tool='%s', iteration=%d/%d, model='%s', error=%w", toolCall.Name, iteration+1, maxIterations, modelName, err)
		}

		/* Add assistant's tool call and tool result to conversation */
		messages = append(messages, Message{
			Role:    "assistant",
			Content: fmt.Sprintf("I'll use the tool '%s' to help answer your question.", toolCall.Name),
		})

		toolResultStr := fmt.Sprintf("Tool '%s' returned: %v", toolCall.Name, toolResult)
		messages = append(messages, Message{
			Role:    "tool_result",
			Content: toolResultStr,
		})

		/* Continue loop to let LLM process the result */
	}

	/* If we've exhausted iterations, return what we have */
	return &SamplingResponse{
		Content:    "I've attempted to process your request but reached the maximum number of tool calls. Please try rephrasing your question.",
		Model:      modelName,
		StopReason: "max_tokens",
		Metadata:   req.Metadata,
	}, nil
}

/* ModelConfig represents a model configuration */
type ModelConfig struct {
	Provider string
	BaseURL  string
	APIKey   string
}

/* getModelConfig gets model configuration */
func (m *Manager) getModelConfig(ctx context.Context, modelName string) (*ModelConfig, error) {
	if m == nil || m.db == nil {
		/* Fallback if db not available */
		return &ModelConfig{
			Provider: "ollama",
			BaseURL:  "http://localhost:11434",
			APIKey:   "",
		}, nil
	}

	/* Try to get model config from neurondesk database */
	query := `
		SELECT 
			model_provider,
			COALESCE(base_url, '') as base_url,
			COALESCE(api_key, '') as api_key
		FROM model_configs
		WHERE model_name = $1
		LIMIT 1
	`

	var config ModelConfig
	err := m.db.QueryRow(ctx, query, modelName).Scan(
		&config.Provider,
		&config.BaseURL,
		&config.APIKey,
	)

	if err != nil {
		/* Fallback: assume it's Ollama if no config found */
		if m.logger != nil {
			m.logger.Warn("Model config not found, assuming Ollama", map[string]interface{}{
				"model": modelName,
				"error": err.Error(),
			})
		}
		return &ModelConfig{
			Provider: "ollama",
			BaseURL:  "http://localhost:11434",
			APIKey:   "",
		}, nil
	}

	/* Set default base URLs if not configured */
	if config.BaseURL == "" {
		switch config.Provider {
		case "ollama":
			config.BaseURL = "http://localhost:11434"
		case "openai":
			config.BaseURL = "https://api.openai.com/v1"
		case "anthropic":
			config.BaseURL = "https://api.anthropic.com"
		case "google":
			config.BaseURL = "https://generativelanguage.googleapis.com/v1beta"
		}
	}

	return &config, nil
}

/* buildPromptWithTools builds a prompt including available tools */
func (m *Manager) buildPromptWithTools(messages []Message) string {
	var promptBuilder strings.Builder

	/* Add system message with tool information if we have tools */
	if m.toolRegistry != nil {
		toolDefs := m.toolRegistry.GetAllDefinitions()
		if len(toolDefs) > 0 {
			promptBuilder.WriteString("You are an AI assistant with PostgreSQL database tools.\n\n")
			promptBuilder.WriteString("Rules:\n")
			promptBuilder.WriteString("1. For database queries -> use TOOL_CALL: {\"name\": \"tool_name\", \"arguments\": {...}}\n")
			promptBuilder.WriteString("2. For general knowledge -> answer directly\n")
			promptBuilder.WriteString("3. Complete your response - don't truncate JSON\n\n")
			promptBuilder.WriteString("Key tools: postgresql_version, postgresql_stats, postgresql_connections, vector_search, generate_embedding\n")
			promptBuilder.WriteString("Example: TOOL_CALL: {\"name\": \"postgresql_version\", \"arguments\": {}}\n\n")
		}
	}

	/* Add conversation messages */
	for _, msg := range messages {
		switch msg.Role {
		case "user":
			promptBuilder.WriteString("User: ")
			promptBuilder.WriteString(msg.Content)
			promptBuilder.WriteString("\n\n")
		case "assistant":
			promptBuilder.WriteString("Assistant: ")
			promptBuilder.WriteString(msg.Content)
			promptBuilder.WriteString("\n\n")
		case "system":
			promptBuilder.WriteString("System: ")
			promptBuilder.WriteString(msg.Content)
			promptBuilder.WriteString("\n\n")
		case "tool_result":
			promptBuilder.WriteString("Tool Result: ")
			promptBuilder.WriteString(msg.Content)
			promptBuilder.WriteString("\n\n")
		default:
			promptBuilder.WriteString(fmt.Sprintf("%s: %s\n\n", msg.Role, msg.Content))
		}
	}

	/* Add final prompt for assistant response */
	promptBuilder.WriteString("Assistant: ")
	return promptBuilder.String()
}

/* callLLM calls the NeuronDB LLM function */
func (m *Manager) callLLM(ctx context.Context, prompt string, temperature float64, maxTokens int) (string, error) {
	if m == nil || m.db == nil {
		return "", fmt.Errorf("sampling: database instance is not initialized")
	}

	if prompt == "" {
		return "", fmt.Errorf("sampling: prompt cannot be empty")
	}

	llmParamsJSON := fmt.Sprintf(`{"temperature": %.2f, "max_tokens": %d}`, temperature, maxTokens)

	/* Call neurondb.llm() function directly */
	/* neurondb.llm(task, model, input_text, input_array, params, max_length) returns JSONB */
	/* Task 'complete' generates text completion */
	/* Use NULL for model to use the configured default (neurondb.llm_model setting) */
	query := `SELECT (neurondb.llm('complete', NULL, $1, NULL, $2::jsonb, $3)->>'text') AS response`

	var completion string
	err := m.db.QueryRow(ctx, query, prompt, llmParamsJSON, maxTokens).Scan(&completion)
	if err != nil {
		return "", fmt.Errorf("sampling: failed to call neurondb.llm(): prompt_length=%d, temperature=%f, max_tokens=%d, error=%w", len(prompt), temperature, maxTokens, err)
	}

	if completion == "" {
		return "", fmt.Errorf("sampling: empty response from neurondb.llm(): prompt_length=%d, temperature=%f, max_tokens=%d", len(prompt), temperature, maxTokens)
	}

	return completion, nil
}

/* ToolCall represents a parsed tool call from LLM response */
type ToolCall struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

/* parseToolCall attempts to parse a tool call from LLM response */
func (m *Manager) parseToolCall(response string) (*ToolCall, error) {
	/* Look for TOOL_CALL: pattern */
	if !strings.Contains(response, "TOOL_CALL:") {
		return nil, fmt.Errorf("no tool call detected")
	}

	/* Extract JSON after TOOL_CALL: */
	parts := strings.SplitN(response, "TOOL_CALL:", 2)
	if len(parts) < 2 {
		return nil, fmt.Errorf("invalid tool call format")
	}

	jsonStr := strings.TrimSpace(parts[1])

	/* Parse JSON */
	var toolCall ToolCall
	decoder := json.NewDecoder(strings.NewReader(jsonStr))
	if err := decoder.Decode(&toolCall); err != nil {
		return nil, fmt.Errorf("failed to parse tool call JSON: %w", err)
	}

	if toolCall.Name == "" {
		return nil, fmt.Errorf("tool call missing name")
	}

	return &toolCall, nil
}

/* executeTool executes a tool by name with given arguments */
func (m *Manager) executeTool(ctx context.Context, name string, arguments map[string]interface{}) (interface{}, error) {
	if m == nil {
		return nil, fmt.Errorf("sampling: manager instance is nil")
	}
	if m.toolRegistry == nil {
		return nil, fmt.Errorf("sampling: tool registry not configured: tool='%s'", name)
	}

	if name == "" {
		return nil, fmt.Errorf("sampling: tool name cannot be empty")
	}

	if arguments == nil {
		arguments = make(map[string]interface{})
	}

	tool := m.toolRegistry.GetTool(name)
	if tool == nil {
		return nil, fmt.Errorf("sampling: tool not found: tool_name='%s'", name)
	}

	result, err := tool.Execute(ctx, arguments)
	if err != nil {
		return nil, fmt.Errorf("sampling: tool execution error: tool='%s', error=%w", name, err)
	}

	if result == nil {
		return nil, fmt.Errorf("sampling: tool returned nil result: tool='%s'", name)
	}

	if !result.Success {
		if result.Error != nil {
			return nil, fmt.Errorf("sampling: tool returned error: tool='%s', error_code='%s', error_message='%s'", name, result.Error.Code, result.Error.Message)
		}
		return nil, fmt.Errorf("sampling: tool execution failed without error details: tool='%s'", name)
	}

	return result.Data, nil
}