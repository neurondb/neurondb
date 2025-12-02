package neurondb

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/jmoiron/sqlx"
)

// LLMClient handles LLM generation via NeuronDB
type LLMClient struct {
	db *sqlx.DB
}

// NewLLMClient creates a new LLM client
func NewLLMClient(db *sqlx.DB) *LLMClient {
	return &LLMClient{db: db}
}

// Generate generates text using the LLM with the given prompt and config
func (c *LLMClient) Generate(ctx context.Context, prompt string, config LLMConfig) (*LLMGenerateResult, error) {
	// Build parameters JSON
	params := make(map[string]interface{})
	if config.Temperature != nil {
		params["temperature"] = *config.Temperature
	}
	if config.MaxTokens != nil {
		params["max_tokens"] = *config.MaxTokens
	}
	if config.TopP != nil {
		params["top_p"] = *config.TopP
	}

	paramsJSON, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("LLM generation failed: model_name='%s', prompt_length=%d, parameter_marshaling_error=true, error=%w",
			config.Model, len(prompt), err)
	}

	// Try neurondb_llm_generate first, fallback to neurondb_llm_complete
	var output string
	query := `SELECT neurondb_llm_generate($1, $2, $3::jsonb) AS output`
	
	err = c.db.GetContext(ctx, &output, query, config.Model, prompt, paramsJSON)
	if err != nil {
		// Fallback to neurondb_llm_complete if available
		query = `SELECT neurondb_llm_complete($1, $2, $3::jsonb) AS output`
		err = c.db.GetContext(ctx, &output, query, config.Model, prompt, paramsJSON)
		if err != nil {
			promptTokens := len(strings.Split(prompt, " "))
			temperature := "default"
			if config.Temperature != nil {
				temperature = fmt.Sprintf("%.2f", *config.Temperature)
			}
			maxTokens := "default"
			if config.MaxTokens != nil {
				maxTokens = fmt.Sprintf("%d", *config.MaxTokens)
			}
			topP := "default"
			if config.TopP != nil {
				topP = fmt.Sprintf("%.2f", *config.TopP)
			}
			return nil, fmt.Errorf("LLM generation failed via NeuronDB: model_name='%s', prompt_length=%d, prompt_tokens_approx=%d, temperature=%s, max_tokens=%s, top_p=%s, function='neurondb_llm_generate' (fallback: 'neurondb_llm_complete'), error=%w",
				config.Model, len(prompt), promptTokens, temperature, maxTokens, topP, err)
		}
	}

	return &LLMGenerateResult{
		Output:      output,
		TokensUsed:  0, // Token count would need to be extracted from response
		FinishReason: "stop",
	}, nil
}

// GenerateStream generates text with streaming support
// Uses a cursor-based approach to stream results chunk by chunk
func (c *LLMClient) GenerateStream(ctx context.Context, prompt string, config LLMConfig, writer io.Writer) error {
	// Build parameters JSON
	params := make(map[string]interface{})
	if config.Temperature != nil {
		params["temperature"] = *config.Temperature
	}
	if config.MaxTokens != nil {
		params["max_tokens"] = *config.MaxTokens
	}
	if config.TopP != nil {
		params["top_p"] = *config.TopP
	}
	params["stream"] = true

	paramsJSON, err := json.Marshal(params)
	if err != nil {
		return fmt.Errorf("failed to marshal params: %w", err)
	}

	// Try streaming query - if not supported, fall back to chunked writes
	query := `SELECT neurondb_llm_generate_stream($1, $2, $3::jsonb) AS chunk`
	
	rows, err := c.db.QueryContext(ctx, query, config.Model, prompt, paramsJSON)
	if err != nil {
		// Fallback: generate full response and write in chunks
		result, err := c.Generate(ctx, prompt, config)
		if err != nil {
			promptTokens := len(strings.Split(prompt, " "))
			return fmt.Errorf("LLM streaming generation failed via NeuronDB: model_name='%s', prompt_length=%d, prompt_tokens_approx=%d, function='neurondb_llm_generate_stream', fallback_generation_failed=true, error=%w",
				config.Model, len(prompt), promptTokens, err)
		}
		
		// Write in chunks to simulate streaming
		chunkSize := 100
		output := []byte(result.Output)
		for i := 0; i < len(output); i += chunkSize {
			end := i + chunkSize
			if end > len(output) {
				end = len(output)
			}
			if _, err := writer.Write(output[i:end]); err != nil {
				return err
			}
			// Small delay to simulate streaming
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}
		}
		return nil
	}
	defer rows.Close()

	// Stream chunks from database
		for rows.Next() {
		var chunk string
		if err := rows.Scan(&chunk); err != nil {
			return fmt.Errorf("LLM streaming chunk scan failed via NeuronDB: model_name='%s', prompt_length=%d, function='neurondb_llm_generate_stream', error=%w",
				config.Model, len(prompt), err)
		}
		
		if _, err := writer.Write([]byte(chunk)); err != nil {
			return err
		}
		
		// Check context
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
	}
	
	return rows.Err()
}

// GenerateWithTools generates text with tool calling support
func (c *LLMClient) GenerateWithTools(ctx context.Context, prompt string, config LLMConfig, tools []ToolDefinition) (*LLMGenerateResult, []ToolCall, error) {
	// Add tools to the prompt or use a tool-aware function
	// This is a simplified version - full implementation would handle tool schemas
	result, err := c.Generate(ctx, prompt, config)
	if err != nil {
		return nil, nil, err
	}

	// Parse tool calls from response (would need proper parsing logic)
	toolCalls := parseToolCalls(result.Output)

	return result, toolCalls, nil
}

// ToolDefinition represents a tool that can be called by the LLM
type ToolDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// ToolCall represents a tool call from the LLM
type ToolCall struct {
	ID        string                 `json:"id"`
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

// parseToolCalls parses tool calls from LLM response
// Attempts to extract tool calls from various formats
func parseToolCalls(output string) []ToolCall {
	var toolCalls []ToolCall
	
	// Try to find JSON tool calls in the response
	// Look for patterns like {"tool_calls": [...]} or "function_call": {...}
	
	// Method 1: Look for complete JSON object with tool_calls
	if strings.Contains(output, "tool_calls") {
		// Try to extract JSON object
		start := strings.Index(output, `"tool_calls"`)
		if start != -1 {
			// Find the opening brace before tool_calls
			objStart := strings.LastIndex(output[:start], "{")
			if objStart != -1 {
				// Find matching closing brace
				braceCount := 0
				objEnd := -1
				for i := objStart; i < len(output); i++ {
					if output[i] == '{' {
						braceCount++
					} else if output[i] == '}' {
						braceCount--
						if braceCount == 0 {
							objEnd = i + 1
							break
						}
					}
				}
				
				if objEnd > objStart {
					var data map[string]interface{}
					if err := json.Unmarshal([]byte(output[objStart:objEnd]), &data); err == nil {
						if tc, ok := data["tool_calls"].([]interface{}); ok {
							for i, t := range tc {
								if tcMap, ok := t.(map[string]interface{}); ok {
									call := ToolCall{
										ID: fmt.Sprintf("call_%d", i),
									}
									
									if name, ok := tcMap["name"].(string); ok {
										call.Name = name
									}
									
									if args, ok := tcMap["arguments"]; ok {
										if argsMap, ok := args.(map[string]interface{}); ok {
											call.Arguments = argsMap
										} else if argsStr, ok := args.(string); ok {
											// Try to parse JSON string
											var argsMap map[string]interface{}
											if err := json.Unmarshal([]byte(argsStr), &argsMap); err == nil {
												call.Arguments = argsMap
											}
										}
									}
									
									if call.Name != "" {
										toolCalls = append(toolCalls, call)
									}
								}
							}
						}
					}
				}
			}
		}
	}
	
	// Method 2: Look for function_call pattern
	if len(toolCalls) == 0 && strings.Contains(output, "function_call") {
		// Similar extraction logic for function_call format
		start := strings.Index(output, `"function_call"`)
		if start != -1 {
			objStart := strings.LastIndex(output[:start], "{")
			if objStart != -1 {
				braceCount := 0
				objEnd := -1
				for i := objStart; i < len(output); i++ {
					if output[i] == '{' {
						braceCount++
					} else if output[i] == '}' {
						braceCount--
						if braceCount == 0 {
							objEnd = i + 1
							break
						}
					}
				}
				
				if objEnd > objStart {
					var data map[string]interface{}
					if err := json.Unmarshal([]byte(output[objStart:objEnd]), &data); err == nil {
						if fc, ok := data["function_call"].(map[string]interface{}); ok {
							call := ToolCall{
								ID: "call_0",
							}
							
							if name, ok := fc["name"].(string); ok {
								call.Name = name
							}
							
							if args, ok := fc["arguments"]; ok {
								if argsMap, ok := args.(map[string]interface{}); ok {
									call.Arguments = argsMap
								} else if argsStr, ok := args.(string); ok {
									var argsMap map[string]interface{}
									if err := json.Unmarshal([]byte(argsStr), &argsMap); err == nil {
										call.Arguments = argsMap
									}
								}
							}
							
							if call.Name != "" {
								toolCalls = append(toolCalls, call)
							}
						}
					}
				}
			}
		}
	}
	
	return toolCalls
}

