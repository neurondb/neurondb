/*-------------------------------------------------------------------------
 *
 * manager.go
 *    Completion manager for NeuronMCP
 *
 * Provides autocomplete functionality for prompt arguments and resource URIs
 * according to the MCP specification.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/completion/manager.go
 *
 *-------------------------------------------------------------------------
 */

package completion

import (
	"context"
	"fmt"
	"strings"

	"github.com/neurondb/NeuronMCP/internal/prompts"
	"github.com/neurondb/NeuronMCP/internal/resources"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* Manager manages completion requests */
type Manager struct {
	promptsManager  *prompts.Manager
	resourcesManager *resources.Manager
}

/* NewManager creates a new completion manager */
func NewManager(promptsManager *prompts.Manager, resourcesManager *resources.Manager) *Manager {
	return &Manager{
		promptsManager:   promptsManager,
		resourcesManager: resourcesManager,
	}
}

/* Complete handles a completion request */
func (m *Manager) Complete(ctx context.Context, req mcp.CompletionRequest) (*mcp.CompletionResponse, error) {
	if m == nil {
		return nil, fmt.Errorf("completion: manager instance is nil")
	}
	if ctx == nil {
		return nil, fmt.Errorf("completion: context cannot be nil")
	}

	/* Validate reference type */
	if req.Ref.Type == "" {
		return nil, fmt.Errorf("completion: reference type is required")
	}

	switch req.Ref.Type {
	case "ref/prompt":
		return m.completePromptArgument(ctx, req)
	case "ref/resource":
		return m.completeResourceURI(ctx, req)
	default:
		return nil, fmt.Errorf("completion: unsupported reference type: %s (supported: ref/prompt, ref/resource)", req.Ref.Type)
	}
}

/* completePromptArgument completes a prompt argument */
func (m *Manager) completePromptArgument(ctx context.Context, req mcp.CompletionRequest) (*mcp.CompletionResponse, error) {
	if m.promptsManager == nil {
		return nil, fmt.Errorf("completion: prompts manager is not initialized")
	}

	/* Validate prompt name */
	if req.Ref.Name == "" {
		return nil, fmt.Errorf("completion: prompt name is required for ref/prompt")
	}

	/* Get the prompt to find its arguments */
	promptList, err := m.promptsManager.ListPrompts(ctx)
	if err != nil {
		return nil, fmt.Errorf("completion: failed to list prompts: %w", err)
	}

	/* Find the specific prompt */
	var targetPrompt *prompts.Prompt
	for i := range promptList {
		if promptList[i].Name == req.Ref.Name {
			targetPrompt = &promptList[i]
			break
		}
	}

	if targetPrompt == nil {
		return nil, fmt.Errorf("completion: prompt not found: %s", req.Ref.Name)
	}

	/* Find the argument */
	var targetArg *prompts.VariableDefinition
	for i := range targetPrompt.Variables {
		if targetPrompt.Variables[i].Name == req.Argument.Name {
			targetArg = &targetPrompt.Variables[i]
			break
		}
	}

	if targetArg == nil {
		return nil, fmt.Errorf("completion: argument not found in prompt: prompt=%s, argument=%s", req.Ref.Name, req.Argument.Name)
	}

	/* Generate completion suggestions based on argument name and partial value */
	values := m.generatePromptArgumentCompletions(targetArg, req.Argument.Value, req.Context)

	return &mcp.CompletionResponse{
		Completion: mcp.CompletionResult{
			Values:  values,
			Total:   len(values),
			HasMore: false, /* For now, we return all matches */
		},
	}, nil
}

/* completeResourceURI completes a resource URI */
func (m *Manager) completeResourceURI(ctx context.Context, req mcp.CompletionRequest) (*mcp.CompletionResponse, error) {
	if m.resourcesManager == nil {
		return nil, fmt.Errorf("completion: resources manager is not initialized")
	}

	/* Get all available resources */
	resourceDefs := m.resourcesManager.ListResources()

	/* Filter resources based on partial URI value */
	var matchingURIs []string
	partialValue := strings.ToLower(req.Argument.Value)

	for _, def := range resourceDefs {
		uri := strings.ToLower(def.URI)
		/* Match if URI starts with partial value or contains it */
		if partialValue == "" || strings.HasPrefix(uri, partialValue) || strings.Contains(uri, partialValue) {
			matchingURIs = append(matchingURIs, def.URI)
		}
	}

	/* Limit results to reasonable number */
	maxResults := 50
	if len(matchingURIs) > maxResults {
		matchingURIs = matchingURIs[:maxResults]
	}

	return &mcp.CompletionResponse{
		Completion: mcp.CompletionResult{
			Values:  matchingURIs,
			Total:   len(matchingURIs),
			HasMore: len(resourceDefs) > maxResults,
		},
	}, nil
}

/* generatePromptArgumentCompletions generates completion suggestions for a prompt argument */
func (m *Manager) generatePromptArgumentCompletions(arg *prompts.VariableDefinition, partialValue string, context map[string]interface{}) []string {
	var suggestions []string

	partialLower := strings.ToLower(partialValue)

	/* Generate suggestions based on argument name patterns */
	argNameLower := strings.ToLower(arg.Name)

	/* Common patterns for different argument types */
	if strings.Contains(argNameLower, "language") || strings.Contains(argNameLower, "lang") {
		/* Programming languages */
		languages := []string{"python", "javascript", "typescript", "go", "rust", "java", "cpp", "c", "sql", "html", "css", "json", "yaml", "markdown"}
		for _, lang := range languages {
			if partialValue == "" || strings.HasPrefix(strings.ToLower(lang), partialLower) {
				suggestions = append(suggestions, lang)
			}
		}
	} else if strings.Contains(argNameLower, "model") {
		/* Model names */
		models := []string{"gpt-4", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "llama-2", "mistral"}
		for _, model := range models {
			if partialValue == "" || strings.HasPrefix(strings.ToLower(model), partialLower) {
				suggestions = append(suggestions, model)
			}
		}
	} else if strings.Contains(argNameLower, "format") || strings.Contains(argNameLower, "type") {
		/* Format types */
		formats := []string{"json", "xml", "yaml", "csv", "text", "markdown", "html"}
		for _, format := range formats {
			if partialValue == "" || strings.HasPrefix(strings.ToLower(format), partialLower) {
				suggestions = append(suggestions, format)
			}
		}
	} else if strings.Contains(argNameLower, "action") || strings.Contains(argNameLower, "operation") {
		/* Common actions */
		actions := []string{"create", "read", "update", "delete", "list", "search", "analyze"}
		for _, action := range actions {
			if partialValue == "" || strings.HasPrefix(strings.ToLower(action), partialLower) {
				suggestions = append(suggestions, action)
			}
		}
	} else if strings.Contains(argNameLower, "status") || strings.Contains(argNameLower, "state") {
		/* Status values */
		statuses := []string{"active", "inactive", "pending", "completed", "failed", "running", "stopped"}
		for _, status := range statuses {
			if partialValue == "" || strings.HasPrefix(strings.ToLower(status), partialLower) {
				suggestions = append(suggestions, status)
			}
		}
	}

	/* Use context if available to provide more specific suggestions */
	if context != nil {
		/* Check if there are related arguments that might help */
		/* For example, if language is "python", suggest python-specific options */
	}

	/* If no specific suggestions, return empty list */
	/* In a real implementation, you might query a database or use ML for suggestions */
	if len(suggestions) == 0 && partialValue != "" {
		/* If partial value is provided but no matches, return empty */
		/* This allows clients to show "no suggestions" */
	}

	/* Limit to reasonable number */
	maxSuggestions := 20
	if len(suggestions) > maxSuggestions {
		suggestions = suggestions[:maxSuggestions]
	}

	return suggestions
}
