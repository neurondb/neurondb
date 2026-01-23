/*-------------------------------------------------------------------------
 *
 * manager_test.go
 *    Tests for completion manager
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/completion/manager_test.go
 *
 *-------------------------------------------------------------------------
 */

package completion

import (
	"context"
	"testing"

	"github.com/neurondb/NeuronMCP/internal/prompts"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* TestCompletePromptArgument tests prompt argument completion */
func TestCompletePromptArgument(t *testing.T) {
	/* Create a mock prompts manager */
	/* In a real test, you would use a mock or test database */
	
	/* This is a basic structure test - full integration tests would require database setup */
	manager := &Manager{
		promptsManager:   nil, /* Would be mocked in real test */
		resourcesManager: nil, /* Would be mocked in real test */
	}

	req := mcp.CompletionRequest{
		Ref: mcp.CompletionReference{
			Type: "ref/prompt",
			Name: "test_prompt",
		},
		Argument: mcp.CompletionArgument{
			Name:  "language",
			Value: "py",
		},
	}

	/* Test with nil managers (should error) */
	_, err := manager.Complete(context.Background(), req)
	if err == nil {
		t.Error("Expected error when prompts manager is nil")
	}
}

/* TestCompleteResourceURI tests resource URI completion */
func TestCompleteResourceURI(t *testing.T) {
	manager := &Manager{
		promptsManager:   nil,
		resourcesManager: nil,
	}

	req := mcp.CompletionRequest{
		Ref: mcp.CompletionReference{
			Type: "ref/resource",
		},
		Argument: mcp.CompletionArgument{
			Name:  "uri",
			Value: "schema",
		},
	}

	/* Test with nil managers (should error) */
	_, err := manager.Complete(context.Background(), req)
	if err == nil {
		t.Error("Expected error when resources manager is nil")
	}
}

/* TestGeneratePromptArgumentCompletions tests completion generation */
func TestGeneratePromptArgumentCompletions(t *testing.T) {
	manager := &Manager{}

	/* Test language completion */
	varDef := &prompts.VariableDefinition{
		Name:        "language",
		Description: "Programming language",
		Required:    true,
	}

	completions := manager.generatePromptArgumentCompletions(varDef, "py", nil)

	/* Should return Python-related suggestions */
	if len(completions) == 0 {
		t.Error("Expected completion suggestions for language argument")
	}

	/* Check that Python is in the suggestions */
	foundPython := false
	for _, comp := range completions {
		if comp == "python" {
			foundPython = true
			break
		}
	}
	if !foundPython {
		t.Error("Expected 'python' in completion suggestions")
	}
}
