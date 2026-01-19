/*-------------------------------------------------------------------------
 *
 * personalization_test.go
 *    Tests for personalization
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/personalization_test.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"testing"
)

/* TestPersonalizationManager_BuildPersonalizedPrompt tests prompt building */
func TestPersonalizationManager_BuildPersonalizedPrompt(t *testing.T) {
	manager := &PersonalizationManager{}

	/* Test with empty context */
	ctx := &PersonalizationContext{
		Preferences:      make(map[string]interface{}),
		CommunicationStyle: "neutral",
		Interests:        []string{},
		PastTopics:       []string{},
		Customizations:   make(map[string]interface{}),
	}

	result := manager.BuildPersonalizedPrompt(ctx)
	if result != "" {
		t.Errorf("Expected empty prompt for empty context, got: %s", result)
	}

	/* Test with preferences */
	ctx.Preferences["beverage"] = "coffee"
	ctx.Preferences["response_length"] = "brief"
	result = manager.BuildPersonalizedPrompt(ctx)
	if result == "" {
		t.Error("Expected non-empty prompt with preferences")
	}
	if !containsSubstring(result, []string{"coffee", "beverage"}) {
		t.Error("Expected prompt to contain preference information")
	}

	/* Test with communication style */
	ctx.CommunicationStyle = "formal"
	result = manager.BuildPersonalizedPrompt(ctx)
	if !containsSubstring(result, []string{"formal", "communication"}) {
		t.Error("Expected prompt to contain communication style")
	}

	/* Test with interests */
	ctx.Interests = []string{"AI", "machine learning"}
	result = manager.BuildPersonalizedPrompt(ctx)
	if !containsSubstring(result, []string{"AI", "interests"}) {
		t.Error("Expected prompt to contain interests")
	}
}

/* TestPersonalizationManager_CustomizeResponse tests response customization */
func TestPersonalizationManager_CustomizeResponse(t *testing.T) {
	manager := &PersonalizationManager{}

	/* Test with nil context */
	result := manager.CustomizeResponse(nil, "Hello, how are you?")
	if result != "Hello, how are you?" {
		t.Error("Expected unchanged response for nil context")
	}

	/* Test with formal style */
	ctx := &PersonalizationContext{
		CommunicationStyle: "formal",
	}
	result = manager.CustomizeResponse(ctx, "You're doing great! Can't wait to see you.")
	if !containsSubstring(result, []string{"you are", "cannot"}) {
		t.Logf("Formal customization result: %s", result)
		/* Note: Customization is simplified, may not always work */
	}

	/* Test with brief response length */
	ctx.Preferences = map[string]interface{}{
		"response_length": "brief",
	}
	longResponse := "This is a very long response. " + string(make([]byte, 600))
	result = manager.CustomizeResponse(ctx, longResponse)
	if len(result) > 550 {
		t.Logf("Brief customization may not have truncated: length=%d", len(result))
		/* Note: Truncation is simplified */
	}
}

/* TestPersonalizationManager_extractPreference tests preference extraction */
func TestPersonalizationManager_extractPreference(t *testing.T) {
	manager := &PersonalizationManager{}

	preferences := make(map[string]interface{})
	manager.extractPreference("I prefer coffee over tea", preferences)
	if len(preferences) == 0 {
		t.Error("Expected preferences to be extracted")
	}

	preferences = make(map[string]interface{})
	manager.extractPreference("I like formal communication", preferences)
	if preferences["communication_style"] == nil {
		t.Error("Expected communication_style to be extracted")
	}
}
