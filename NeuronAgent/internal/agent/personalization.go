/*-------------------------------------------------------------------------
 *
 * personalization.go
 *    Personalization based on stored memories
 *
 * Extracts user preferences from Long-Term Personal Memory (LPM) and
 * customizes responses based on stored preferences, communication style,
 * and past conversations.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/personalization.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

/* PersonalizationManager manages personalization based on memories */
type PersonalizationManager struct {
	hierMemory *HierarchicalMemoryManager
	queries    *db.Queries
}

/* NewPersonalizationManager creates a new personalization manager */
func NewPersonalizationManager(hierMemory *HierarchicalMemoryManager, queries *db.Queries) *PersonalizationManager {
	return &PersonalizationManager{
		hierMemory: hierMemory,
		queries:    queries,
	}
}

/* PersonalizationContext contains personalized context for an agent */
type PersonalizationContext struct {
	Preferences      map[string]interface{} /* User preferences */
	CommunicationStyle string                /* Communication style preferences */
	Interests        []string               /* User interests */
	PastTopics       []string               /* Topics discussed before */
	Customizations   map[string]interface{} /* Custom response customizations */
}

/* GetPersonalizationContext extracts personalization context from memories */
func (p *PersonalizationManager) GetPersonalizationContext(ctx context.Context, agentID uuid.UUID, userID *uuid.UUID) (*PersonalizationContext, error) {
	context := &PersonalizationContext{
		Preferences:      make(map[string]interface{}),
		CommunicationStyle: "neutral",
		Interests:        make([]string, 0),
		PastTopics:       make([]string, 0),
		Customizations:   make(map[string]interface{}),
	}

	if p.hierMemory == nil {
		return context, nil /* Return empty context if hierarchical memory not available */
	}

	/* Retrieve LPM memories for preferences using hierarchical retrieval */
	lpmResults, err := p.hierMemory.RetrieveHierarchical(ctx, agentID, "preferences interests communication style", []string{"lpm"}, 20)
	if err != nil {
		/* Log error but return empty context */
		metrics.WarnWithContext(ctx, "Failed to retrieve LPM for personalization", map[string]interface{}{
			"agent_id": agentID.String(),
			"error":    err.Error(),
		})
		return context, nil
	}

	/* Process hierarchical retrieval results */
	for _, result := range lpmResults {
		content, _ := result["content"].(string)
		metadata, _ := result["metadata"].(map[string]interface{})
		
		/* Try to get category from metadata */
		category := ""
		if metadata != nil {
			if cat, ok := metadata["category"].(string); ok {
				category = cat
			}
		}

		/* Category extraction from metadata is sufficient for now */
		/* Future: Can query database directly if needed */

		/* Process based on category or content */
		switch category {
		case "preference":
			p.extractPreference(content, context.Preferences)
		case "interest":
			context.Interests = append(context.Interests, content)
		case "communication_style":
			context.CommunicationStyle = strings.ToLower(content)
		case "topic":
			context.PastTopics = append(context.PastTopics, content)
		default:
			/* Try to infer from content */
			if strings.Contains(strings.ToLower(content), "prefer") || strings.Contains(strings.ToLower(content), "like") {
				p.extractPreference(content, context.Preferences)
			} else if strings.Contains(strings.ToLower(content), "interest") {
				context.Interests = append(context.Interests, content)
			} else if strings.Contains(strings.ToLower(content), "communication") || strings.Contains(strings.ToLower(content), "style") {
				context.CommunicationStyle = strings.ToLower(content)
			}
		}
	}

	return context, nil
}

/* extractPreference extracts preference from content text */
func (p *PersonalizationManager) extractPreference(content string, preferences map[string]interface{}) {
	/* Simple extraction - in production, use LLM for better extraction */
	contentLower := strings.ToLower(content)

	/* Common preference patterns */
	preferencePatterns := map[string]string{
		"coffee":     "beverage",
		"tea":        "beverage",
		"formal":     "communication_style",
		"casual":     "communication_style",
		"brief":      "response_length",
		"detailed":   "response_length",
		"technical":  "detail_level",
		"simple":     "detail_level",
	}

	for keyword, category := range preferencePatterns {
		if strings.Contains(contentLower, keyword) {
			preferences[category] = keyword
		}
	}

	/* Store raw preference if no pattern matched */
	if len(preferences) == 0 {
		preferences["raw"] = content
	}
}

/* BuildPersonalizedPrompt builds personalized prompt section */
func (p *PersonalizationManager) BuildPersonalizedPrompt(ctx *PersonalizationContext) string {
	if ctx == nil {
		return ""
	}

	var parts []string

	/* Add preferences */
	if len(ctx.Preferences) > 0 {
		parts = append(parts, "\n## User Preferences:")
		for key, value := range ctx.Preferences {
			parts = append(parts, fmt.Sprintf("- %s: %v", key, value))
		}
	}

	/* Add communication style */
	if ctx.CommunicationStyle != "neutral" && ctx.CommunicationStyle != "" {
		parts = append(parts, fmt.Sprintf("\n## Communication Style: %s", ctx.CommunicationStyle))
		parts = append(parts, fmt.Sprintf("Adjust your communication style to be %s.", ctx.CommunicationStyle))
	}

	/* Add interests */
	if len(ctx.Interests) > 0 {
		parts = append(parts, "\n## User Interests:")
		for _, interest := range ctx.Interests {
			parts = append(parts, fmt.Sprintf("- %s", interest))
		}
		parts = append(parts, "You can reference these interests when relevant to the conversation.")
	}

	/* Add past topics */
	if len(ctx.PastTopics) > 0 {
		parts = append(parts, "\n## Previously Discussed Topics:")
		for _, topic := range ctx.PastTopics {
			parts = append(parts, fmt.Sprintf("- %s", topic))
		}
		parts = append(parts, "You can reference these past discussions when relevant.")
	}

	return strings.Join(parts, "\n")
}

/* CustomizeResponse customizes response based on personalization context */
func (p *PersonalizationManager) CustomizeResponse(ctx *PersonalizationContext, response string) string {
	if ctx == nil {
		return response
	}

	/* Apply communication style customizations */
	if ctx.CommunicationStyle == "formal" {
		/* Make response more formal (simplified) */
		response = strings.ReplaceAll(response, "you're", "you are")
		response = strings.ReplaceAll(response, "can't", "cannot")
		response = strings.ReplaceAll(response, "won't", "will not")
	} else if ctx.CommunicationStyle == "casual" {
		/* Keep casual - no changes needed */
	}

	/* Apply response length preferences */
	if responseLength, ok := ctx.Preferences["response_length"].(string); ok {
		if responseLength == "brief" {
			/* Truncate if too long (simplified) */
			if len(response) > 500 {
				/* Find last sentence before 500 chars */
				truncated := response[:500]
				lastPeriod := strings.LastIndex(truncated, ".")
				if lastPeriod > 400 {
					response = truncated[:lastPeriod+1]
				} else {
					response = truncated + "..."
				}
			}
		}
	}

	return response
}

/* GetUserIDFromSession extracts user ID from session if available */
func GetUserIDFromSession(queries *db.Queries, ctx context.Context, sessionID uuid.UUID) (*uuid.UUID, error) {
	/* For now, return nil - user ID extraction can be enhanced */
	/* Future: Extract from session metadata or external_user_id mapping */
	return nil, nil
}
