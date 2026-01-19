/*-------------------------------------------------------------------------
 *
 * memory_auto_writer.go
 *    Automatic memory writing during conversations
 *
 * Uses LLM to extract important facts, preferences, and learnings from
 * conversations and stores them to appropriate memory tiers.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/memory_auto_writer.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

/* MemoryAutoWriter automatically extracts and stores important information */
type MemoryAutoWriter struct {
	llm        *LLMClient
	hierMemory *HierarchicalMemoryManager
	queries    *db.Queries
}

/* NewMemoryAutoWriter creates a new automatic memory writer */
func NewMemoryAutoWriter(llm *LLMClient, hierMemory *HierarchicalMemoryManager, queries *db.Queries) *MemoryAutoWriter {
	return &MemoryAutoWriter{
		llm:        llm,
		hierMemory: hierMemory,
		queries:    queries,
	}
}

/* ExtractAndStore extracts important information and stores it to appropriate tier */
func (w *MemoryAutoWriter) ExtractAndStore(ctx context.Context, agentID, sessionID uuid.UUID, userMessage, assistantResponse string, enabled bool) error {
	if !enabled {
		return nil /* Auto memory writing disabled */
	}

	/* Check if there's meaningful content to extract */
	if len(userMessage) < 10 && len(assistantResponse) < 10 {
		return nil /* Too short to extract meaningful information */
	}

	/* Use LLM to extract important information */
	extractions, err := w.extractImportantInfo(ctx, userMessage, assistantResponse)
	if err != nil {
		/* Log error but don't fail - auto memory writing is non-critical */
		metrics.WarnWithContext(ctx, "Failed to extract important information", map[string]interface{}{
			"agent_id":   agentID.String(),
			"session_id": sessionID.String(),
			"error":      err.Error(),
		})
		return nil
	}

	/* Store each extraction to appropriate tier */
	for _, extraction := range extractions {
		if extraction.Importance < 0.3 {
			continue /* Skip low-importance extractions */
		}

		var err error
		switch extraction.Tier {
		case "lpm":
			/* Long-term personal memory - preferences, facts about user */
			err = w.storeLPM(ctx, agentID, sessionID, extraction)
		case "mtm":
			/* Mid-term memory - topic summaries, patterns */
			err = w.storeMTM(ctx, agentID, sessionID, extraction)
		case "stm":
			/* Short-term memory - conversation context */
			err = w.storeSTM(ctx, agentID, sessionID, extraction)
		default:
			/* Default to STM */
			err = w.storeSTM(ctx, agentID, sessionID, extraction)
		}

		if err != nil {
			metrics.WarnWithContext(ctx, "Failed to store extracted memory", map[string]interface{}{
				"agent_id":   agentID.String(),
				"session_id": sessionID.String(),
				"tier":       extraction.Tier,
				"error":      err.Error(),
			})
			/* Continue with other extractions */
		}
	}

	return nil
}

/* Extraction represents extracted important information */
type Extraction struct {
	Content    string
	Category   string
	Tier       string /* stm, mtm, lpm */
	Importance float64
	Metadata   map[string]interface{}
}

/* extractImportantInfo uses LLM to extract important information */
func (w *MemoryAutoWriter) extractImportantInfo(ctx context.Context, userMessage, assistantResponse string) ([]Extraction, error) {
	if w.llm == nil {
		return []Extraction{}, nil
	}

	prompt := fmt.Sprintf(`Extract important information from this conversation that should be remembered. Focus on:
1. User preferences, likes, dislikes
2. Important facts about the user
3. Key learnings or insights
4. Topics discussed that might be relevant later

User: "%s"
Assistant: "%s"

Return a JSON array of extractions, each with:
- content: The information to remember (string)
- category: Category (e.g., "preference", "fact", "learning", "topic") (string)
- tier: Memory tier - "lpm" for long-term personal (preferences, facts), "mtm" for mid-term (topics, patterns), "stm" for short-term (conversation context) (string)
- importance: Importance score 0.0-1.0 (float)
- metadata: Optional metadata object

Only extract information that is:
- Specific and actionable
- Likely to be useful in future conversations
- Not already obvious from the conversation flow

Return ONLY valid JSON array, no other text. Example:
[
  {
    "content": "User prefers coffee over tea",
    "category": "preference",
    "tier": "lpm",
    "importance": 0.8,
    "metadata": {}
  }
]`, userMessage, assistantResponse)

	llmConfig := map[string]interface{}{
		"temperature": 0.3,
		"max_tokens":  1000,
	}

	response, err := w.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		return nil, fmt.Errorf("LLM extraction failed: %w", err)
	}

	/* Parse JSON response */
	content := strings.TrimSpace(response.Content)
	/* Remove markdown code blocks if present */
	if strings.HasPrefix(content, "```json") {
		content = strings.TrimPrefix(content, "```json")
		content = strings.TrimSuffix(content, "```")
		content = strings.TrimSpace(content)
	} else if strings.HasPrefix(content, "```") {
		content = strings.TrimPrefix(content, "```")
		content = strings.TrimSuffix(content, "```")
		content = strings.TrimSpace(content)
	}

	var extractions []Extraction
	if err := json.Unmarshal([]byte(content), &extractions); err != nil {
		/* Try to extract JSON from response if it's embedded in text */
		startIdx := strings.Index(content, "[")
		endIdx := strings.LastIndex(content, "]")
		if startIdx >= 0 && endIdx > startIdx {
			jsonStr := content[startIdx : endIdx+1]
			if err := json.Unmarshal([]byte(jsonStr), &extractions); err != nil {
				return nil, fmt.Errorf("failed to parse extraction JSON: %w", err)
			}
		} else {
			return nil, fmt.Errorf("failed to parse extraction JSON: %w", err)
		}
	}

	return extractions, nil
}

/* storeLPM stores to long-term personal memory */
func (w *MemoryAutoWriter) storeLPM(ctx context.Context, agentID, sessionID uuid.UUID, extraction Extraction) error {
	if w.hierMemory == nil {
		return fmt.Errorf("hierarchical memory not available")
	}

	/* Use category as the category for LPM */
	category := extraction.Category
	if category == "" {
		category = "general"
	}

	/* Get user ID from session if available */
	/* Note: For now, we don't extract user ID from session - can be enhanced */
	var userID *uuid.UUID
	/* Future: Extract user ID from session metadata or external_user_id */

	_, err := w.hierMemory.StoreLPM(ctx, agentID, category, extraction.Content, extraction.Importance, userID)
	return err
}

/* storeMTM stores to mid-term memory */
func (w *MemoryAutoWriter) storeMTM(ctx context.Context, agentID, sessionID uuid.UUID, extraction Extraction) error {
	if w.hierMemory == nil {
		return fmt.Errorf("hierarchical memory not available")
	}

	/* Use category as topic for MTM */
	topic := extraction.Category
	if topic == "" {
		topic = "general"
	}

	/* StoreMTM signature: (ctx, agentID, topic, content, importance) */
	_, err := w.hierMemory.StoreMTM(ctx, agentID, topic, extraction.Content, extraction.Importance)
	return err
}

/* storeSTM stores to short-term memory */
func (w *MemoryAutoWriter) storeSTM(ctx context.Context, agentID, sessionID uuid.UUID, extraction Extraction) error {
	if w.hierMemory == nil {
		return fmt.Errorf("hierarchical memory not available")
	}

	_, err := w.hierMemory.StoreSTM(ctx, agentID, sessionID, extraction.Content, extraction.Importance)
	return err
}

/* ShouldStoreMemory determines if memory should be stored based on config */
func ShouldStoreMemory(agent *db.Agent) bool {
	if agent.Config == nil {
		return true /* Default: enabled */
	}

	if autoMemory, ok := agent.Config["auto_memory_enabled"].(bool); ok {
		return autoMemory
	}

	return true /* Default: enabled */
}
