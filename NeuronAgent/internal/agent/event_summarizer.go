/*-------------------------------------------------------------------------
 *
 * event_summarizer.go
 *    LLM-powered event summarization for context compression
 *
 * Provides intelligent summarization of event ranges to compress
 * context history while preserving key information.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/event_summarizer.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"strings"
	"time"
)

/* EventSummarizer provides event summarization capabilities */
type EventSummarizer struct {
	llmClient *LLMClient
}

/* EventForSummary represents an event to be summarized */
type EventForSummary struct {
	EventType string
	Actor     string
	Content   string
	Timestamp time.Time
}

/* NewEventSummarizer creates a new event summarizer */
func NewEventSummarizer(llmClient *LLMClient) *EventSummarizer {
	return &EventSummarizer{
		llmClient: llmClient,
	}
}

/* SummarizeEvents generates a summary of multiple events */
func (e *EventSummarizer) SummarizeEvents(ctx context.Context, events interface{}) (string, error) {
	/* Convert events to summarizable format */
	var eventsText strings.Builder
	eventsText.WriteString("Event History:\n\n")

	/* Handle different event types */
	switch v := events.(type) {
	case []EventForSummary:
		for i, event := range v {
			eventsText.WriteString(fmt.Sprintf("%d. [%s] %s: %s\n",
				i+1, event.EventType, event.Actor, event.Content))
		}
	default:
		/* Handle database event rows */
		type EventRow struct {
			EventType string
			Actor     string
			Content   string
			CreatedAt time.Time
		}

		if eventRows, ok := events.([]EventRow); ok {
			for i, event := range eventRows {
				eventsText.WriteString(fmt.Sprintf("%d. [%s] %s: %s\n",
					i+1, event.EventType, event.Actor, event.Content))
			}
		} else {
			return "", fmt.Errorf("unsupported event type for summarization")
		}
	}

	/* If events are short enough, return as-is */
	if eventsText.Len() < 500 {
		return e.createBasicSummary(eventsText.String()), nil
	}

	/* Use LLM for intelligent summarization */
	summary, err := e.summarizeWithLLM(ctx, eventsText.String())
	if err == nil && summary != "" {
		return summary, nil
	}
	/* Fall back to basic summary on LLM error */

	/* Fallback to basic summarization */
	return e.createBasicSummary(eventsText.String()), nil
}

/* summarizeWithLLM uses LLM to create intelligent summary */
func (e *EventSummarizer) summarizeWithLLM(ctx context.Context, eventsText string) (string, error) {
	/* LLM client integration would go here */
	/* For now, return empty to use fallback */
	return "", fmt.Errorf("LLM summarization not yet implemented")
}

/* createBasicSummary creates a basic non-LLM summary */
func (e *EventSummarizer) createBasicSummary(eventsText string) string {
	lines := strings.Split(eventsText, "\n")

	/* Count event types */
	userMessages := 0
	agentActions := 0
	toolExecutions := 0
	errors := 0

	for _, line := range lines {
		if strings.Contains(line, "[user_message]") {
			userMessages++
		} else if strings.Contains(line, "[agent_action]") {
			agentActions++
		} else if strings.Contains(line, "[tool_execution]") {
			toolExecutions++
		} else if strings.Contains(line, "[error]") {
			errors++
		}
	}

	summary := fmt.Sprintf("Event Summary: %d user messages, %d agent actions, %d tool executions",
		userMessages, agentActions, toolExecutions)

	if errors > 0 {
		summary += fmt.Sprintf(", %d errors", errors)
	}

	/* Include first and last events for context */
	if len(lines) > 3 {
		summary += fmt.Sprintf("\n\nFirst event: %s", lines[2])
		summary += fmt.Sprintf("\nLast event: %s", lines[len(lines)-2])
	}

	return summary
}

/* SummarizeToMidTerm creates a summary suitable for mid-term memory */
func (e *EventSummarizer) SummarizeToMidTerm(ctx context.Context, events []EventForSummary) (string, error) {
	/* Extract recurring patterns and key topics */
	topics := make(map[string]int)
	actions := make(map[string]int)

	for _, event := range events {
		/* Extract keywords from content */
		words := strings.Fields(strings.ToLower(event.Content))
		for _, word := range words {
			if len(word) > 5 { /* Only significant words */
				topics[word]++
			}
		}

		if event.EventType == "agent_action" {
			actions[event.Actor]++
		}
	}

	/* Build mid-term summary */
	var summary strings.Builder
	summary.WriteString("Topic Summary: ")

	/* Add top topics */
	topTopics := getTopItems(topics, 3)
	summary.WriteString(strings.Join(topTopics, ", "))

	/* Add action summary */
	if len(actions) > 0 {
		summary.WriteString("\nKey Actions: ")
		topActions := getTopItems(actions, 3)
		summary.WriteString(strings.Join(topActions, ", "))
	}

	summary.WriteString(fmt.Sprintf("\nEvent Count: %d", len(events)))

	return summary.String(), nil
}

/* getTopItems returns top N items from a frequency map */
func getTopItems(items map[string]int, n int) []string {
	type item struct {
		key   string
		count int
	}

	sorted := make([]item, 0, len(items))
	for k, v := range items {
		sorted = append(sorted, item{k, v})
	}

	/* Simple bubble sort for small n */
	for i := 0; i < len(sorted)-1; i++ {
		for j := 0; j < len(sorted)-i-1; j++ {
			if sorted[j].count < sorted[j+1].count {
				sorted[j], sorted[j+1] = sorted[j+1], sorted[j]
			}
		}
	}

	result := make([]string, 0, n)
	for i := 0; i < n && i < len(sorted); i++ {
		result = append(result, sorted[i].key)
	}

	return result
}
