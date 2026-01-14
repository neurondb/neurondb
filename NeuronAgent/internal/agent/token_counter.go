/*-------------------------------------------------------------------------
 *
 * token_counter.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/token_counter.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"strings"
	"unicode/utf8"
)

/* EstimateTokens estimates token count for text (rough approximation) */
/* For GPT models, ~4 characters = 1 token, but this varies */
func EstimateTokens(text string) int {
	/* Simple approximation: count words and add some overhead */
	words := strings.Fields(text)
	baseTokens := len(words)

	/* Add tokens for punctuation and special characters */
	charCount := utf8.RuneCountInString(text)
	charTokens := charCount / 4

	/* Use the larger estimate */
	if charTokens > baseTokens {
		return charTokens
	}
	return baseTokens
}

/* CountTokensInMessages counts total tokens in messages */
func CountTokensInMessages(messages []interface{}) int {
	total := 0
	for _, msg := range messages {
		if msgMap, ok := msg.(map[string]interface{}); ok {
			if content, ok := msgMap["content"].(string); ok {
				total += EstimateTokens(content)
			}
		} else if msgStr, ok := msg.(string); ok {
			total += EstimateTokens(msgStr)
		}
	}
	return total
}

/* TruncateToMaxTokens truncates text to fit within max tokens */
func TruncateToMaxTokens(text string, maxTokens int) string {
	tokens := EstimateTokens(text)
	if tokens <= maxTokens {
		return text
	}

	/* Rough truncation - remove from end */
	charsPerToken := len(text) / tokens
	maxChars := maxTokens * charsPerToken

	if maxChars >= len(text) {
		return text
	}

	/* Truncate and add ellipsis */
	truncated := text[:maxChars]
	/* Try to cut at word boundary */
	if lastSpace := strings.LastIndex(truncated, " "); lastSpace > maxChars*3/4 {
		truncated = truncated[:lastSpace]
	}
	return truncated + "..."
}
