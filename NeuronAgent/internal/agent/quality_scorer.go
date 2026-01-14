/*-------------------------------------------------------------------------
 *
 * quality_scorer.go
 *    Response quality scoring
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/quality_scorer.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"strings"
)

type QualityScorer struct {
	reflector *Reflector
}

/* NewQualityScorer creates a new quality scorer */
func NewQualityScorer(reflector *Reflector) *QualityScorer {
	return &QualityScorer{reflector: reflector}
}

/* ScoreResponse scores the quality of a response */
func (q *QualityScorer) ScoreResponse(ctx context.Context, userMessage, response string, toolCalls []ToolCall, toolResults []ToolResult) (*QualityScore, error) {
	/* Use reflector for comprehensive scoring */
	reflection, err := q.reflector.Reflect(ctx, userMessage, response, toolCalls, toolResults)
	if err != nil {
		return nil, fmt.Errorf("quality scoring failed: user_message_length=%d, response_length=%d, reflection_error=%w",
			len(userMessage), len(response), err)
	}

	return &QualityScore{
		Overall:      reflection.QualityScore,
		Accuracy:     reflection.Accuracy,
		Completeness: reflection.Completeness,
		Clarity:      reflection.Clarity,
		Relevance:    reflection.Relevance,
		Confidence:   reflection.Confidence,
		Issues:       reflection.Issues,
	}, nil
}

/* QuickScore provides a quick heuristic-based score */
func (q *QualityScorer) QuickScore(response string) float64 {
	score := 0.5 /* Base score */

	/* Length check */
	if len(response) < 20 {
		score -= 0.3
	} else if len(response) > 100 {
		score += 0.1
	}

	/* Check for error indicators */
	responseLower := strings.ToLower(response)
	errorIndicators := []string{"error", "failed", "unable", "cannot"}
	for _, indicator := range errorIndicators {
		if strings.Contains(responseLower, indicator) {
			score -= 0.2
			break
		}
	}

	/* Check for positive indicators */
	positiveIndicators := []string{"success", "completed", "found", "result"}
	for _, indicator := range positiveIndicators {
		if strings.Contains(responseLower, indicator) {
			score += 0.1
			break
		}
	}

	/* Normalize */
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	return score
}

/* QualityScore represents a quality score */
type QualityScore struct {
	Overall      float64  `json:"overall"`
	Accuracy     float64  `json:"accuracy"`
	Completeness float64  `json:"completeness"`
	Clarity      float64  `json:"clarity"`
	Relevance    float64  `json:"relevance"`
	Confidence   float64  `json:"confidence"`
	Issues       []string `json:"issues"`
}
