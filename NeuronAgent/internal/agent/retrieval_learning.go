/*-------------------------------------------------------------------------
 *
 * retrieval_learning.go
 *    Retrieval learning mechanism for agentic RAG
 *
 * Tracks retrieval decisions and outcomes to improve routing over time.
 * Learns from successful patterns and user feedback.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/retrieval_learning.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

/* RetrievalLearningManager manages learning from retrieval decisions */
type RetrievalLearningManager struct {
	db      *db.DB
	queries *db.Queries
	router  *KnowledgeRouter
}

/* NewRetrievalLearningManager creates a new retrieval learning manager */
func NewRetrievalLearningManager(db *db.DB, queries *db.Queries, router *KnowledgeRouter) *RetrievalLearningManager {
	return &RetrievalLearningManager{
		db:      db,
		queries: queries,
		router:  router,
	}
}

/* RetrievalDecision represents a retrieval decision */
type RetrievalDecision struct {
	ID            uuid.UUID
	AgentID       uuid.UUID
	SessionID     *uuid.UUID
	Query         string
	QueryType     string
	ShouldRetrieve bool
	Confidence    float64
	Reason        string
	Sources       []string
	SourceScores  map[string]float64
	CreatedAt     time.Time
}

/* RetrievalOutcome represents the outcome of a retrieval */
type RetrievalOutcome struct {
	ID             uuid.UUID
	DecisionID     uuid.UUID
	AgentID        uuid.UUID
	SessionID      *uuid.UUID
	Source         string
	ResultsCount   int
	RelevanceScore *float64
	UsedInResponse bool
	UserFeedback   *string
	QualityScore   *float64
	Metadata       map[string]interface{}
	CreatedAt      time.Time
}

/* RecordDecision records a retrieval decision for learning */
func (m *RetrievalLearningManager) RecordDecision(ctx context.Context, decision *RetrievalDecision) (uuid.UUID, error) {
	decisionID := uuid.New()
	if decision.ID != uuid.Nil {
		decisionID = decision.ID
	}

	/* Determine query type if not provided */
	queryType := decision.QueryType
	if queryType == "" && m.router != nil {
		/* Use knowledge router to classify query */
		_, _, err := m.router.RouteQuery(ctx, decision.Query)
		if err == nil {
			/* Extract query type from routing (simplified) */
			queryType = m.inferQueryType(decision.Query)
		}
	}

	sourceScoresJSON, _ := json.Marshal(decision.SourceScores)

	query := `INSERT INTO neurondb_agent.retrieval_decisions
		(id, agent_id, session_id, query, query_type, should_retrieve, confidence, reason, sources, source_scores, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::text[], $10::jsonb, NOW())
		RETURNING id`

	err := m.db.DB.GetContext(ctx, &decisionID, query,
		decisionID,
		decision.AgentID,
		decision.SessionID,
		decision.Query,
		queryType,
		decision.ShouldRetrieve,
		decision.Confidence,
		decision.Reason,
		decision.Sources,
		sourceScoresJSON,
	)

	if err != nil {
		return uuid.Nil, fmt.Errorf("failed to record retrieval decision: %w", err)
	}

	/* Record metrics */
	metrics.InfoWithContext(ctx, "Retrieval decision recorded", map[string]interface{}{
		"decision_id":    decisionID.String(),
		"agent_id":       decision.AgentID.String(),
		"should_retrieve": decision.ShouldRetrieve,
		"query_type":     queryType,
		"sources":        decision.Sources,
	})

	return decisionID, nil
}

/* RecordOutcome records the outcome of a retrieval decision */
func (m *RetrievalLearningManager) RecordOutcome(ctx context.Context, outcome *RetrievalOutcome) error {
	outcomeID := uuid.New()
	if outcome.ID != uuid.Nil {
		outcomeID = outcome.ID
	}

	metadataJSON, _ := json.Marshal(outcome.Metadata)

	query := `INSERT INTO neurondb_agent.retrieval_outcomes
		(id, decision_id, agent_id, session_id, source, results_count, relevance_score, used_in_response, user_feedback, quality_score, metadata, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, NOW())`

	_, err := m.db.DB.ExecContext(ctx, query,
		outcomeID,
		outcome.DecisionID,
		outcome.AgentID,
		outcome.SessionID,
		outcome.Source,
		outcome.ResultsCount,
		outcome.RelevanceScore,
		outcome.UsedInResponse,
		outcome.UserFeedback,
		outcome.QualityScore,
		metadataJSON,
	)

	if err != nil {
		return fmt.Errorf("failed to record retrieval outcome: %w", err)
	}

	/* Record metrics */
	metrics.InfoWithContext(ctx, "Retrieval outcome recorded", map[string]interface{}{
		"outcome_id":      outcomeID.String(),
		"decision_id":     outcome.DecisionID.String(),
		"source":          outcome.Source,
		"results_count":   outcome.ResultsCount,
		"used_in_response": outcome.UsedInResponse,
	})

	return nil
}

/* LearnFromPatterns analyzes past decisions to improve routing */
func (m *RetrievalLearningManager) LearnFromPatterns(ctx context.Context, agentID uuid.UUID, queryType string) ([]string, map[string]float64, error) {
	/* Query successful patterns for this query type */
	query := `SELECT source, AVG(quality_score) as avg_quality, COUNT(*) as usage_count
		FROM neurondb_agent.retrieval_outcomes ro
		JOIN neurondb_agent.retrieval_decisions rd ON ro.decision_id = rd.id
		WHERE rd.agent_id = $1
		AND ($2 IS NULL OR rd.query_type = $2)
		AND ro.quality_score IS NOT NULL
		AND ro.quality_score > 0.6
		AND ro.created_at > NOW() - INTERVAL '30 days'
		GROUP BY source
		ORDER BY avg_quality DESC, usage_count DESC
		LIMIT 5`

	type PatternRow struct {
		Source      string  `db:"source"`
		AvgQuality  float64 `db:"avg_quality"`
		UsageCount  int     `db:"usage_count"`
	}

	var patterns []PatternRow
	err := m.db.DB.SelectContext(ctx, &patterns, query, agentID, queryType)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to learn from patterns: %w", err)
	}

	if len(patterns) == 0 {
		/* No patterns found, return empty */
		return []string{}, make(map[string]float64), nil
	}

	/* Build recommended sources and scores */
	sources := make([]string, 0, len(patterns))
	scores := make(map[string]float64)

	for _, pattern := range patterns {
		sources = append(sources, pattern.Source)
		/* Normalize score to 0-1 range, weighted by usage count */
		normalizedScore := pattern.AvgQuality
		if pattern.UsageCount > 10 {
			normalizedScore = normalizedScore * 1.1 /* Boost for high usage */
		}
		if normalizedScore > 1.0 {
			normalizedScore = 1.0
		}
		scores[pattern.Source] = normalizedScore
	}

	return sources, scores, nil
}

/* GetDecisionCache checks if we have a cached decision for a similar query */
func (m *RetrievalLearningManager) GetDecisionCache(ctx context.Context, agentID uuid.UUID, query string, similarityThreshold float64) (*RetrievalDecision, error) {
	/* For now, use simple exact match or substring match */
	/* In production, could use embedding similarity */
	querySQL := `SELECT id, agent_id, session_id, query, query_type, should_retrieve, confidence, reason, sources, source_scores, created_at
		FROM neurondb_agent.retrieval_decisions
		WHERE agent_id = $1
		AND query = $2
		AND created_at > NOW() - INTERVAL '7 days'
		ORDER BY created_at DESC
		LIMIT 1`

	var decision RetrievalDecision
	var sourceScoresJSON []byte
	var sessionID *uuid.UUID

	err := m.db.DB.GetContext(ctx, &struct {
		ID            uuid.UUID       `db:"id"`
		AgentID       uuid.UUID       `db:"agent_id"`
		SessionID     *uuid.UUID      `db:"session_id"`
		Query         string          `db:"query"`
		QueryType     *string         `db:"query_type"`
		ShouldRetrieve bool           `db:"should_retrieve"`
		Confidence    float64         `db:"confidence"`
		Reason        *string         `db:"reason"`
		Sources       []string        `db:"sources"`
		SourceScores  []byte          `db:"source_scores"`
		CreatedAt     time.Time       `db:"created_at"`
	}{
		ID: decision.ID,
		AgentID: decision.AgentID,
		SessionID: sessionID,
		Query: decision.Query,
		ShouldRetrieve: decision.ShouldRetrieve,
		Confidence: decision.Confidence,
		Sources: decision.Sources,
		SourceScores: sourceScoresJSON,
		CreatedAt: decision.CreatedAt,
	}, querySQL, agentID, query)

	if err != nil {
		/* No cached decision found */
		return nil, nil
	}

	/* Parse source scores */
	if len(sourceScoresJSON) > 0 {
		json.Unmarshal(sourceScoresJSON, &decision.SourceScores)
	}

	return &decision, nil
}

/* UpdateQualityScore updates quality score for an outcome based on feedback */
func (m *RetrievalLearningManager) UpdateQualityScore(ctx context.Context, outcomeID uuid.UUID, qualityScore float64) error {
	query := `UPDATE neurondb_agent.retrieval_outcomes
		SET quality_score = $1, updated_at = NOW()
		WHERE id = $2`

	_, err := m.db.DB.ExecContext(ctx, query, qualityScore, outcomeID)
	return err
}

/* GetRetrievalStats returns statistics about retrieval decisions */
func (m *RetrievalLearningManager) GetRetrievalStats(ctx context.Context, agentID uuid.UUID, days int) (map[string]interface{}, error) {
	if days <= 0 {
		days = 30
	}

	stats := make(map[string]interface{})

	/* Total decisions */
	var totalDecisions int
	err := m.db.DB.GetContext(ctx, &totalDecisions, `
		SELECT COUNT(*) FROM neurondb_agent.retrieval_decisions
		WHERE agent_id = $1 AND created_at > NOW() - INTERVAL '1 day' * $2`,
		agentID, days)
	if err == nil {
		stats["total_decisions"] = totalDecisions
	}

	/* Average confidence */
	var avgConfidence float64
	err = m.db.DB.GetContext(ctx, &avgConfidence, `
		SELECT AVG(confidence) FROM neurondb_agent.retrieval_decisions
		WHERE agent_id = $1 AND created_at > NOW() - INTERVAL '1 day' * $2`,
		agentID, days)
	if err == nil {
		stats["avg_confidence"] = avgConfidence
	}

	/* Source usage */
	type SourceUsage struct {
		Source string `db:"source"`
		Count  int    `db:"count"`
	}
	var sourceUsages []SourceUsage
	err = m.db.DB.SelectContext(ctx, &sourceUsages, `
		SELECT source, COUNT(*) as count
		FROM neurondb_agent.retrieval_outcomes
		WHERE agent_id = $1 AND created_at > NOW() - INTERVAL '1 day' * $2
		GROUP BY source
		ORDER BY count DESC`,
		agentID, days)
	if err == nil {
		sourceMap := make(map[string]int)
		for _, usage := range sourceUsages {
			sourceMap[usage.Source] = usage.Count
		}
		stats["source_usage"] = sourceMap
	}

	/* Average quality score */
	var avgQuality float64
	err = m.db.DB.GetContext(ctx, &avgQuality, `
		SELECT AVG(quality_score) FROM neurondb_agent.retrieval_outcomes
		WHERE agent_id = $1 AND quality_score IS NOT NULL AND created_at > NOW() - INTERVAL '1 day' * $2`,
		agentID, days)
	if err == nil {
		stats["avg_quality_score"] = avgQuality
	}

	return stats, nil
}

/* inferQueryType infers query type from query text (simplified heuristic) */
func (m *RetrievalLearningManager) inferQueryType(query string) string {
	queryLower := strings.ToLower(query)
	/* Simple heuristic - in production, use LLM classification */
	if containsSubstring(queryLower, []string{"today", "now", "current", "latest", "news"}) {
		return "current_events"
	}
	if containsSubstring(queryLower, []string{"remember", "told", "prefer", "like", "my"}) {
		return "semantic"
	}
	if containsSubstring(queryLower, []string{"api", "database", "query", "sql"}) {
		return "structured"
	}
	return "factual"
}

/* containsSubstring checks if string contains any of the substrings */
func containsSubstring(s string, substrings []string) bool {
	for _, substr := range substrings {
		if len(s) >= len(substr) {
			for i := 0; i <= len(s)-len(substr); i++ {
				if s[i:i+len(substr)] == substr {
					return true
				}
			}
		}
	}
	return false
}
