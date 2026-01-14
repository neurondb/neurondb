/*-------------------------------------------------------------------------
 *
 * feedback.go
 *    User feedback integration
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/humanloop/feedback.go
 *
 *-------------------------------------------------------------------------
 */

package humanloop

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/google/uuid"
	"github.com/jmoiron/sqlx"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* FeedbackType represents feedback type */
type FeedbackType string

const (
	FeedbackPositive   FeedbackType = "positive"
	FeedbackNegative   FeedbackType = "negative"
	FeedbackNeutral    FeedbackType = "neutral"
	FeedbackCorrection FeedbackType = "correction"
)

/* UserFeedback represents user feedback */
type UserFeedback struct {
	ID           int64       `db:"id"`
	AgentID      *uuid.UUID  `db:"agent_id"`
	SessionID    *uuid.UUID  `db:"session_id"`
	MessageID    *int64      `db:"message_id"`
	UserID       *string     `db:"user_id"`
	FeedbackType string      `db:"feedback_type"`
	Rating       *int        `db:"rating"`
	Comment      *string     `db:"comment"`
	Metadata     db.JSONBMap `db:"metadata"`
	CreatedAt    string      `db:"created_at"`
}

/* FeedbackManager manages user feedback */
type FeedbackManager struct {
	db *sqlx.DB
}

/* NewFeedbackManager creates a new feedback manager */
func NewFeedbackManager(db *sqlx.DB) *FeedbackManager {
	return &FeedbackManager{db: db}
}

/* SubmitFeedback submits user feedback */
func (fm *FeedbackManager) SubmitFeedback(ctx context.Context, feedback *UserFeedback) error {
	query := `INSERT INTO neurondb_agent.user_feedback 
		(agent_id, session_id, message_id, user_id, feedback_type, rating, comment, metadata, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, NOW())
		RETURNING id, created_at`

	metadataJSON := db.FromMap(feedback.Metadata)
	if metadataJSON == nil {
		metadataJSON = make(db.JSONBMap)
	}

	err := fm.db.GetContext(ctx, feedback, query,
		feedback.AgentID, feedback.SessionID, feedback.MessageID, feedback.UserID,
		feedback.FeedbackType, feedback.Rating, feedback.Comment, metadataJSON)
	if err != nil {
		return fmt.Errorf("failed to submit feedback: %w", err)
	}
	return nil
}

/* GetFeedback gets feedback by ID */
func (fm *FeedbackManager) GetFeedback(ctx context.Context, id int64) (*UserFeedback, error) {
	var feedback UserFeedback
	query := `SELECT * FROM neurondb_agent.user_feedback WHERE id = $1`
	err := fm.db.GetContext(ctx, &feedback, query, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("feedback not found")
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get feedback: %w", err)
	}
	return &feedback, nil
}

/* ListFeedback lists feedback with filters */
func (fm *FeedbackManager) ListFeedback(ctx context.Context, agentID *uuid.UUID, sessionID *uuid.UUID, feedbackType *string, limit, offset int) ([]UserFeedback, error) {
	var feedbacks []UserFeedback
	query := `SELECT * FROM neurondb_agent.user_feedback 
		WHERE ($1::uuid IS NULL OR agent_id = $1)
		AND ($2::uuid IS NULL OR session_id = $2)
		AND ($3::text IS NULL OR feedback_type = $3)
		ORDER BY created_at DESC 
		LIMIT $4 OFFSET $5`

	err := fm.db.SelectContext(ctx, &feedbacks, query, agentID, sessionID, feedbackType, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to list feedback: %w", err)
	}
	return feedbacks, nil
}

/* GetFeedbackStats gets feedback statistics */
func (fm *FeedbackManager) GetFeedbackStats(ctx context.Context, agentID *uuid.UUID) (map[string]interface{}, error) {
	query := `SELECT 
		COUNT(*) as total,
		SUM(CASE WHEN feedback_type = 'positive' THEN 1 ELSE 0 END) as positive_count,
		SUM(CASE WHEN feedback_type = 'negative' THEN 1 ELSE 0 END) as negative_count,
		SUM(CASE WHEN feedback_type = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
		AVG(rating) as average_rating
		FROM neurondb_agent.user_feedback
		WHERE $1::uuid IS NULL OR agent_id = $1`

	var stats struct {
		Total         int      `db:"total"`
		PositiveCount int      `db:"positive_count"`
		NegativeCount int      `db:"negative_count"`
		NeutralCount  int      `db:"neutral_count"`
		AverageRating *float64 `db:"average_rating"`
	}

	err := fm.db.GetContext(ctx, &stats, query, agentID)
	if err != nil {
		return nil, fmt.Errorf("failed to get feedback stats: %w", err)
	}

	result := map[string]interface{}{
		"total":          stats.Total,
		"positive_count": stats.PositiveCount,
		"negative_count": stats.NegativeCount,
		"neutral_count":  stats.NeutralCount,
	}

	if stats.AverageRating != nil {
		result["average_rating"] = *stats.AverageRating
	}

	return result, nil
}
