/*-------------------------------------------------------------------------
 *
 * reflection_queries.go
 *    Database queries for reflections
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/reflection_queries.go
 *
 *-------------------------------------------------------------------------
 */

package db

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/google/uuid"
)

/* Reflection queries */
const (
	createReflectionQuery = `
		INSERT INTO neurondb_agent.reflections 
		(agent_id, session_id, message_id, user_message, agent_response, quality_score, accuracy_score, 
		 completeness_score, clarity_score, relevance_score, confidence, issues, suggestions, was_retried)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb, $13::jsonb, $14)
		RETURNING id, created_at`

	getReflectionQuery = `SELECT * FROM neurondb_agent.reflections WHERE id = $1`

	listReflectionsQuery = `
		SELECT * FROM neurondb_agent.reflections 
		WHERE ($1::uuid IS NULL OR agent_id = $1)
		AND ($2::uuid IS NULL OR session_id = $2)
		ORDER BY created_at DESC 
		LIMIT $3 OFFSET $4`
)

/* Reflection represents a reflection record */
type Reflection struct {
	ID                int64      `db:"id"`
	AgentID           *uuid.UUID `db:"agent_id"`
	SessionID         *uuid.UUID `db:"session_id"`
	MessageID         *int64     `db:"message_id"`
	UserMessage       string     `db:"user_message"`
	AgentResponse     string     `db:"agent_response"`
	QualityScore      *float64   `db:"quality_score"`
	AccuracyScore     *float64   `db:"accuracy_score"`
	CompletenessScore *float64   `db:"completeness_score"`
	ClarityScore      *float64   `db:"clarity_score"`
	RelevanceScore    *float64   `db:"relevance_score"`
	Confidence        *float64   `db:"confidence"`
	Issues            JSONBMap   `db:"issues"`
	Suggestions       JSONBMap   `db:"suggestions"`
	WasRetried        bool       `db:"was_retried"`
	CreatedAt         string     `db:"created_at"`
}

/* CreateReflection creates a new reflection */
func (q *Queries) CreateReflection(ctx context.Context, reflection *Reflection) error {
	params := []interface{}{
		reflection.AgentID, reflection.SessionID, reflection.MessageID,
		reflection.UserMessage, reflection.AgentResponse,
		reflection.QualityScore, reflection.AccuracyScore, reflection.CompletenessScore,
		reflection.ClarityScore, reflection.RelevanceScore, reflection.Confidence,
		reflection.Issues, reflection.Suggestions, reflection.WasRetried,
	}
	err := q.DB.GetContext(ctx, reflection, createReflectionQuery, params...)
	if err != nil {
		return q.formatQueryError("INSERT", createReflectionQuery, len(params), "neurondb_agent.reflections", err)
	}
	return nil
}

/* GetReflection gets a reflection by ID */
func (q *Queries) GetReflection(ctx context.Context, id int64) (*Reflection, error) {
	var reflection Reflection
	err := q.DB.GetContext(ctx, &reflection, getReflectionQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("reflection not found on %s: query='%s', reflection_id=%d, table='neurondb_agent.reflections', error=%w",
			q.getConnInfoString(), getReflectionQuery, id, err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getReflectionQuery, 1, "neurondb_agent.reflections", err)
	}
	return &reflection, nil
}

/* ListReflections lists reflections with optional filters */
func (q *Queries) ListReflections(ctx context.Context, agentID, sessionID *uuid.UUID, limit, offset int) ([]Reflection, error) {
	var reflections []Reflection
	params := []interface{}{agentID, sessionID, limit, offset}
	err := q.DB.SelectContext(ctx, &reflections, listReflectionsQuery, params...)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listReflectionsQuery, len(params), "neurondb_agent.reflections", err)
	}
	return reflections, nil
}
