/*-------------------------------------------------------------------------
 *
 * event_stream.go
 *    Event stream architecture for chronological action logging
 *
 * Provides event logging, retrieval, summarization, and context management
 * for maintaining session history and building agent context windows.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/event_stream.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* EventStreamManager manages event stream operations */
type EventStreamManager struct {
	queries    *db.Queries
	summarizer *EventSummarizer
}

/* Event represents a single event in the stream */
type Event struct {
	ID        uuid.UUID
	SessionID uuid.UUID
	EventType string
	Actor     string
	Content   string
	Metadata  map[string]interface{}
	Timestamp time.Time
}

/* EventSummary represents a summary of multiple events */
type EventSummary struct {
	ID           uuid.UUID
	SessionID    uuid.UUID
	StartEventID uuid.UUID
	EndEventID   uuid.UUID
	EventCount   int
	SummaryText  string
	CreatedAt    time.Time
}

/* NewEventStreamManager creates a new event stream manager */
func NewEventStreamManager(queries *db.Queries, llmClient *LLMClient) *EventStreamManager {
	return &EventStreamManager{
		queries:    queries,
		summarizer: NewEventSummarizer(llmClient),
	}
}

/* LogEvent logs a new event to the stream */
func (e *EventStreamManager) LogEvent(ctx context.Context, sessionID uuid.UUID, eventType, actor, content string, metadata map[string]interface{}) (uuid.UUID, error) {
	/* Validate event type */
	validTypes := map[string]bool{
		"user_message":   true,
		"agent_action":   true,
		"tool_execution": true,
		"agent_response": true,
		"error":          true,
		"system":         true,
	}
	if !validTypes[eventType] {
		return uuid.Nil, fmt.Errorf("invalid event type: %s", eventType)
	}

	/* Create event record */
	query := `INSERT INTO neurondb_agent.event_stream
		(session_id, event_type, actor, content, metadata)
		VALUES ($1, $2, $3, $4, $5::jsonb)
		RETURNING id`

	var eventID uuid.UUID
	err := e.queries.GetDB().GetContext(ctx, &eventID, query, sessionID, eventType, actor, content, metadata)
	if err != nil {
		return uuid.Nil, fmt.Errorf("event logging failed: session_id=%s, event_type=%s, error=%w",
			sessionID.String(), eventType, err)
	}

	return eventID, nil
}

/* GetEventHistory retrieves event history for a session */
func (e *EventStreamManager) GetEventHistory(ctx context.Context, sessionID uuid.UUID, limit int) ([]Event, error) {
	query := `SELECT id, session_id, event_type, actor, content, metadata, created_at
		FROM neurondb_agent.event_stream
		WHERE session_id = $1
		ORDER BY created_at DESC
		LIMIT $2`

	type EventRow struct {
		ID        uuid.UUID              `db:"id"`
		SessionID uuid.UUID              `db:"session_id"`
		EventType string                 `db:"event_type"`
		Actor     string                 `db:"actor"`
		Content   string                 `db:"content"`
		Metadata  map[string]interface{} `db:"metadata"`
		CreatedAt time.Time              `db:"created_at"`
	}

	var rows []EventRow
	err := e.queries.GetDB().SelectContext(ctx, &rows, query, sessionID, limit)
	if err != nil {
		return nil, fmt.Errorf("event history retrieval failed: session_id=%s, limit=%d, error=%w",
			sessionID.String(), limit, err)
	}

	events := make([]Event, len(rows))
	for i, row := range rows {
		events[i] = Event{
			ID:        row.ID,
			SessionID: row.SessionID,
			EventType: row.EventType,
			Actor:     row.Actor,
			Content:   row.Content,
			Metadata:  row.Metadata,
			Timestamp: row.CreatedAt,
		}
	}

	return events, nil
}

/* SummarizeEvents creates a summary of events in a range */
func (e *EventStreamManager) SummarizeEvents(ctx context.Context, sessionID uuid.UUID, startEventID, endEventID uuid.UUID) (uuid.UUID, error) {
	/* Retrieve events in range */
	query := `SELECT id, event_type, actor, content, created_at
		FROM neurondb_agent.event_stream
		WHERE session_id = $1
		AND created_at >= (SELECT created_at FROM neurondb_agent.event_stream WHERE id = $2)
		AND created_at <= (SELECT created_at FROM neurondb_agent.event_stream WHERE id = $3)
		ORDER BY created_at ASC`

	type EventRow struct {
		ID        uuid.UUID `db:"id"`
		EventType string    `db:"event_type"`
		Actor     string    `db:"actor"`
		Content   string    `db:"content"`
		CreatedAt time.Time `db:"created_at"`
	}

	var events []EventRow
	err := e.queries.GetDB().SelectContext(ctx, &events, query, sessionID, startEventID, endEventID)
	if err != nil {
		return uuid.Nil, fmt.Errorf("event range retrieval failed: session_id=%s, error=%w",
			sessionID.String(), err)
	}

	if len(events) == 0 {
		return uuid.Nil, fmt.Errorf("no events found in range")
	}

	/* Generate summary using LLM */
	summaryText, err := e.summarizer.SummarizeEvents(ctx, events)
	if err != nil {
		return uuid.Nil, fmt.Errorf("event summarization failed: event_count=%d, error=%w",
			len(events), err)
	}

	/* Store summary */
	insertQuery := `INSERT INTO neurondb_agent.event_summaries
		(session_id, start_event_id, end_event_id, event_count, summary_text)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id`

	var summaryID uuid.UUID
	err = e.queries.GetDB().GetContext(ctx, &summaryID, insertQuery,
		sessionID, startEventID, endEventID, len(events), summaryText)
	if err != nil {
		return uuid.Nil, fmt.Errorf("summary storage failed: error=%w", err)
	}

	return summaryID, nil
}

/* GetContextWindow retrieves recent events for context building */
func (e *EventStreamManager) GetContextWindow(ctx context.Context, sessionID uuid.UUID, maxEvents int) ([]Event, []EventSummary, error) {
	/* Get recent events */
	events, err := e.GetEventHistory(ctx, sessionID, maxEvents)
	if err != nil {
		return nil, nil, err
	}

	/* Get recent summaries */
	summaryQuery := `SELECT id, session_id, start_event_id, end_event_id, event_count, summary_text, created_at
		FROM neurondb_agent.event_summaries
		WHERE session_id = $1
		ORDER BY created_at DESC
		LIMIT 5`

	type SummaryRow struct {
		ID           uuid.UUID `db:"id"`
		SessionID    uuid.UUID `db:"session_id"`
		StartEventID uuid.UUID `db:"start_event_id"`
		EndEventID   uuid.UUID `db:"end_event_id"`
		EventCount   int       `db:"event_count"`
		SummaryText  string    `db:"summary_text"`
		CreatedAt    time.Time `db:"created_at"`
	}

	var summaryRows []SummaryRow
	err = e.queries.GetDB().SelectContext(ctx, &summaryRows, summaryQuery, sessionID)
	if err != nil {
		return events, nil, nil /* Return events even if summaries fail */
	}

	summaries := make([]EventSummary, len(summaryRows))
	for i, row := range summaryRows {
		summaries[i] = EventSummary{
			ID:           row.ID,
			SessionID:    row.SessionID,
			StartEventID: row.StartEventID,
			EndEventID:   row.EndEventID,
			EventCount:   row.EventCount,
			SummaryText:  row.SummaryText,
			CreatedAt:    row.CreatedAt,
		}
	}

	return events, summaries, nil
}

/* ArchiveOldEvents archives events older than specified timestamp */
func (e *EventStreamManager) ArchiveOldEvents(ctx context.Context, sessionID uuid.UUID, beforeTimestamp time.Time) (int, error) {
	query := `DELETE FROM neurondb_agent.event_stream
		WHERE session_id = $1
		AND created_at < $2
		RETURNING id`

	var deletedIDs []uuid.UUID
	err := e.queries.GetDB().SelectContext(ctx, &deletedIDs, query, sessionID, beforeTimestamp)
	if err != nil {
		return 0, fmt.Errorf("event archival failed: session_id=%s, error=%w",
			sessionID.String(), err)
	}

	return len(deletedIDs), nil
}

/* GetEventCount returns total event count for a session */
func (e *EventStreamManager) GetEventCount(ctx context.Context, sessionID uuid.UUID) (int, error) {
	query := `SELECT COUNT(*) FROM neurondb_agent.event_stream WHERE session_id = $1`

	var count int
	err := e.queries.GetDB().GetContext(ctx, &count, query, sessionID)
	if err != nil {
		return 0, fmt.Errorf("event count retrieval failed: session_id=%s, error=%w",
			sessionID.String(), err)
	}

	return count, nil
}

/* CheckSummarizationNeeded checks if summarization should be triggered */
func (e *EventStreamManager) CheckSummarizationNeeded(ctx context.Context, sessionID uuid.UUID, threshold int) (bool, uuid.UUID, uuid.UUID, error) {
	/* Get unsummarized event count */
	query := `SELECT id, created_at FROM neurondb_agent.event_stream
		WHERE session_id = $1
		AND id NOT IN (
			SELECT start_event_id FROM neurondb_agent.event_summaries WHERE session_id = $1
			UNION
			SELECT end_event_id FROM neurondb_agent.event_summaries WHERE session_id = $1
		)
		ORDER BY created_at ASC`

	type EventRow struct {
		ID        uuid.UUID `db:"id"`
		CreatedAt time.Time `db:"created_at"`
	}

	var events []EventRow
	err := e.queries.GetDB().SelectContext(ctx, &events, query, sessionID)
	if err != nil {
		return false, uuid.Nil, uuid.Nil, err
	}

	if len(events) >= threshold {
		return true, events[0].ID, events[len(events)-1].ID, nil
	}

	return false, uuid.Nil, uuid.Nil, nil
}
