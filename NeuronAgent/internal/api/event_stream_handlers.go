/*-------------------------------------------------------------------------
 *
 * event_stream_handlers.go
 *    Event Stream API handlers for NeuronAgent
 *
 * Provides REST API endpoints for event stream operations including
 * event logging, retrieval, summarization, and context window management.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/event_stream_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/validation"
)

/* EventStreamHandlers handles event stream API requests */
type EventStreamHandlers struct {
	queries *db.Queries
	runtime *agent.Runtime
}

/* NewEventStreamHandlers creates new event stream handlers */
func NewEventStreamHandlers(queries *db.Queries, runtime *agent.Runtime) *EventStreamHandlers {
	return &EventStreamHandlers{
		queries: queries,
		runtime: runtime,
	}
}

/* LogEventRequest represents a request to log an event */
type LogEventRequest struct {
	EventType string                 `json:"event_type"` // user_message, agent_action, tool_execution, agent_response, error, system
	Actor     string                 `json:"actor"`
	Content   string                 `json:"content"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

/* EventResponse represents an event in API responses */
type EventResponse struct {
	ID        string                 `json:"id"`
	SessionID string                 `json:"session_id"`
	EventType string                 `json:"event_type"`
	Actor     string                 `json:"actor"`
	Content   string                 `json:"content"`
	Metadata  map[string]interface{} `json:"metadata"`
	Timestamp time.Time              `json:"timestamp"`
}

/* EventSummaryResponse represents an event summary */
type EventSummaryResponse struct {
	ID           string    `json:"id"`
	SessionID    string    `json:"session_id"`
	StartEventID string    `json:"start_event_id"`
	EndEventID   string    `json:"end_event_id"`
	EventCount   int       `json:"event_count"`
	SummaryText  string    `json:"summary_text"`
	CreatedAt    time.Time `json:"created_at"`
}

/* LogEvent logs a new event to the stream */
func (h *EventStreamHandlers) LogEvent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate session ID */
	if err := validation.ValidateUUIDRequired(vars["session_id"], "session_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	sessionID, err := uuid.Parse(vars["session_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID format", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	/* Validate request body size */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req LogEventRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	/* Validate event type */
	validTypes := map[string]bool{
		"user_message":   true,
		"agent_action":   true,
		"tool_execution": true,
		"agent_response": true,
		"error":          true,
		"system":         true,
	}
	if !validTypes[req.EventType] {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid event_type", nil, requestID, r.URL.Path, r.Method, "event_stream", "", map[string]interface{}{
			"valid_types": []string{"user_message", "agent_action", "tool_execution", "agent_response", "error", "system"},
		}))
		return
	}

	/* Validate required fields */
	if req.Actor == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "actor is required", nil, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}
	if req.Content == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "content is required", nil, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	/* Get event stream manager from runtime */
	eventStream := h.runtime.EventStream()
	if eventStream == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "event stream manager not available", nil, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	/* Log event */
	eventID, err := eventStream.LogEvent(r.Context(), sessionID, req.EventType, req.Actor, req.Content, req.Metadata)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to log event", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	respondJSON(w, http.StatusCreated, map[string]interface{}{
		"id":         eventID.String(),
		"session_id": sessionID.String(),
	})
}

/* GetEventHistory retrieves event history for a session */
func (h *EventStreamHandlers) GetEventHistory(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate session ID */
	if err := validation.ValidateUUIDRequired(vars["session_id"], "session_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	sessionID, err := uuid.Parse(vars["session_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID format", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	/* Get limit from query parameter */
	limit := 100
	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		parsedLimit, err := strconv.Atoi(limitStr)
		if err == nil && parsedLimit > 0 && parsedLimit <= 1000 {
			limit = parsedLimit
		}
	}

	/* Get event stream manager */
	eventStream := h.runtime.EventStream()
	if eventStream == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "event stream manager not available", nil, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	/* Get event history */
	events, err := eventStream.GetEventHistory(r.Context(), sessionID, limit)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to retrieve event history", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	/* Convert to response format */
	responses := make([]EventResponse, len(events))
	for i, event := range events {
		responses[i] = EventResponse{
			ID:        event.ID.String(),
			SessionID: event.SessionID.String(),
			EventType: event.EventType,
			Actor:     event.Actor,
			Content:   event.Content,
			Metadata:  event.Metadata,
			Timestamp: event.Timestamp,
		}
	}

	respondJSON(w, http.StatusOK, responses)
}

/* SummarizeEventsRequest represents a request to summarize events */
type SummarizeEventsRequest struct {
	StartEventID string `json:"start_event_id"`
	EndEventID   string `json:"end_event_id"`
}

/* SummarizeEvents creates a summary of events in a range */
func (h *EventStreamHandlers) SummarizeEvents(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate session ID */
	if err := validation.ValidateUUIDRequired(vars["session_id"], "session_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	sessionID, err := uuid.Parse(vars["session_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID format", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	/* Validate request body */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req SummarizeEventsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	/* Validate event IDs */
	startEventID, err := uuid.Parse(req.StartEventID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid start_event_id format", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	endEventID, err := uuid.Parse(req.EndEventID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid end_event_id format", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	/* Get event stream manager */
	eventStream := h.runtime.EventStream()
	if eventStream == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "event stream manager not available", nil, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	/* Create summary */
	summaryID, err := eventStream.SummarizeEvents(r.Context(), sessionID, startEventID, endEventID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to create event summary", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	respondJSON(w, http.StatusCreated, map[string]interface{}{
		"id":         summaryID.String(),
		"session_id": sessionID.String(),
	})
}

/* GetContextWindow retrieves recent events and summaries for context building */
func (h *EventStreamHandlers) GetContextWindow(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate session ID */
	if err := validation.ValidateUUIDRequired(vars["session_id"], "session_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	sessionID, err := uuid.Parse(vars["session_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID format", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	/* Get maxEvents from query parameter */
	maxEvents := 50
	if maxEventsStr := r.URL.Query().Get("max_events"); maxEventsStr != "" {
		parsedMaxEvents, err := strconv.Atoi(maxEventsStr)
		if err == nil && parsedMaxEvents > 0 && parsedMaxEvents <= 500 {
			maxEvents = parsedMaxEvents
		}
	}

	/* Get event stream manager */
	eventStream := h.runtime.EventStream()
	if eventStream == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "event stream manager not available", nil, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	/* Get context window */
	events, summaries, err := eventStream.GetContextWindow(r.Context(), sessionID, maxEvents)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to retrieve context window", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	/* Convert events to response format */
	eventResponses := make([]EventResponse, len(events))
	for i, event := range events {
		eventResponses[i] = EventResponse{
			ID:        event.ID.String(),
			SessionID: event.SessionID.String(),
			EventType: event.EventType,
			Actor:     event.Actor,
			Content:   event.Content,
			Metadata:  event.Metadata,
			Timestamp: event.Timestamp,
		}
	}

	/* Convert summaries to response format */
	summaryResponses := make([]EventSummaryResponse, len(summaries))
	for i, summary := range summaries {
		summaryResponses[i] = EventSummaryResponse{
			ID:           summary.ID.String(),
			SessionID:    summary.SessionID.String(),
			StartEventID: summary.StartEventID.String(),
			EndEventID:   summary.EndEventID.String(),
			EventCount:   summary.EventCount,
			SummaryText:  summary.SummaryText,
			CreatedAt:    summary.CreatedAt,
		}
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"events":    eventResponses,
		"summaries": summaryResponses,
	})
}

/* GetEventCount returns total event count for a session */
func (h *EventStreamHandlers) GetEventCount(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate session ID */
	if err := validation.ValidateUUIDRequired(vars["session_id"], "session_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	sessionID, err := uuid.Parse(vars["session_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID format", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	/* Get event stream manager */
	eventStream := h.runtime.EventStream()
	if eventStream == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "event stream manager not available", nil, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	/* Get event count */
	count, err := eventStream.GetEventCount(r.Context(), sessionID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to get event count", err, requestID, r.URL.Path, r.Method, "event_stream", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"session_id": sessionID.String(),
		"count":      count,
	})
}








