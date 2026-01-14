/*-------------------------------------------------------------------------
 *
 * batch_handlers.go
 *    API handlers for batch operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/batch_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/validation"
)

/* BatchCreateAgents creates multiple agents with comprehensive validation */
func (h *Handlers) BatchCreateAgents(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())

	/* Validate request body size (max 10MB for batch operations) */
	const maxBodySize = 10 * 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var reqs []CreateAgentRequest
	if err := json.NewDecoder(r.Body).Decode(&reqs); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}

	/* Validate batch size */
	if len(reqs) == 0 {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "batch request must contain at least one agent", nil, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}
	if len(reqs) > 100 {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "batch request must not exceed 100 agents", nil, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}

	results := make([]BatchResult, 0, len(reqs))
	var agent *db.Agent
	for i, req := range reqs {
		/* Validate each request */
		if err := ValidateCreateAgentRequest(&req); err != nil {
			results = append(results, BatchResult{
				Index:   i,
				Success: false,
				Error:   err.Error(),
			})
			continue
		}

		/* Check for duplicate names in batch */
		for j := 0; j < i; j++ {
			if reqs[j].Name == req.Name {
				results = append(results, BatchResult{
					Index:   i,
					Success: false,
					Error:   fmt.Sprintf("duplicate agent name in batch: %s", req.Name),
				})
				goto next
			}
		}

		agent = &db.Agent{
			Name:         req.Name,
			Description:  req.Description,
			SystemPrompt: req.SystemPrompt,
			ModelName:    req.ModelName,
			MemoryTable:  req.MemoryTable,
			EnabledTools: req.EnabledTools,
			Config:       db.FromMap(req.Config),
		}

		if err := h.queries.CreateAgent(r.Context(), agent); err != nil {
			results = append(results, BatchResult{
				Index:   i,
				Success: false,
				Error:   err.Error(),
			})
			continue
		}

		results = append(results, BatchResult{
			Index:   i,
			Success: true,
			Data:    toAgentResponse(agent),
		})
	next:
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"results": results,
		"total":   len(reqs),
		"success": countSuccess(results),
		"failed":  countFailed(results),
	})
}

/* BatchDeleteAgents deletes multiple agents with validation */
func (h *Handlers) BatchDeleteAgents(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())

	/* Validate request body size (max 1MB) */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req struct {
		IDs []uuid.UUID `json:"ids"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}

	/* Validate batch size */
	if len(req.IDs) == 0 {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "batch request must contain at least one agent ID", nil, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}
	if len(req.IDs) > 100 {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "batch request must not exceed 100 agent IDs", nil, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}

	/* Validate UUIDs */
	for i, id := range req.IDs {
		if id == uuid.Nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, fmt.Sprintf("invalid UUID at index %d", i), nil, requestID, r.URL.Path, r.Method, "agent", "", nil))
			return
		}
	}

	results := make([]BatchResult, 0, len(req.IDs))
	for i, id := range req.IDs {
		if err := h.queries.DeleteAgent(r.Context(), id); err != nil {
			results = append(results, BatchResult{
				Index:   i,
				Success: false,
				Error:   err.Error(),
			})
			continue
		}

		results = append(results, BatchResult{
			Index:   i,
			Success: true,
		})
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"results": results,
		"total":   len(req.IDs),
		"success": countSuccess(results),
		"failed":  countFailed(results),
	})
}

/* BatchDeleteMessages deletes multiple messages with validation */
func (h *Handlers) BatchDeleteMessages(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())

	/* Validate request body size (max 1MB) */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "message", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req struct {
		IDs []int64 `json:"ids"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "message", "", nil))
		return
	}

	/* Validate batch size */
	if len(req.IDs) == 0 {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "batch request must contain at least one message ID", nil, requestID, r.URL.Path, r.Method, "message", "", nil))
		return
	}
	if len(req.IDs) > 100 {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "batch request must not exceed 100 message IDs", nil, requestID, r.URL.Path, r.Method, "message", "", nil))
		return
	}

	/* Validate IDs */
	for i, id := range req.IDs {
		if id <= 0 {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, fmt.Sprintf("invalid message ID at index %d: must be positive", i), nil, requestID, r.URL.Path, r.Method, "message", "", nil))
			return
		}
	}

	results := make([]BatchResult, 0, len(req.IDs))
	for i, id := range req.IDs {
		if err := h.queries.DeleteMessage(r.Context(), id); err != nil {
			results = append(results, BatchResult{
				Index:   i,
				Success: false,
				Error:   err.Error(),
			})
			continue
		}

		results = append(results, BatchResult{
			Index:   i,
			Success: true,
		})
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"results": results,
		"total":   len(req.IDs),
		"success": countSuccess(results),
		"failed":  countFailed(results),
	})
}

/* BatchDeleteTools deletes multiple tools with validation */
func (h *Handlers) BatchDeleteTools(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())

	/* Validate request body size (max 1MB) */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "tool", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req struct {
		Names []string `json:"names"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "tool", "", nil))
		return
	}

	/* Validate batch size */
	if len(req.Names) == 0 {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "batch request must contain at least one tool name", nil, requestID, r.URL.Path, r.Method, "tool", "", nil))
		return
	}
	if len(req.Names) > 100 {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "batch request must not exceed 100 tool names", nil, requestID, r.URL.Path, r.Method, "tool", "", nil))
		return
	}

	/* Validate tool names */
	for i, name := range req.Names {
		if name == "" {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, fmt.Sprintf("empty tool name at index %d", i), nil, requestID, r.URL.Path, r.Method, "tool", "", nil))
			return
		}
		if len(name) > 100 {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, fmt.Sprintf("tool name at index %d exceeds maximum length", i), nil, requestID, r.URL.Path, r.Method, "tool", "", nil))
			return
		}
	}

	results := make([]BatchResult, 0, len(req.Names))
	for i, name := range req.Names {
		if err := h.queries.DeleteTool(r.Context(), name); err != nil {
			results = append(results, BatchResult{
				Index:   i,
				Success: false,
				Error:   err.Error(),
			})
			continue
		}

		results = append(results, BatchResult{
			Index:   i,
			Success: true,
		})
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"results": results,
		"total":   len(req.Names),
		"success": countSuccess(results),
		"failed":  countFailed(results),
	})
}

/* BatchResult represents the result of a batch operation */
type BatchResult struct {
	Index   int         `json:"index"`
	Success bool        `json:"success"`
	Error   string      `json:"error,omitempty"`
	Data    interface{} `json:"data,omitempty"`
}

func countSuccess(results []BatchResult) int {
	count := 0
	for _, r := range results {
		if r.Success {
			count++
		}
	}
	return count
}

func countFailed(results []BatchResult) int {
	count := 0
	for _, r := range results {
		if !r.Success {
			count++
		}
	}
	return count
}
