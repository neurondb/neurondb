/*-------------------------------------------------------------------------
 *
 * verification_handlers.go
 *    Verification Agent API handlers for NeuronAgent
 *
 * Provides REST API endpoints for verification operations including
 * queueing verifications, retrieving results, and managing verification rules.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/verification_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/validation"
)

/* VerificationHandlers handles verification API requests */
type VerificationHandlers struct {
	queries *db.Queries
	runtime *agent.Runtime
}

/* NewVerificationHandlers creates new verification handlers */
func NewVerificationHandlers(queries *db.Queries, runtime *agent.Runtime) *VerificationHandlers {
	return &VerificationHandlers{
		queries: queries,
		runtime: runtime,
	}
}

/* QueueVerificationRequest represents a request to queue a verification */
type QueueVerificationRequest struct {
	SessionID     string  `json:"session_id"`
	OutputID      *string `json:"output_id,omitempty"`
	OutputContent string  `json:"output_content"`
	Priority      string  `json:"priority,omitempty"` // low, medium, high
}

/* VerificationResultResponse represents a verification result */
type VerificationResultResponse struct {
	ID           string                 `json:"id"`
	QueueID      string                 `json:"queue_id"`
	VerifierID   *string                `json:"verifier_agent_id,omitempty"`
	Passed       bool                   `json:"passed"`
	Issues       []map[string]interface{} `json:"issues"`
	Suggestions  []string               `json:"suggestions"`
	Confidence   float64                `json:"confidence"`
	VerifiedAt   string                 `json:"verified_at"`
}

/* VerificationRuleRequest represents a verification rule */
type VerificationRuleRequest struct {
	RuleType string                 `json:"rule_type"` // output_format, data_accuracy, logical_consistency, completeness
	Criteria map[string]interface{} `json:"criteria"`
	Enabled  bool                   `json:"enabled"`
}

/* VerificationRuleResponse represents a verification rule in API responses */
type VerificationRuleResponse struct {
	ID        string                 `json:"id"`
	AgentID   string                 `json:"agent_id"`
	RuleType  string                 `json:"rule_type"`
	Criteria  map[string]interface{} `json:"criteria"`
	Enabled   bool                   `json:"enabled"`
	CreatedAt string                 `json:"created_at"`
	UpdatedAt string                 `json:"updated_at"`
}

/* QueueVerification queues an output for verification */
func (h *VerificationHandlers) QueueVerification(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate agent ID */
	if err := validation.ValidateUUIDRequired(vars["agent_id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["agent_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	/* Validate request body */
	const maxBodySize = 10 * 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req QueueVerificationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	/* Validate required fields */
	if req.SessionID == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "session_id is required", nil, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}
	if req.OutputContent == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "output_content is required", nil, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	/* Validate session ID */
	sessionID, err := uuid.Parse(req.SessionID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session_id format", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	/* Validate priority */
	priority := req.Priority
	if priority == "" {
		priority = "medium"
	}
	validPriorities := map[string]bool{"low": true, "medium": true, "high": true}
	if !validPriorities[priority] {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid priority", nil, requestID, r.URL.Path, r.Method, "verification", "", map[string]interface{}{
			"valid_priorities": []string{"low", "medium", "high"},
		}))
		return
	}

	/* Parse output ID if provided */
	var outputID *uuid.UUID
	if req.OutputID != nil {
		parsedOutputID, err := uuid.Parse(*req.OutputID)
		if err != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid output_id format", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
			return
		}
		outputID = &parsedOutputID
	}

	/* Get verifier from runtime */
	verifier := h.runtime.Verifier()
	if verifier == nil {
		/* Create verifier if not exists */
		verifier = agent.NewVerificationAgent(agentID, h.runtime, h.queries)
	}

	/* Queue verification */
	queueID, err := verifier.QueueVerification(r.Context(), sessionID, outputID, req.OutputContent, priority)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to queue verification", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	respondJSON(w, http.StatusCreated, map[string]interface{}{
		"queue_id":     queueID.String(),
		"session_id":   sessionID.String(),
		"agent_id":     agentID.String(),
		"status":       "pending",
		"priority":     priority,
	})
}

/* GetVerificationResults retrieves verification results for a queue item */
func (h *VerificationHandlers) GetVerificationResults(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate queue ID */
	if err := validation.ValidateUUIDRequired(vars["queue_id"], "queue_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid queue ID", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	queueID, err := uuid.Parse(vars["queue_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid queue ID format", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	/* Query verification results */
	query := `SELECT id, queue_id, verifier_agent_id, passed, issues, suggestions, confidence, verified_at
		FROM neurondb_agent.verification_results
		WHERE queue_id = $1
		ORDER BY verified_at DESC
		LIMIT 1`

	type ResultRow struct {
		ID             uuid.UUID              `db:"id"`
		QueueID        uuid.UUID              `db:"queue_id"`
		VerifierID     *uuid.UUID             `db:"verifier_agent_id"`
		Passed         bool                    `db:"passed"`
		Issues         map[string]interface{} `db:"issues"`
		Suggestions    []string                `db:"suggestions"`
		Confidence     float64                 `db:"confidence"`
		VerifiedAt     string                  `db:"verified_at"`
	}

	var row ResultRow
	err = h.queries.GetDB().GetContext(r.Context(), &row, query, queueID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusNotFound, "verification result not found", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	/* Convert issues to proper format */
	issues := []map[string]interface{}{}
	if row.Issues != nil {
		if issuesList, ok := row.Issues["issues"].([]interface{}); ok {
			for _, issue := range issuesList {
				if issueMap, ok := issue.(map[string]interface{}); ok {
					issues = append(issues, issueMap)
				}
			}
		}
	}

	verifierIDStr := ""
	if row.VerifierID != nil {
		verifierIDStr = row.VerifierID.String()
	}

	response := VerificationResultResponse{
		ID:          row.ID.String(),
		QueueID:     row.QueueID.String(),
		VerifierID:  &verifierIDStr,
		Passed:      row.Passed,
		Issues:      issues,
		Suggestions: row.Suggestions,
		Confidence:  row.Confidence,
		VerifiedAt:  row.VerifiedAt,
	}

	if verifierIDStr == "" {
		response.VerifierID = nil
	}

	respondJSON(w, http.StatusOK, response)
}

/* ListVerificationRules lists verification rules for an agent */
func (h *VerificationHandlers) ListVerificationRules(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate agent ID */
	if err := validation.ValidateUUIDRequired(vars["agent_id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["agent_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	/* Query verification rules */
	query := `SELECT id, agent_id, rule_type, criteria, enabled, created_at, updated_at
		FROM neurondb_agent.verification_rules
		WHERE agent_id = $1
		ORDER BY created_at DESC`

	type RuleRow struct {
		ID        uuid.UUID              `db:"id"`
		AgentID   uuid.UUID              `db:"agent_id"`
		RuleType  string                 `db:"rule_type"`
		Criteria  map[string]interface{} `db:"criteria"`
		Enabled   bool                   `db:"enabled"`
		CreatedAt string                 `db:"created_at"`
		UpdatedAt string                 `db:"updated_at"`
	}

	var rows []RuleRow
	err = h.queries.GetDB().SelectContext(r.Context(), &rows, query, agentID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list verification rules", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	/* Convert to response format */
	responses := make([]VerificationRuleResponse, len(rows))
	for i, row := range rows {
		responses[i] = VerificationRuleResponse{
			ID:        row.ID.String(),
			AgentID:   row.AgentID.String(),
			RuleType:  row.RuleType,
			Criteria:  row.Criteria,
			Enabled:   row.Enabled,
			CreatedAt: row.CreatedAt,
			UpdatedAt: row.UpdatedAt,
		}
	}

	respondJSON(w, http.StatusOK, responses)
}

/* CreateVerificationRule creates a new verification rule */
func (h *VerificationHandlers) CreateVerificationRule(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate agent ID */
	if err := validation.ValidateUUIDRequired(vars["agent_id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["agent_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	/* Validate request body */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req VerificationRuleRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	/* Validate rule type */
	validTypes := map[string]bool{
		"output_format":      true,
		"data_accuracy":      true,
		"logical_consistency": true,
		"completeness":       true,
	}
	if !validTypes[req.RuleType] {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid rule_type", nil, requestID, r.URL.Path, r.Method, "verification", "", map[string]interface{}{
			"valid_types": []string{"output_format", "data_accuracy", "logical_consistency", "completeness"},
		}))
		return
	}

	/* Insert verification rule */
	insertQuery := `INSERT INTO neurondb_agent.verification_rules
		(agent_id, rule_type, criteria, enabled)
		VALUES ($1, $2, $3::jsonb, $4)
		RETURNING id, created_at, updated_at`

	type RuleResult struct {
		ID        uuid.UUID `db:"id"`
		CreatedAt string    `db:"created_at"`
		UpdatedAt string    `db:"updated_at"`
	}

	var result RuleResult
	err = h.queries.GetDB().GetContext(r.Context(), &result, insertQuery, agentID, req.RuleType, req.Criteria, req.Enabled)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to create verification rule", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	respondJSON(w, http.StatusCreated, VerificationRuleResponse{
		ID:        result.ID.String(),
		AgentID:   agentID.String(),
		RuleType:  req.RuleType,
		Criteria:  req.Criteria,
		Enabled:   req.Enabled,
		CreatedAt: result.CreatedAt,
		UpdatedAt: result.UpdatedAt,
	})
}

/* UpdateVerificationRule updates a verification rule */
func (h *VerificationHandlers) UpdateVerificationRule(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate rule ID */
	if err := validation.ValidateUUIDRequired(vars["rule_id"], "rule_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid rule ID", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	ruleID, err := uuid.Parse(vars["rule_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid rule ID format", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	/* Validate request body */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req VerificationRuleRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	/* Update verification rule */
	updateQuery := `UPDATE neurondb_agent.verification_rules
		SET rule_type = $1, criteria = $2::jsonb, enabled = $3, updated_at = NOW()
		WHERE id = $4
		RETURNING agent_id, created_at, updated_at`

	type RuleResult struct {
		AgentID   uuid.UUID `db:"agent_id"`
		CreatedAt string    `db:"created_at"`
		UpdatedAt string    `db:"updated_at"`
	}

	var result RuleResult
	err = h.queries.GetDB().GetContext(r.Context(), &result, updateQuery, req.RuleType, req.Criteria, req.Enabled, ruleID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusNotFound, "verification rule not found", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, VerificationRuleResponse{
		ID:        ruleID.String(),
		AgentID:   result.AgentID.String(),
		RuleType:  req.RuleType,
		Criteria:  req.Criteria,
		Enabled:   req.Enabled,
		CreatedAt: result.CreatedAt,
		UpdatedAt: result.UpdatedAt,
	})
}

/* DeleteVerificationRule deletes a verification rule */
func (h *VerificationHandlers) DeleteVerificationRule(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate rule ID */
	if err := validation.ValidateUUIDRequired(vars["rule_id"], "rule_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid rule ID", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	ruleID, err := uuid.Parse(vars["rule_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid rule ID format", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	/* Delete verification rule */
	deleteQuery := `DELETE FROM neurondb_agent.verification_rules WHERE id = $1`
	result, err := h.queries.GetDB().ExecContext(r.Context(), deleteQuery, ruleID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to delete verification rule", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to delete verification rule", err, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	if rowsAffected == 0 {
		respondError(w, NewErrorWithContext(http.StatusNotFound, "verification rule not found", nil, requestID, r.URL.Path, r.Method, "verification", "", nil))
		return
	}

	w.WriteHeader(http.StatusNoContent)
}







