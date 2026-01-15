/*-------------------------------------------------------------------------
 *
 * compliance_handlers.go
 *    API handlers for compliance features
 *
 * Provides REST API endpoints for compliance reporting and audit logs.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/compliance_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronAgent/internal/compliance"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* ComplianceHandlers provides compliance API handlers */
type ComplianceHandlers struct {
	queries            *db.Queries
	complianceFramework *compliance.ComplianceFramework
}

/* NewComplianceHandlers creates new compliance handlers */
func NewComplianceHandlers(queries *db.Queries) *ComplianceHandlers {
	return &ComplianceHandlers{
		queries:             queries,
		complianceFramework: compliance.NewComplianceFramework(queries),
	}
}

/* GenerateComplianceReport generates a compliance report */
func (h *ComplianceHandlers) GenerateComplianceReport(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	reportType := vars["type"]

	var req struct {
		StartDate string `json:"start_date"`
		EndDate   string `json:"end_date"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	startDate, err := time.Parse(time.RFC3339, req.StartDate)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid start_date", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	endDate, err := time.Parse(time.RFC3339, req.EndDate)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid end_date", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	report, err := h.complianceFramework.GenerateComplianceReport(r.Context(), reportType, startDate, endDate)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to generate compliance report", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, report)
}

/* GetAuditLogs retrieves audit logs */
func (h *ComplianceHandlers) GetAuditLogs(w http.ResponseWriter, r *http.Request) {
	/* Query parameters for filtering */
	eventType := r.URL.Query().Get("event_type")
	userID := r.URL.Query().Get("user_id")
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	offset, _ := strconv.Atoi(r.URL.Query().Get("offset"))

	if limit <= 0 {
		limit = 100
	}

	query := `SELECT id, event_type, user_id, resource_type, resource_id, action, timestamp, metadata, created_at
		FROM neurondb_agent.audit_logs
		WHERE ($1::text IS NULL OR event_type = $1)
		  AND ($2::text IS NULL OR user_id = $2)
		ORDER BY timestamp DESC
		LIMIT $3 OFFSET $4`

	type AuditLogRow struct {
		ID          string                 `json:"id"`
		EventType   string                 `json:"event_type"`
		UserID      string                 `json:"user_id"`
		ResourceType string                `json:"resource_type"`
		ResourceID   string                `json:"resource_id"`
		Action      string                 `json:"action"`
		Timestamp   time.Time              `json:"timestamp"`
		Metadata    map[string]interface{} `json:"metadata"`
		CreatedAt   time.Time              `json:"created_at"`
	}

	var rows []AuditLogRow
	err := h.queries.DB.SelectContext(r.Context(), &rows, query, eventType, userID, limit, offset)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to retrieve audit logs", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, rows)
}

