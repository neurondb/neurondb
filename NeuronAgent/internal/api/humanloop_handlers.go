/*-------------------------------------------------------------------------
 *
 * humanloop_handlers.go
 *    API handlers for human-in-the-loop features
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/humanloop_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/humanloop"
)

/* ListApprovalRequests lists approval requests */
func (h *Handlers) ListApprovalRequests(w http.ResponseWriter, r *http.Request) {
	var agentID *uuid.UUID
	if agentIDStr := r.URL.Query().Get("agent_id"); agentIDStr != "" {
		id, err := uuid.Parse(agentIDStr)
		if err == nil {
			agentID = &id
		}
	}

	limit := 50
	offset := 0
	if l := r.URL.Query().Get("limit"); l != "" {
		fmt.Sscanf(l, "%d", &limit)
	}
	if o := r.URL.Query().Get("offset"); o != "" {
		fmt.Sscanf(o, "%d", &offset)
	}

	approvalMgr := humanloop.NewApprovalManager(h.queries.DB)
	reqs, err := approvalMgr.ListPendingApprovals(r.Context(), agentID, limit, offset)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list approval requests", err, requestID, r.URL.Path, r.Method, "approval", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, reqs)
}

/* GetApprovalRequest gets an approval request */
func (h *Handlers) GetApprovalRequest(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid approval request id", err, requestID, r.URL.Path, r.Method, "approval", "", nil))
		return
	}

	approvalMgr := humanloop.NewApprovalManager(h.queries.DB)
	req, err := approvalMgr.GetApprovalRequest(r.Context(), id)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "approval request not found", err, requestID, r.URL.Path, r.Method, "approval", id.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, req)
}

/* ApproveRequest approves an approval request */
func (h *Handlers) ApproveRequest(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid approval request id", err, requestID, r.URL.Path, r.Method, "approval", "", nil))
		return
	}

	var req struct {
		ApprovedBy string `json:"approved_by"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "approval", id.String(), nil))
		return
	}

	if req.ApprovedBy == "" {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "approved_by is required", nil, requestID, r.URL.Path, r.Method, "approval", id.String(), nil))
		return
	}

	approvalMgr := humanloop.NewApprovalManager(h.queries.DB)
	if err := approvalMgr.ApproveRequest(r.Context(), id, req.ApprovedBy); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to approve request", err, requestID, r.URL.Path, r.Method, "approval", id.String(), nil))
		return
	}

	w.WriteHeader(http.StatusOK)
}

/* RejectRequest rejects an approval request */
func (h *Handlers) RejectRequest(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid approval request id", err, requestID, r.URL.Path, r.Method, "approval", "", nil))
		return
	}

	var req struct {
		RejectedBy string `json:"rejected_by"`
		Reason     string `json:"reason"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "approval", id.String(), nil))
		return
	}

	if req.RejectedBy == "" {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "rejected_by is required", nil, requestID, r.URL.Path, r.Method, "approval", id.String(), nil))
		return
	}

	approvalMgr := humanloop.NewApprovalManager(h.queries.DB)
	if err := approvalMgr.RejectRequest(r.Context(), id, req.RejectedBy, req.Reason); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to reject request", err, requestID, r.URL.Path, r.Method, "approval", id.String(), nil))
		return
	}

	w.WriteHeader(http.StatusOK)
}

/* SubmitFeedback submits user feedback */
func (h *Handlers) SubmitFeedback(w http.ResponseWriter, r *http.Request) {
	var req struct {
		AgentID      *uuid.UUID             `json:"agent_id"`
		SessionID    *uuid.UUID             `json:"session_id"`
		MessageID    *int64                 `json:"message_id"`
		UserID       *string                `json:"user_id"`
		FeedbackType string                 `json:"feedback_type"`
		Rating       *int                   `json:"rating"`
		Comment      *string                `json:"comment"`
		Metadata     map[string]interface{} `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "feedback", "", nil))
		return
	}

	if req.FeedbackType == "" {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "feedback_type is required", nil, requestID, r.URL.Path, r.Method, "feedback", "", nil))
		return
	}

	validTypes := map[string]bool{
		"positive":   true,
		"negative":   true,
		"neutral":    true,
		"correction": true,
	}
	if !validTypes[req.FeedbackType] {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid feedback_type", nil, requestID, r.URL.Path, r.Method, "feedback", "", nil))
		return
	}

	var metadata map[string]interface{}
	if req.Metadata != nil {
		metadata = req.Metadata
	} else {
		metadata = make(map[string]interface{})
	}

	metadataJSON := db.FromMap(metadata)
	if metadataJSON == nil {
		metadataJSON = make(db.JSONBMap)
	}

	feedback := &humanloop.UserFeedback{
		AgentID:      req.AgentID,
		SessionID:    req.SessionID,
		MessageID:    req.MessageID,
		UserID:       req.UserID,
		FeedbackType: req.FeedbackType,
		Rating:       req.Rating,
		Comment:      req.Comment,
		Metadata:     metadataJSON,
	}

	feedbackMgr := humanloop.NewFeedbackManager(h.queries.DB)
	if err := feedbackMgr.SubmitFeedback(r.Context(), feedback); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to submit feedback", err, requestID, r.URL.Path, r.Method, "feedback", "", nil))
		return
	}

	respondJSON(w, http.StatusCreated, feedback)
}

/* ListFeedback lists user feedback */
func (h *Handlers) ListFeedback(w http.ResponseWriter, r *http.Request) {
	var agentID, sessionID *uuid.UUID
	var feedbackType *string

	if agentIDStr := r.URL.Query().Get("agent_id"); agentIDStr != "" {
		id, err := uuid.Parse(agentIDStr)
		if err == nil {
			agentID = &id
		}
	}

	if sessionIDStr := r.URL.Query().Get("session_id"); sessionIDStr != "" {
		id, err := uuid.Parse(sessionIDStr)
		if err == nil {
			sessionID = &id
		}
	}

	if ft := r.URL.Query().Get("feedback_type"); ft != "" {
		feedbackType = &ft
	}

	limit := 50
	offset := 0
	if l := r.URL.Query().Get("limit"); l != "" {
		fmt.Sscanf(l, "%d", &limit)
	}
	if o := r.URL.Query().Get("offset"); o != "" {
		fmt.Sscanf(o, "%d", &offset)
	}

	feedbackMgr := humanloop.NewFeedbackManager(h.queries.DB)
	feedbacks, err := feedbackMgr.ListFeedback(r.Context(), agentID, sessionID, feedbackType, limit, offset)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list feedback", err, requestID, r.URL.Path, r.Method, "feedback", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, feedbacks)
}

/* GetFeedbackStats gets feedback statistics */
func (h *Handlers) GetFeedbackStats(w http.ResponseWriter, r *http.Request) {
	var agentID *uuid.UUID
	if agentIDStr := r.URL.Query().Get("agent_id"); agentIDStr != "" {
		id, err := uuid.Parse(agentIDStr)
		if err == nil {
			agentID = &id
		}
	}

	feedbackMgr := humanloop.NewFeedbackManager(h.queries.DB)
	stats, err := feedbackMgr.GetFeedbackStats(r.Context(), agentID)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to get feedback stats", err, requestID, r.URL.Path, r.Method, "feedback", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, stats)
}
