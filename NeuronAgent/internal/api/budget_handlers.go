/*-------------------------------------------------------------------------
 *
 * budget_handlers.go
 *    API handlers for budget management
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/budget_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* GetBudget gets the budget status for an agent */
func (h *Handlers) GetBudget(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id", err, requestID, r.URL.Path, r.Method, "budget", "", nil))
		return
	}

	periodType := r.URL.Query().Get("period_type")
	if periodType == "" {
		periodType = "monthly" /* Default to monthly */
	}

	validPeriods := map[string]bool{
		"daily":   true,
		"weekly":  true,
		"monthly": true,
		"yearly":  true,
		"total":   true,
	}
	if !validPeriods[periodType] {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid period_type", nil, requestID, r.URL.Path, r.Method, "budget", agentID.String(), nil))
		return
	}

	/* Get budget */
	budget, err := h.queries.GetBudget(r.Context(), agentID, periodType)
	if err != nil {
		/* Return empty budget status if not found */
		status := map[string]interface{}{
			"agent_id":    agentID,
			"period_type": periodType,
			"budget_set":  false,
		}
		respondJSON(w, http.StatusOK, status)
		return
	}

	/* Get cost summary */
	costTracker := agent.NewCostTracker(h.queries)
	var startDate, endDate time.Time
	now := time.Now()

	switch periodType {
	case "daily":
		startDate = time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, now.Location())
		endDate = now
	case "weekly":
		weekday := int(now.Weekday())
		if weekday == 0 {
			weekday = 7
		}
		startDate = now.AddDate(0, 0, -weekday+1)
		startDate = time.Date(startDate.Year(), startDate.Month(), startDate.Day(), 0, 0, 0, 0, startDate.Location())
		endDate = now
	case "monthly":
		startDate = time.Date(now.Year(), now.Month(), 1, 0, 0, 0, 0, now.Location())
		endDate = now
	case "yearly":
		startDate = time.Date(now.Year(), 1, 1, 0, 0, 0, 0, now.Location())
		endDate = now
	default:
		startDate = budget.StartDate
		if budget.EndDate != nil {
			endDate = *budget.EndDate
		} else {
			endDate = now
		}
	}

	costSummary, err := costTracker.GetCostSummary(r.Context(), agentID, startDate, endDate)
	if err != nil {
		costSummary = &agent.CostSummary{
			TotalCost:   0,
			TotalTokens: 0,
		}
	}

	remaining := budget.BudgetAmount - costSummary.TotalCost
	if remaining < 0 {
		remaining = 0
	}

	status := map[string]interface{}{
		"agent_id":      agentID,
		"period_type":   periodType,
		"budget_amount": budget.BudgetAmount,
		"total_cost":    costSummary.TotalCost,
		"remaining":     remaining,
		"within_budget": costSummary.TotalCost < budget.BudgetAmount,
		"budget_set":    true,
		"start_date":    startDate,
		"end_date":      endDate,
	}

	respondJSON(w, http.StatusOK, status)
}

/* SetBudget sets a budget for an agent */
func (h *Handlers) SetBudget(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id", err, requestID, r.URL.Path, r.Method, "budget", "", nil))
		return
	}

	/* Verify agent exists */
	_, err = h.queries.GetAgentByID(r.Context(), agentID)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "agent not found", err, requestID, r.URL.Path, r.Method, "budget", agentID.String(), nil))
		return
	}

	var req struct {
		BudgetAmount float64                `json:"budget_amount"`
		PeriodType   string                 `json:"period_type"`
		StartDate    *time.Time             `json:"start_date"`
		EndDate      *time.Time             `json:"end_date"`
		Metadata     map[string]interface{} `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "budget", agentID.String(), nil))
		return
	}

	if req.BudgetAmount < 0 {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "budget_amount must be >= 0", nil, requestID, r.URL.Path, r.Method, "budget", agentID.String(), nil))
		return
	}

	validPeriods := map[string]bool{
		"daily":   true,
		"weekly":  true,
		"monthly": true,
		"yearly":  true,
		"total":   true,
	}
	if !validPeriods[req.PeriodType] {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid period_type", nil, requestID, r.URL.Path, r.Method, "budget", agentID.String(), nil))
		return
	}

	startDate := time.Now()
	if req.StartDate != nil {
		startDate = *req.StartDate
	}

	var metadata db.JSONBMap
	if req.Metadata != nil {
		metadata = db.FromMap(req.Metadata)
	} else {
		metadata = make(db.JSONBMap)
	}

	budget := &db.AgentBudget{
		AgentID:      agentID,
		BudgetAmount: req.BudgetAmount,
		PeriodType:   req.PeriodType,
		StartDate:    startDate,
		EndDate:      req.EndDate,
		IsActive:     true,
		Metadata:     metadata,
	}

	if err := h.queries.CreateBudget(r.Context(), budget); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to set budget", err, requestID, r.URL.Path, r.Method, "budget", agentID.String(), nil))
		return
	}

	respondJSON(w, http.StatusCreated, budget)
}

/* UpdateBudget updates an existing budget */
func (h *Handlers) UpdateBudget(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id", err, requestID, r.URL.Path, r.Method, "budget", "", nil))
		return
	}

	periodType := r.URL.Query().Get("period_type")
	if periodType == "" {
		periodType = "monthly"
	}

	/* Get existing budget */
	budget, err := h.queries.GetBudget(r.Context(), agentID, periodType)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "budget not found", err, requestID, r.URL.Path, r.Method, "budget", agentID.String(), nil))
		return
	}

	var req struct {
		BudgetAmount *float64               `json:"budget_amount"`
		StartDate    *time.Time             `json:"start_date"`
		EndDate      *time.Time             `json:"end_date"`
		Metadata     map[string]interface{} `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "budget", agentID.String(), nil))
		return
	}

	/* Update fields if provided */
	if req.BudgetAmount != nil {
		if *req.BudgetAmount < 0 {
			requestID := GetRequestID(r.Context())
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "budget_amount must be >= 0", nil, requestID, r.URL.Path, r.Method, "budget", agentID.String(), nil))
			return
		}
		budget.BudgetAmount = *req.BudgetAmount
	}
	if req.StartDate != nil {
		budget.StartDate = *req.StartDate
	}
	if req.EndDate != nil {
		budget.EndDate = req.EndDate
	}
	if req.Metadata != nil {
		budget.Metadata = db.FromMap(req.Metadata)
	}

	if err := h.queries.UpdateBudget(r.Context(), budget); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to update budget", err, requestID, r.URL.Path, r.Method, "budget", agentID.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, budget)
}


