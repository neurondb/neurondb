/*-------------------------------------------------------------------------
 *
 * alert_preferences_handlers.go
 *    API handlers for task alert preferences
 *
 * Provides HTTP handlers for managing user alert preferences for task notifications.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/alert_preferences_handlers.go
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
)

type AlertPreferencesHandlers struct {
	queries *db.Queries
}

func NewAlertPreferencesHandlers(queries *db.Queries) *AlertPreferencesHandlers {
	return &AlertPreferencesHandlers{
		queries: queries,
	}
}

type SetAlertPreferencesRequest struct {
	UserID       *uuid.UUID `json:"user_id,omitempty"`
	AgentID      *uuid.UUID `json:"agent_id,omitempty"`
	AlertTypes   []string   `json:"alert_types"`
	Channels     []string   `json:"channels"`
	EmailAddress *string    `json:"email_address,omitempty"`
	WebhookURL   *string    `json:"webhook_url,omitempty"`
	Enabled      bool       `json:"enabled"`
}

/* SetAlertPreferences sets alert preferences for a user/agent */
func (h *AlertPreferencesHandlers) SetAlertPreferences(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())

	var req SetAlertPreferencesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, WrapError(NewError(http.StatusBadRequest, "alert preferences update failed: request body parsing error", err), requestID))
		return
	}

	if len(req.AlertTypes) == 0 {
		req.AlertTypes = []string{"completion", "failure"}
	}

	if len(req.Channels) == 0 {
		req.Channels = []string{"webhook"}
	}

	/* Validate alert types */
	validAlertTypes := map[string]bool{
		"completion": true,
		"failure":    true,
		"progress":   true,
		"milestone":  true,
	}
	for _, alertType := range req.AlertTypes {
		if !validAlertTypes[alertType] {
			respondError(w, WrapError(NewError(http.StatusBadRequest, "invalid alert_type: "+alertType, nil), requestID))
			return
		}
	}

	/* Validate channels */
	validChannels := map[string]bool{
		"email":   true,
		"webhook": true,
		"push":    true,
	}
	for _, channel := range req.Channels {
		if !validChannels[channel] {
			respondError(w, WrapError(NewError(http.StatusBadRequest, "invalid channel: "+channel, nil), requestID))
			return
		}
	}

	/* Upsert preferences */
	query := `INSERT INTO neurondb_agent.task_alert_preferences
		(user_id, agent_id, alert_types, channels, email_address, webhook_url, enabled, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
		ON CONFLICT (user_id, agent_id) DO UPDATE SET
			alert_types = EXCLUDED.alert_types,
			channels = EXCLUDED.channels,
			email_address = EXCLUDED.email_address,
			webhook_url = EXCLUDED.webhook_url,
			enabled = EXCLUDED.enabled,
			updated_at = NOW()
		RETURNING id`

	var prefID uuid.UUID
	err := h.queries.DB.QueryRowContext(r.Context(), query,
		req.UserID, req.AgentID, req.AlertTypes, req.Channels,
		req.EmailAddress, req.WebhookURL, req.Enabled,
	).Scan(&prefID)

	if err != nil {
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "alert preferences update failed", err), requestID))
		return
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"id":      prefID.String(),
		"status":  "updated",
		"message": "Alert preferences updated successfully",
	})
}

/* GetAlertPreferences retrieves alert preferences */
func (h *AlertPreferencesHandlers) GetAlertPreferences(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())

	vars := mux.Vars(r)
	agentIDStr := vars["agent_id"]

	var agentID *uuid.UUID
	if agentIDStr != "" {
		if id, err := uuid.Parse(agentIDStr); err == nil {
			agentID = &id
		}
	}

	var userID *uuid.UUID
	if userIDStr := r.URL.Query().Get("user_id"); userIDStr != "" {
		if id, err := uuid.Parse(userIDStr); err == nil {
			userID = &id
		}
	}

	query := `SELECT id, user_id, agent_id, alert_types, channels, email_address, webhook_url, enabled, created_at, updated_at
		FROM neurondb_agent.task_alert_preferences
		WHERE 1=1`
	args := []interface{}{}
	argPos := 1

	if userID != nil {
		query += " AND (user_id = $1 OR user_id IS NULL)"
		args = append(args, *userID)
		argPos++
	}

	if agentID != nil {
		query += fmt.Sprintf(" AND (agent_id = $%d OR agent_id IS NULL)", argPos)
		args = append(args, *agentID)
	}

	query += " ORDER BY created_at DESC"

	rows, err := h.queries.DB.QueryContext(r.Context(), query, args...)
	if err != nil {
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "alert preferences retrieval failed", err), requestID))
		return
	}
	defer rows.Close()

	type AlertPreference struct {
		ID           uuid.UUID  `json:"id"`
		UserID       *uuid.UUID `json:"user_id"`
		AgentID      *uuid.UUID `json:"agent_id"`
		AlertTypes   []string   `json:"alert_types"`
		Channels     []string   `json:"channels"`
		EmailAddress *string    `json:"email_address"`
		WebhookURL   *string    `json:"webhook_url"`
		Enabled      bool       `json:"enabled"`
		CreatedAt    string     `json:"created_at"`
		UpdatedAt    string     `json:"updated_at"`
	}

	var prefs []AlertPreference
	for rows.Next() {
		var pref AlertPreference
		var createdAt, updatedAt string
		err := rows.Scan(
			&pref.ID, &pref.UserID, &pref.AgentID,
			&pref.AlertTypes, &pref.Channels,
			&pref.EmailAddress, &pref.WebhookURL, &pref.Enabled,
			&createdAt, &updatedAt,
		)
		if err != nil {
			continue
		}
		pref.CreatedAt = createdAt
		pref.UpdatedAt = updatedAt
		prefs = append(prefs, pref)
	}

	respondJSON(w, http.StatusOK, prefs)
}
