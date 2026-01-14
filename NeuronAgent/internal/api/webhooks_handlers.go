/*-------------------------------------------------------------------------
 *
 * webhooks_handlers.go
 *    API handlers for webhooks
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/webhooks_handlers.go
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

/* ListWebhooks lists all webhooks */
func (h *Handlers) ListWebhooks(w http.ResponseWriter, r *http.Request) {
	webhooks, err := h.queries.ListWebhooks(r.Context())
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list webhooks", err, requestID, r.URL.Path, r.Method, "webhook", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, webhooks)
}

/* CreateWebhook creates a webhook */
func (h *Handlers) CreateWebhook(w http.ResponseWriter, r *http.Request) {
	var req struct {
		URL            string                 `json:"url"`
		Events         []string               `json:"events"`
		Secret         *string                `json:"secret"`
		Enabled        bool                   `json:"enabled"`
		TimeoutSeconds int                    `json:"timeout_seconds"`
		RetryCount     int                    `json:"retry_count"`
		Metadata       map[string]interface{} `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "webhook", "", nil))
		return
	}

	if req.URL == "" {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "url is required", nil, requestID, r.URL.Path, r.Method, "webhook", "", nil))
		return
	}

	if len(req.Events) == 0 {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "events are required", nil, requestID, r.URL.Path, r.Method, "webhook", "", nil))
		return
	}

	if req.TimeoutSeconds == 0 {
		req.TimeoutSeconds = 30
	}

	if req.RetryCount == 0 {
		req.RetryCount = 3
	}

	var metadata db.JSONBMap
	if req.Metadata != nil {
		metadata = db.FromMap(req.Metadata)
	} else {
		metadata = make(db.JSONBMap)
	}

	webhook := &db.Webhook{
		URL:            req.URL,
		Events:         req.Events,
		Secret:         req.Secret,
		Enabled:        req.Enabled,
		TimeoutSeconds: req.TimeoutSeconds,
		RetryCount:     req.RetryCount,
		Metadata:       metadata,
	}

	if err := h.queries.CreateWebhook(r.Context(), webhook); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to create webhook", err, requestID, r.URL.Path, r.Method, "webhook", "", nil))
		return
	}

	respondJSON(w, http.StatusCreated, webhook)
}

/* GetWebhook gets a webhook by ID */
func (h *Handlers) GetWebhook(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid webhook id", err, requestID, r.URL.Path, r.Method, "webhook", "", nil))
		return
	}

	webhook, err := h.queries.GetWebhook(r.Context(), id)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "webhook not found", err, requestID, r.URL.Path, r.Method, "webhook", id.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, webhook)
}

/* UpdateWebhook updates a webhook */
func (h *Handlers) UpdateWebhook(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid webhook id", err, requestID, r.URL.Path, r.Method, "webhook", "", nil))
		return
	}

	webhook, err := h.queries.GetWebhook(r.Context(), id)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "webhook not found", err, requestID, r.URL.Path, r.Method, "webhook", id.String(), nil))
		return
	}

	var req struct {
		URL            *string                `json:"url"`
		Events         []string               `json:"events"`
		Secret         *string                `json:"secret"`
		Enabled        *bool                  `json:"enabled"`
		TimeoutSeconds *int                   `json:"timeout_seconds"`
		RetryCount     *int                   `json:"retry_count"`
		Metadata       map[string]interface{} `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "webhook", id.String(), nil))
		return
	}

	if req.URL != nil {
		webhook.URL = *req.URL
	}
	if req.Events != nil {
		webhook.Events = req.Events
	}
	if req.Secret != nil {
		webhook.Secret = req.Secret
	}
	if req.Enabled != nil {
		webhook.Enabled = *req.Enabled
	}
	if req.TimeoutSeconds != nil {
		webhook.TimeoutSeconds = *req.TimeoutSeconds
	}
	if req.RetryCount != nil {
		webhook.RetryCount = *req.RetryCount
	}
	if req.Metadata != nil {
		webhook.Metadata = db.FromMap(req.Metadata)
	}

	if err := h.queries.UpdateWebhook(r.Context(), webhook); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to update webhook", err, requestID, r.URL.Path, r.Method, "webhook", id.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, webhook)
}

/* DeleteWebhook deletes a webhook */
func (h *Handlers) DeleteWebhook(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid webhook id", err, requestID, r.URL.Path, r.Method, "webhook", "", nil))
		return
	}

	if err := h.queries.DeleteWebhook(r.Context(), id); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "webhook not found", err, requestID, r.URL.Path, r.Method, "webhook", id.String(), nil))
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

/* ListWebhookDeliveries lists deliveries for a webhook */
func (h *Handlers) ListWebhookDeliveries(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid webhook id", err, requestID, r.URL.Path, r.Method, "webhook", "", nil))
		return
	}

	limit := 50
	offset := 0
	if l := r.URL.Query().Get("limit"); l != "" {
		fmt.Sscanf(l, "%d", &limit)
	}
	if o := r.URL.Query().Get("offset"); o != "" {
		fmt.Sscanf(o, "%d", &offset)
	}

	deliveries, err := h.queries.ListWebhookDeliveries(r.Context(), id, limit, offset)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list webhook deliveries", err, requestID, r.URL.Path, r.Method, "webhook", id.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, deliveries)
}


