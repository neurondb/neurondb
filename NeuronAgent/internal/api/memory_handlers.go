/*-------------------------------------------------------------------------
 *
 * memory_handlers.go
 *    API handlers for memory management
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/memory_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/internal/validation"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* ListMemoryChunks lists memory chunks for an agent */
func (h *Handlers) ListMemoryChunks(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id format", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	limit := 50
	offset := 0
	if l := r.URL.Query().Get("limit"); l != "" {
		fmt.Sscanf(l, "%d", &limit)
		if err := validation.ValidateLimit(limit); err != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid limit parameter", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
			return
		}
	}
	if o := r.URL.Query().Get("offset"); o != "" {
		fmt.Sscanf(o, "%d", &offset)
		if err := validation.ValidateOffset(offset); err != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid offset parameter", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
			return
		}
	}

	chunks, err := h.queries.ListMemoryChunks(r.Context(), agentID, limit, offset)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list memory chunks", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, chunks)
}

/* GetMemoryChunk gets a memory chunk by ID */
func (h *Handlers) GetMemoryChunk(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	var id int64
	if _, err := fmt.Sscanf(vars["chunk_id"], "%d", &id); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid chunk id", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	chunk, err := h.queries.GetMemoryChunk(r.Context(), id)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "memory chunk not found", err, requestID, r.URL.Path, r.Method, "memory", fmt.Sprintf("%d", id), nil))
		return
	}

	respondJSON(w, http.StatusOK, chunk)
}

/* DeleteMemoryChunk deletes a memory chunk */
func (h *Handlers) DeleteMemoryChunk(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	var id int64
	if _, err := fmt.Sscanf(vars["chunk_id"], "%d", &id); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid chunk id", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	if err := h.queries.DeleteMemoryChunk(r.Context(), id); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "memory chunk not found", err, requestID, r.URL.Path, r.Method, "memory", fmt.Sprintf("%d", id), nil))
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

/* SearchMemory searches memory chunks by query text */
func (h *Handlers) SearchMemory(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id format", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	/* Validate request body size (max 1MB) */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req struct {
		Query string `json:"query"`
		TopK  int    `json:"top_k"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	if err := validation.ValidateRequired(req.Query, "query"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "query validation failed", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}
	if err := validation.ValidateMaxLength(req.Query, "query", 10000); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "query too long", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	if req.TopK <= 0 {
		req.TopK = 5
	}
	if req.TopK > 100 {
		req.TopK = 100
	}

	/* Generate embedding for query */
	embedClient := neurondb.NewClient(h.queries.GetDB()).Embedding
	embeddingModel := "all-MiniLM-L6-v2"
	embedding, err := embedClient.Embed(r.Context(), req.Query, embeddingModel)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to generate embedding", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Search memory */
	chunks, err := h.queries.SearchMemory(r.Context(), agentID, embedding, req.TopK)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to search memory", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, chunks)
}

/* CheckMemoryCorruption checks for memory corruption */
func (h *Handlers) CheckMemoryCorruption(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id format", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	/* Verify agent exists */
	_, err = h.queries.GetAgentByID(r.Context(), agentID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusNotFound, "agent not found", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Get corruption detector from runtime */
	detector := h.runtime.GetCorruptionDetector()
	if detector == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "corruption detector not available", nil, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Execute corruption detection */
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()

	issues, err := detector.ValidateMemoryIntegrity(ctx, agentID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "corruption check failed", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Also check for conflicts */
	conflicts, err := detector.DetectConflicts(ctx, agentID)
	if err != nil {
		/* Log but don't fail */
		metrics.WarnWithContext(ctx, "Conflict detection failed during corruption check", map[string]interface{}{
			"agent_id": agentID.String(),
			"error":    err.Error(),
		})
	}

	/* Count issues by severity */
	severityCounts := make(map[string]int)
	repairableCount := 0
	for _, issue := range issues {
		severityCounts[issue.Severity]++
		if issue.Repairable {
			repairableCount++
		}
	}

	/* Attempt to repair repairable issues */
	repaired := 0
	for _, issue := range issues {
		if issue.Repairable {
			if err := detector.RepairCorruptedMemory(ctx, issue); err == nil {
				repaired++
			}
		}
	}

	duration := time.Since(start)
	respondJSON(w, http.StatusOK, map[string]interface{}{
		"agent_id":        agentID.String(),
		"status":          "completed",
		"total_issues":    len(issues),
		"conflicts":       len(conflicts),
		"repairable":      repairableCount,
		"repaired":        repaired,
		"severity_counts": severityCounts,
		"issues":          issues,
		"duration_ms":     duration.Milliseconds(),
		"checked_at":      time.Now().UTC(),
	})
}

/* ForgetMemories triggers intelligent forgetting */
func (h *Handlers) ForgetMemories(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id format", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	/* Verify agent exists */
	_, err = h.queries.GetAgentByID(r.Context(), agentID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusNotFound, "agent not found", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Validate request body size */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req struct {
		Tier   string                 `json:"tier"`   /* "stm", "mtm", "lpm", or "all" */
		Config map[string]interface{} `json:"config"` /* Optional forgetting configuration */
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Validate tier */
	if req.Tier != "" && req.Tier != "stm" && req.Tier != "mtm" && req.Tier != "lpm" && req.Tier != "all" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid tier: must be 'stm', 'mtm', 'lpm', or 'all'", nil, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Get forgetting manager from runtime */
	forgettingManager := h.runtime.GetForgettingManager()
	if forgettingManager == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "forgetting manager not available", nil, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Build forgetting config from request */
	config := agent.DefaultForgettingConfig()
	if req.Config != nil {
		if strategy, ok := req.Config["strategy"].(string); ok {
			config.Strategy = agent.ForgettingStrategy(strategy)
		}
		if timeThresholdHours, ok := req.Config["time_threshold_hours"].(float64); ok {
			config.TimeThreshold = time.Duration(timeThresholdHours) * time.Hour
		}
		if importanceThreshold, ok := req.Config["importance_threshold"].(float64); ok {
			config.ImportanceThreshold = importanceThreshold
		}
		if relevanceThreshold, ok := req.Config["relevance_threshold"].(float64); ok {
			config.RelevanceThreshold = relevanceThreshold
		}
		if maxMemorySize, ok := req.Config["max_memory_size"].(float64); ok {
			config.MaxMemorySize = int(maxMemorySize)
		}
		if archiveBeforeDelete, ok := req.Config["archive_before_delete"].(bool); ok {
			config.ArchiveBeforeDelete = archiveBeforeDelete
		}
	}

	/* Execute forgetting */
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Minute)
	defer cancel()

	tiers := []string{"stm", "mtm", "lpm"}
	if req.Tier != "" && req.Tier != "all" {
		tiers = []string{req.Tier}
	}

	totalForgotten := 0
	tierResults := make(map[string]int)
	for _, tier := range tiers {
		forgotten, err := forgettingManager.ForgetMemories(ctx, agentID, tier, config)
		if err != nil {
			metrics.WarnWithContext(ctx, "Forgetting failed for tier", map[string]interface{}{
				"agent_id": agentID.String(),
				"tier":     tier,
				"error":    err.Error(),
			})
			continue
		}
		tierResults[tier] = forgotten
		totalForgotten += forgotten
	}

	duration := time.Since(start)
	respondJSON(w, http.StatusOK, map[string]interface{}{
		"agent_id":      agentID.String(),
		"tier":          req.Tier,
		"status":        "completed",
		"total_forgotten": totalForgotten,
		"tier_results":  tierResults,
		"config_used":   config,
		"duration_ms":   duration.Milliseconds(),
		"completed_at":  time.Now().UTC(),
	})
}

/* ResolveMemoryConflicts resolves memory conflicts */
func (h *Handlers) ResolveMemoryConflicts(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id format", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	/* Verify agent exists */
	_, err = h.queries.GetAgentByID(r.Context(), agentID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusNotFound, "agent not found", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Validate request body size */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req struct {
		Strategy string `json:"strategy"` /* "timestamp", "confidence", "source", "llm", "merge" */
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Validate and set default strategy */
	validStrategies := map[string]bool{
		"timestamp": true,
		"confidence": true,
		"source":    true,
		"llm":       true,
		"merge":     true,
	}
	if req.Strategy == "" {
		req.Strategy = "timestamp"
	}
	if !validStrategies[req.Strategy] {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, fmt.Sprintf("invalid strategy: must be one of %v", validStrategies), nil, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Get conflict resolver from runtime */
	conflictResolver := h.runtime.GetConflictResolver()
	if conflictResolver == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "conflict resolver not available", nil, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Execute conflict detection and resolution */
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Minute)
	defer cancel()

	/* Detect conflicts */
	conflicts, err := conflictResolver.DetectConflicts(ctx, agentID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "conflict detection failed", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Resolve conflicts */
	resolved := 0
	resolutionStrategy := agent.ResolutionStrategyTimestamp
	switch req.Strategy {
	case "confidence":
		resolutionStrategy = agent.ResolutionStrategyConfidence
	case "source":
		resolutionStrategy = agent.ResolutionStrategySource
	case "llm":
		resolutionStrategy = agent.ResolutionStrategyLLM
	case "merge":
		resolutionStrategy = agent.ResolutionStrategyMerge
	}

	resolutionResults := make([]map[string]interface{}, 0)
	for _, conflict := range conflicts {
		if !conflict.Resolved {
			if err := conflictResolver.ResolveConflict(ctx, conflict, resolutionStrategy); err == nil {
				resolved++
				resolutionResults = append(resolutionResults, map[string]interface{}{
					"conflict_id": conflict.ConflictID.String(),
					"tier":        conflict.Tier,
					"type":        conflict.ConflictType,
					"resolved":    true,
				})
			} else {
				resolutionResults = append(resolutionResults, map[string]interface{}{
					"conflict_id": conflict.ConflictID.String(),
					"tier":        conflict.Tier,
					"type":        conflict.ConflictType,
					"resolved":    false,
					"error":       err.Error(),
				})
			}
		}
	}

	duration := time.Since(start)
	respondJSON(w, http.StatusOK, map[string]interface{}{
		"agent_id":          agentID.String(),
		"strategy":          req.Strategy,
		"status":            "completed",
		"conflicts_found":   len(conflicts),
		"conflicts_resolved": resolved,
		"resolution_results": resolutionResults,
		"duration_ms":       duration.Milliseconds(),
		"completed_at":      time.Now().UTC(),
	})
}

/* GetMemoryQuality gets quality scores for memories */
func (h *Handlers) GetMemoryQuality(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id format", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	/* Verify agent exists */
	_, err = h.queries.GetAgentByID(r.Context(), agentID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusNotFound, "agent not found", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Get quality scorer from runtime */
	qualityScorer := h.runtime.GetQualityScorer()
	if qualityScorer == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "quality scorer not available", nil, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Get memory_id and tier from query params */
	memoryIDStr := r.URL.Query().Get("memory_id")
	tier := r.URL.Query().Get("tier")

	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()

	if memoryIDStr != "" && tier != "" {
		/* Get quality for specific memory */
		memoryID, err := uuid.Parse(memoryIDStr)
		if err != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid memory_id format", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
			return
		}

		/* Validate tier */
		if tier != "stm" && tier != "mtm" && tier != "lpm" {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid tier: must be 'stm', 'mtm', or 'lpm'", nil, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
			return
		}

		score, err := qualityScorer.ScoreMemory(ctx, agentID, memoryID, tier)
		if err != nil {
			respondError(w, NewErrorWithContext(http.StatusInternalServerError, "quality scoring failed", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
			return
		}

		duration := time.Since(start)
		respondJSON(w, http.StatusOK, map[string]interface{}{
			"agent_id":  agentID.String(),
			"memory_id": memoryID.String(),
			"tier":      tier,
			"quality": map[string]interface{}{
				"completeness":  score.Completeness,
				"accuracy":      score.Accuracy,
				"relevance":     score.Relevance,
				"freshness":     score.Freshness,
				"consistency":   score.Consistency,
				"overall_score": score.OverallScore,
			},
			"duration_ms": duration.Milliseconds(),
			"scored_at":   time.Now().UTC(),
		})
		return
	}

	/* Update all memory quality scores and return summary */
	updated, err := qualityScorer.UpdateAllMemoryQuality(ctx, agentID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "bulk quality update failed", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	duration := time.Since(start)
	respondJSON(w, http.StatusOK, map[string]interface{}{
		"agent_id":    agentID.String(),
		"status":      "completed",
		"updated":     updated,
		"message":     "Quality scores updated for all memories. Use memory_id and tier query params for specific memory quality.",
		"duration_ms": duration.Milliseconds(),
		"updated_at":  time.Now().UTC(),
	})
}

/* SubmitMemoryFeedback submits user feedback on a memory */
func (h *Handlers) SubmitMemoryFeedback(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Parse memory_id from URL path */
	memoryIDStr := vars["memory_id"]
	if memoryIDStr == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "memory_id required in path", nil, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	memoryID, err := uuid.Parse(memoryIDStr)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid memory_id format", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	/* Parse request body */
	var req struct {
		AgentID       string   `json:"agent_id"`
		SessionID     *string  `json:"session_id,omitempty"`
		MemoryTier    string   `json:"memory_tier"`    /* chunk, stm, mtm, lpm */
		FeedbackType  string   `json:"feedback_type"`  /* positive, negative, neutral, correction */
		FeedbackText  string   `json:"feedback_text,omitempty"`
		Query         string   `json:"query,omitempty"`
		RelevanceScore *float64 `json:"relevance_score,omitempty"`
		Metadata      map[string]interface{} `json:"metadata,omitempty"`
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "failed to read request body", err, requestID, r.URL.Path, r.Method, "memory", memoryIDStr, nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewBuffer(body))

	if err := json.Unmarshal(body, &req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "memory", memoryIDStr, nil))
		return
	}

	/* Validate required fields */
	if req.AgentID == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "agent_id required", nil, requestID, r.URL.Path, r.Method, "memory", memoryIDStr, nil))
		return
	}

	agentID, err := uuid.Parse(req.AgentID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent_id format", err, requestID, r.URL.Path, r.Method, "memory", memoryIDStr, nil))
		return
	}

	if req.MemoryTier == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "memory_tier required", nil, requestID, r.URL.Path, r.Method, "memory", memoryIDStr, nil))
		return
	}

	if req.FeedbackType == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "feedback_type required", nil, requestID, r.URL.Path, r.Method, "memory", memoryIDStr, nil))
		return
	}

	/* Validate memory_tier */
	validTiers := map[string]bool{"chunk": true, "stm": true, "mtm": true, "lpm": true}
	if !validTiers[req.MemoryTier] {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid memory_tier: must be 'chunk', 'stm', 'mtm', or 'lpm'", nil, requestID, r.URL.Path, r.Method, "memory", memoryIDStr, nil))
		return
	}

	/* Validate feedback_type */
	validTypes := map[string]bool{"positive": true, "negative": true, "neutral": true, "correction": true}
	if !validTypes[req.FeedbackType] {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid feedback_type: must be 'positive', 'negative', 'neutral', or 'correction'", nil, requestID, r.URL.Path, r.Method, "memory", memoryIDStr, nil))
		return
	}

	/* Validate relevance_score if provided */
	if req.RelevanceScore != nil {
		if *req.RelevanceScore < 0 || *req.RelevanceScore > 1 {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "relevance_score must be between 0 and 1", nil, requestID, r.URL.Path, r.Method, "memory", memoryIDStr, nil))
			return
		}
	}

	/* Verify agent exists */
	_, err = h.queries.GetAgentByID(r.Context(), agentID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusNotFound, "agent not found", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Parse session_id if provided */
	var sessionID *uuid.UUID
	if req.SessionID != nil && *req.SessionID != "" {
		parsedSessionID, err := uuid.Parse(*req.SessionID)
		if err != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session_id format", err, requestID, r.URL.Path, r.Method, "memory", memoryIDStr, nil))
			return
		}
		sessionID = &parsedSessionID
	}

	/* Get memory learning manager from runtime */
	memoryLearning := h.runtime.GetMemoryLearning()
	if memoryLearning == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "memory learning manager not available", nil, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Create feedback object */
	feedback := &agent.MemoryFeedback{
		AgentID:       agentID,
		SessionID:     sessionID,
		MemoryID:      memoryID,
		MemoryTier:    req.MemoryTier,
		FeedbackType:  req.FeedbackType,
		FeedbackText:  req.FeedbackText,
		Query:         req.Query,
		RelevanceScore: req.RelevanceScore,
		Metadata:      req.Metadata,
	}

	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	/* Record feedback */
	feedbackID, err := memoryLearning.RecordFeedback(ctx, feedback)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to record feedback", err, requestID, r.URL.Path, r.Method, "memory", memoryIDStr, nil))
		return
	}

	duration := time.Since(start)
	respondJSON(w, http.StatusCreated, map[string]interface{}{
		"feedback_id":   feedbackID.String(),
		"memory_id":     memoryID.String(),
		"agent_id":      agentID.String(),
		"feedback_type": req.FeedbackType,
		"status":        "recorded",
		"message":       "Feedback recorded and memory quality updated",
		"duration_ms":   duration.Milliseconds(),
		"created_at":    time.Now().UTC(),
	})
}
