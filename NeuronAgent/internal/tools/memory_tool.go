/*-------------------------------------------------------------------------
 *
 * memory_tool.go
 *    Memory tool for hierarchical memory access
 *
 * Provides agent access to hierarchical memory system for querying
 * and managing STM, MTM, and LPM tiers.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/memory_tool.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* HierarchicalMemoryInterface defines the interface for hierarchical memory operations */
/* This interface is used to avoid import cycles between tools and agent packages */
type HierarchicalMemoryInterface interface {
	RetrieveHierarchical(ctx context.Context, agentID uuid.UUID, query string, tiers []string, topK int) ([]map[string]interface{}, error)
	StoreSTM(ctx context.Context, agentID, sessionID uuid.UUID, content string, importance float64) (uuid.UUID, error)
}

/* MemoryTool provides hierarchical memory operations for agents */
type MemoryTool struct {
	hierMemory HierarchicalMemoryInterface
}

/* NewMemoryTool creates a new memory tool */
func NewMemoryTool(hierMemory HierarchicalMemoryInterface) *MemoryTool {
	return &MemoryTool{
		hierMemory: hierMemory,
	}
}

/* Execute executes a memory operation */
func (t *MemoryTool) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	action, ok := args["action"].(string)
	if !ok {
		return "", fmt.Errorf("memory tool requires action parameter")
	}

	agentIDStr, ok := args["agent_id"].(string)
	if !ok {
		return "", fmt.Errorf("memory tool requires agent_id parameter")
	}

	agentID, err := uuid.Parse(agentIDStr)
	if err != nil {
		return "", fmt.Errorf("invalid agent_id: %w", err)
	}

	switch action {
	case "query_memory":
		return t.queryMemory(ctx, agentID, args)
	case "store_stm":
		return t.storeSTM(ctx, agentID, args)
	default:
		return "", fmt.Errorf("unknown memory action: %s", action)
	}
}

/* queryMemory queries hierarchical memory */
func (t *MemoryTool) queryMemory(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	query, ok := args["query"].(string)
	if !ok {
		return "", fmt.Errorf("query_memory requires query parameter")
	}

	tiers := []string{"stm", "mtm", "lpm"}
	if tiersArg, ok := args["tiers"].([]interface{}); ok {
		tiers = make([]string, len(tiersArg))
		for i, tier := range tiersArg {
			if tierStr, ok := tier.(string); ok {
				tiers[i] = tierStr
			}
		}
	}

	topK := 5
	if k, ok := args["top_k"].(float64); ok {
		topK = int(k)
	}

	results, err := t.hierMemory.RetrieveHierarchical(ctx, agentID, query, tiers, topK)
	if err != nil {
		return "", fmt.Errorf("memory query failed: %w", err)
	}

	result := map[string]interface{}{
		"action":  "query_memory",
		"query":   query,
		"results": results,
		"status":  "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* storeSTM stores content in short-term memory */
func (t *MemoryTool) storeSTM(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	content, ok := args["content"].(string)
	if !ok {
		return "", fmt.Errorf("store_stm requires content parameter")
	}

	sessionIDStr, ok := args["session_id"].(string)
	if !ok {
		return "", fmt.Errorf("store_stm requires session_id parameter")
	}

	sessionID, err := uuid.Parse(sessionIDStr)
	if err != nil {
		return "", fmt.Errorf("invalid session_id: %w", err)
	}

	importance := 0.5
	if imp, ok := args["importance"].(float64); ok {
		importance = imp
	}

	memoryID, err := t.hierMemory.StoreSTM(ctx, agentID, sessionID, content, importance)
	if err != nil {
		return "", fmt.Errorf("STM storage failed: %w", err)
	}

	result := map[string]interface{}{
		"action":     "store_stm",
		"memory_id":  memoryID.String(),
		"importance": importance,
		"status":     "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* Validate validates tool arguments */
func (t *MemoryTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	action, ok := args["action"].(string)
	if !ok {
		return fmt.Errorf("action parameter required")
	}

	validActions := map[string]bool{
		"query_memory": true,
		"store_stm":    true,
	}

	if !validActions[action] {
		return fmt.Errorf("invalid action: %s", action)
	}

	return nil
}
