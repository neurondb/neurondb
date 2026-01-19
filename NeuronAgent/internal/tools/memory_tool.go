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
	UpdateMemory(ctx context.Context, agentID uuid.UUID, memoryID uuid.UUID, tier string, content *string, importance *float64, topic *string, category *string) error
	DeleteMemory(ctx context.Context, agentID uuid.UUID, memoryID uuid.UUID, tier string) error
	StoreMTM(ctx context.Context, agentID uuid.UUID, topic string, content string, importance float64) (uuid.UUID, error)
	StoreLPM(ctx context.Context, agentID uuid.UUID, category string, content string, importance float64, userID *uuid.UUID) (uuid.UUID, error)
	PromoteMemory(ctx context.Context, agentID uuid.UUID, memoryID uuid.UUID, fromTier string, toTier string, topic *string, category *string, userID *uuid.UUID, reason string) (uuid.UUID, error)
	DemoteMemory(ctx context.Context, agentID uuid.UUID, memoryID uuid.UUID, fromTier string, toTier string, reason string) (uuid.UUID, error)
}

/* MemoryManagementInterface defines the interface for advanced memory management operations */
type MemoryManagementInterface interface {
	CheckCorruption(ctx context.Context, agentID uuid.UUID) ([]map[string]interface{}, error)
	ForgetMemories(ctx context.Context, agentID uuid.UUID, tier string, config map[string]interface{}) (int, error)
	ResolveConflicts(ctx context.Context, agentID uuid.UUID, strategy string) (int, error)
	GetMemoryQuality(ctx context.Context, agentID uuid.UUID, memoryID uuid.UUID, tier string) (map[string]interface{}, error)
}

/* MemoryTool provides hierarchical memory operations for agents */
type MemoryTool struct {
	hierMemory HierarchicalMemoryInterface
	memoryMgmt MemoryManagementInterface
}

/* NewMemoryTool creates a new memory tool */
func NewMemoryTool(hierMemory HierarchicalMemoryInterface) *MemoryTool {
	return &MemoryTool{
		hierMemory: hierMemory,
		memoryMgmt: nil,
	}
}

/* NewMemoryToolWithManagement creates a new memory tool with management capabilities */
func NewMemoryToolWithManagement(hierMemory HierarchicalMemoryInterface, memoryMgmt MemoryManagementInterface) *MemoryTool {
	return &MemoryTool{
		hierMemory: hierMemory,
		memoryMgmt: memoryMgmt,
	}
}

/* SetMemoryManagement sets the memory management interface */
func (t *MemoryTool) SetMemoryManagement(memoryMgmt MemoryManagementInterface) {
	t.memoryMgmt = memoryMgmt
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
	case "update_memory":
		return t.updateMemory(ctx, agentID, args)
	case "delete_memory":
		return t.deleteMemory(ctx, agentID, args)
	case "store_mtm":
		return t.storeMTM(ctx, agentID, args)
	case "store_lpm":
		return t.storeLPM(ctx, agentID, args)
	case "promote_memory":
		return t.promoteMemory(ctx, agentID, args)
	case "demote_memory":
		return t.demoteMemory(ctx, agentID, args)
	case "check_corruption":
		return t.checkCorruption(ctx, agentID, args)
	case "forget_memories":
		return t.forgetMemories(ctx, agentID, args)
	case "resolve_conflicts":
		return t.resolveConflicts(ctx, agentID, args)
	case "get_memory_quality":
		return t.getMemoryQuality(ctx, agentID, args)
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

/* updateMemory updates existing memory */
func (t *MemoryTool) updateMemory(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	memoryIDStr, ok := args["memory_id"].(string)
	if !ok {
		return "", fmt.Errorf("update_memory requires memory_id parameter")
	}

	memoryID, err := uuid.Parse(memoryIDStr)
	if err != nil {
		return "", fmt.Errorf("invalid memory_id: %w", err)
	}

	tier, ok := args["tier"].(string)
	if !ok {
		return "", fmt.Errorf("update_memory requires tier parameter (stm/mtm/lpm)")
	}

	var content *string
	if c, ok := args["content"].(string); ok && c != "" {
		content = &c
	}

	var importance *float64
	if imp, ok := args["importance"].(float64); ok {
		importance = &imp
	}

	var topic *string
	if top, ok := args["topic"].(string); ok && top != "" {
		topic = &top
	}

	var category *string
	if cat, ok := args["category"].(string); ok && cat != "" {
		category = &cat
	}

	err = t.hierMemory.UpdateMemory(ctx, agentID, memoryID, tier, content, importance, topic, category)
	if err != nil {
		return "", fmt.Errorf("memory update failed: %w", err)
	}

	result := map[string]interface{}{
		"action":    "update_memory",
		"memory_id": memoryID.String(),
		"tier":      tier,
		"status":    "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* deleteMemory deletes memory from any tier */
func (t *MemoryTool) deleteMemory(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	memoryIDStr, ok := args["memory_id"].(string)
	if !ok {
		return "", fmt.Errorf("delete_memory requires memory_id parameter")
	}

	memoryID, err := uuid.Parse(memoryIDStr)
	if err != nil {
		return "", fmt.Errorf("invalid memory_id: %w", err)
	}

	tier, ok := args["tier"].(string)
	if !ok {
		return "", fmt.Errorf("delete_memory requires tier parameter (stm/mtm/lpm)")
	}

	err = t.hierMemory.DeleteMemory(ctx, agentID, memoryID, tier)
	if err != nil {
		return "", fmt.Errorf("memory deletion failed: %w", err)
	}

	result := map[string]interface{}{
		"action":    "delete_memory",
		"memory_id": memoryID.String(),
		"tier":      tier,
		"status":    "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* storeMTM stores content directly in mid-term memory */
func (t *MemoryTool) storeMTM(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	topic, ok := args["topic"].(string)
	if !ok {
		return "", fmt.Errorf("store_mtm requires topic parameter")
	}

	content, ok := args["content"].(string)
	if !ok {
		return "", fmt.Errorf("store_mtm requires content parameter")
	}

	importance := 0.6
	if imp, ok := args["importance"].(float64); ok {
		importance = imp
	}

	memoryID, err := t.hierMemory.StoreMTM(ctx, agentID, topic, content, importance)
	if err != nil {
		return "", fmt.Errorf("MTM storage failed: %w", err)
	}

	result := map[string]interface{}{
		"action":     "store_mtm",
		"memory_id":  memoryID.String(),
		"topic":      topic,
		"importance": importance,
		"status":     "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* storeLPM stores content directly in long-term personal memory */
func (t *MemoryTool) storeLPM(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	category, ok := args["category"].(string)
	if !ok {
		return "", fmt.Errorf("store_lpm requires category parameter")
	}

	content, ok := args["content"].(string)
	if !ok {
		return "", fmt.Errorf("store_lpm requires content parameter")
	}

	importance := 0.8
	if imp, ok := args["importance"].(float64); ok {
		importance = imp
	}

	var userID *uuid.UUID
	if userIDStr, ok := args["user_id"].(string); ok && userIDStr != "" {
		parsedUserID, err := uuid.Parse(userIDStr)
		if err != nil {
			return "", fmt.Errorf("invalid user_id: %w", err)
		}
		userID = &parsedUserID
	}

	memoryID, err := t.hierMemory.StoreLPM(ctx, agentID, category, content, importance, userID)
	if err != nil {
		return "", fmt.Errorf("LPM storage failed: %w", err)
	}

	result := map[string]interface{}{
		"action":     "store_lpm",
		"memory_id":  memoryID.String(),
		"category":   category,
		"importance": importance,
		"status":     "success",
	}
	if userID != nil {
		result["user_id"] = userID.String()
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* promoteMemory promotes memory from one tier to another */
func (t *MemoryTool) promoteMemory(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	memoryIDStr, ok := args["memory_id"].(string)
	if !ok {
		return "", fmt.Errorf("promote_memory requires memory_id parameter")
	}

	memoryID, err := uuid.Parse(memoryIDStr)
	if err != nil {
		return "", fmt.Errorf("invalid memory_id: %w", err)
	}

	fromTier, ok := args["from_tier"].(string)
	if !ok {
		return "", fmt.Errorf("promote_memory requires from_tier parameter")
	}

	toTier, ok := args["to_tier"].(string)
	if !ok {
		return "", fmt.Errorf("promote_memory requires to_tier parameter")
	}

	var topic *string
	if top, ok := args["topic"].(string); ok && top != "" {
		topic = &top
	}

	var category *string
	if cat, ok := args["category"].(string); ok && cat != "" {
		category = &cat
	}

	var userID *uuid.UUID
	if userIDStr, ok := args["user_id"].(string); ok && userIDStr != "" {
		parsedUserID, err := uuid.Parse(userIDStr)
		if err != nil {
			return "", fmt.Errorf("invalid user_id: %w", err)
		}
		userID = &parsedUserID
	}

	reason := ""
	if r, ok := args["reason"].(string); ok {
		reason = r
	}

	newMemoryID, err := t.hierMemory.PromoteMemory(ctx, agentID, memoryID, fromTier, toTier, topic, category, userID, reason)
	if err != nil {
		return "", fmt.Errorf("memory promotion failed: %w", err)
	}

	result := map[string]interface{}{
		"action":        "promote_memory",
		"memory_id":     memoryID.String(),
		"new_memory_id": newMemoryID.String(),
		"from_tier":     fromTier,
		"to_tier":       toTier,
		"status":      "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* demoteMemory demotes memory from one tier to another */
func (t *MemoryTool) demoteMemory(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	memoryIDStr, ok := args["memory_id"].(string)
	if !ok {
		return "", fmt.Errorf("demote_memory requires memory_id parameter")
	}

	memoryID, err := uuid.Parse(memoryIDStr)
	if err != nil {
		return "", fmt.Errorf("invalid memory_id: %w", err)
	}

	fromTier, ok := args["from_tier"].(string)
	if !ok {
		return "", fmt.Errorf("demote_memory requires from_tier parameter")
	}

	toTier, ok := args["to_tier"].(string)
	if !ok {
		return "", fmt.Errorf("demote_memory requires to_tier parameter")
	}

	reason := ""
	if r, ok := args["reason"].(string); ok {
		reason = r
	}

	newMemoryID, err := t.hierMemory.DemoteMemory(ctx, agentID, memoryID, fromTier, toTier, reason)
	if err != nil {
		return "", fmt.Errorf("memory demotion failed: %w", err)
	}

	result := map[string]interface{}{
		"action":        "demote_memory",
		"memory_id":     memoryID.String(),
		"new_memory_id": newMemoryID.String(),
		"from_tier":     fromTier,
		"to_tier":       toTier,
		"status":        "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* checkCorruption checks for memory corruption */
func (t *MemoryTool) checkCorruption(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	if t.memoryMgmt == nil {
		return "", fmt.Errorf("memory management interface not available")
	}

	issues, err := t.memoryMgmt.CheckCorruption(ctx, agentID)
	if err != nil {
		return "", fmt.Errorf("corruption check failed: %w", err)
	}

	result := map[string]interface{}{
		"action":      "check_corruption",
		"agent_id":    agentID.String(),
		"issues":      issues,
		"issue_count": len(issues),
		"status":      "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* forgetMemories triggers intelligent forgetting */
func (t *MemoryTool) forgetMemories(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	if t.memoryMgmt == nil {
		return "", fmt.Errorf("memory management interface not available")
	}

	tier := "all"
	if t, ok := args["tier"].(string); ok {
		tier = t
	}

	config := make(map[string]interface{})
	if cfg, ok := args["config"].(map[string]interface{}); ok {
		config = cfg
	}

	forgotten := 0
	var err error

	if tier == "all" {
		for _, tierName := range []string{"stm", "mtm", "lpm"} {
			count, e := t.memoryMgmt.ForgetMemories(ctx, agentID, tierName, config)
			if e != nil {
				err = e
				continue
			}
			forgotten += count
		}
	} else {
		forgotten, err = t.memoryMgmt.ForgetMemories(ctx, agentID, tier, config)
	}

	if err != nil {
		return "", fmt.Errorf("forgetting failed: %w", err)
	}

	result := map[string]interface{}{
		"action":        "forget_memories",
		"agent_id":      agentID.String(),
		"tier":          tier,
		"forgotten":     forgotten,
		"status":        "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* resolveConflicts resolves memory conflicts */
func (t *MemoryTool) resolveConflicts(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	if t.memoryMgmt == nil {
		return "", fmt.Errorf("memory management interface not available")
	}

	strategy := "timestamp"
	if s, ok := args["strategy"].(string); ok {
		strategy = s
	}

	resolved, err := t.memoryMgmt.ResolveConflicts(ctx, agentID, strategy)
	if err != nil {
		return "", fmt.Errorf("conflict resolution failed: %w", err)
	}

	result := map[string]interface{}{
		"action":    "resolve_conflicts",
		"agent_id":  agentID.String(),
		"strategy":  strategy,
		"resolved":  resolved,
		"status":    "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* getMemoryQuality gets quality scores for a memory */
func (t *MemoryTool) getMemoryQuality(ctx context.Context, agentID uuid.UUID, args map[string]interface{}) (string, error) {
	if t.memoryMgmt == nil {
		return "", fmt.Errorf("memory management interface not available")
	}

	memoryIDStr, ok := args["memory_id"].(string)
	if !ok {
		return "", fmt.Errorf("get_memory_quality requires memory_id parameter")
	}

	memoryID, err := uuid.Parse(memoryIDStr)
	if err != nil {
		return "", fmt.Errorf("invalid memory_id: %w", err)
	}

	tier, ok := args["tier"].(string)
	if !ok {
		return "", fmt.Errorf("get_memory_quality requires tier parameter")
	}

	quality, err := t.memoryMgmt.GetMemoryQuality(ctx, agentID, memoryID, tier)
	if err != nil {
		return "", fmt.Errorf("quality retrieval failed: %w", err)
	}

	result := map[string]interface{}{
		"action":     "get_memory_quality",
		"agent_id":   agentID.String(),
		"memory_id":  memoryID.String(),
		"tier":       tier,
		"quality":    quality,
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
		"query_memory":     true,
		"store_stm":        true,
		"update_memory":    true,
		"delete_memory":    true,
		"store_mtm":        true,
		"store_lpm":        true,
		"promote_memory":   true,
		"demote_memory":    true,
		"check_corruption": true,
		"forget_memories":  true,
		"resolve_conflicts": true,
		"get_memory_quality": true,
	}

	if !validActions[action] {
		return fmt.Errorf("invalid action: %s", action)
	}

	return nil
}
