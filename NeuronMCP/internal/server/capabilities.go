/*-------------------------------------------------------------------------
 *
 * capabilities.go
 *    Server capabilities and version negotiation
 *
 * Exposes server version, tool versions, model versions, and feature flags.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/capabilities.go
 *
 *-------------------------------------------------------------------------
 */

package server

import (
	"fmt"
	"sync"

	"github.com/neurondb/NeuronMCP/internal/tools"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* CapabilitiesManager manages server capabilities and version information */
type CapabilitiesManager struct {
	mu             sync.RWMutex
	serverVersion  string
	serverName     string
	toolRegistry   *tools.ToolRegistry
	featureFlags   map[string]bool
	modelVersions  map[string]string
}

/* NewCapabilitiesManager creates a new capabilities manager */
func NewCapabilitiesManager(serverName, serverVersion string, toolRegistry *tools.ToolRegistry) *CapabilitiesManager {
	return &CapabilitiesManager{
		serverVersion: serverVersion,
		serverName:    serverName,
		toolRegistry:  toolRegistry,
		featureFlags: map[string]bool{
			"pagination":        true,
			"streaming":         true,
			"dry_run":           true,
			"idempotency":       true,
			"audit_logging":     true,
			"scoped_auth":       true,
			"rate_limiting":     true,
			"output_validation": true,
			"tool_versioning":   true,
			"deprecation":       true,
			"composite_tools":   true,
			"resource_catalog":  true,
		},
		modelVersions: map[string]string{
			"default_embedding": "2.0.0",
			"default_llm":       "2.0.0",
		},
	}
}

/* GetServerInfo returns server information (thread-safe) */
func (cm *CapabilitiesManager) GetServerInfo() mcp.ServerInfo {
	if cm == nil {
		return mcp.ServerInfo{}
	}
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	return mcp.ServerInfo{
		Name:    cm.serverName,
		Version: cm.serverVersion,
	}
}

/* GetServerCapabilities returns server capabilities (thread-safe) */
func (cm *CapabilitiesManager) GetServerCapabilities() mcp.ServerCapabilities {
	if cm == nil {
		return mcp.ServerCapabilities{}
	}
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	/* Get all tool definitions to extract versions */
	toolVersions := make(map[string]string)
	if cm.toolRegistry != nil {
		toolDefs := cm.toolRegistry.GetAllDefinitions()
		for _, def := range toolDefs {
			if def.Version != "" {
				toolVersions[def.Name] = def.Version
			}
		}
	}

	/* Create a copy of feature flags for thread safety */
	featureFlags := make(map[string]bool)
	if cm.featureFlags != nil {
		for k, v := range cm.featureFlags {
			featureFlags[k] = v
		}
	}

	/* Create a copy of model versions for thread safety */
	modelVersions := make(map[string]string)
	if cm.modelVersions != nil {
		for k, v := range cm.modelVersions {
			modelVersions[k] = v
		}
	}

	return mcp.ServerCapabilities{
		Tools: mcp.ToolsCapability{
			ListChanged: false, /* Set to false for Claude Desktop compatibility - tools list is static */
		},
		Resources: mcp.ResourcesCapability{
			Subscribe:   false,
			ListChanged: false, /* Set to false for Claude Desktop compatibility */
		},
		Experimental: map[string]interface{}{
			"feature_flags":  featureFlags,
			"tool_versions":  toolVersions,
			"model_versions": modelVersions,
			"server_version": cm.serverVersion,
		},
	}
}


/* GetModelVersion gets a model version */
func (cm *CapabilitiesManager) GetModelVersion(modelName string) string {
	if cm == nil {
		return ""
	}
	if modelName == "" {
		return ""
	}
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.modelVersions[modelName]
}

/* ValidateFeatureFlag validates a feature flag name */
func (cm *CapabilitiesManager) ValidateFeatureFlag(name string) error {
	if name == "" {
		return fmt.Errorf("feature flag name cannot be empty")
	}
	if len(name) > 100 {
		return fmt.Errorf("feature flag name too long: %d characters (max: 100)", len(name))
	}
	/* Feature flag names should be alphanumeric with underscores and hyphens */
	for _, char := range name {
		if !((char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') || (char >= '0' && char <= '9') || char == '_' || char == '-') {
			return fmt.Errorf("feature flag name contains invalid character: '%c' (must be alphanumeric, underscore, or hyphen)", char)
		}
	}
	return nil
}

/* SetFeatureFlag sets a feature flag with validation */
func (cm *CapabilitiesManager) SetFeatureFlag(name string, enabled bool) error {
	if cm == nil {
		return fmt.Errorf("capabilities manager is nil")
	}
	if err := cm.ValidateFeatureFlag(name); err != nil {
		return fmt.Errorf("invalid feature flag name: %w", err)
	}
	cm.mu.Lock()
	defer cm.mu.Unlock()
	if cm.featureFlags == nil {
		cm.featureFlags = make(map[string]bool)
	}
	cm.featureFlags[name] = enabled
	return nil
}

/* GetFeatureFlag gets a feature flag (thread-safe) */
func (cm *CapabilitiesManager) GetFeatureFlag(name string) bool {
	if cm == nil {
		return false
	}
	if name == "" {
		return false
	}
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	if cm.featureFlags == nil {
		return false
	}
	return cm.featureFlags[name]
}

/* SetModelVersion sets a model version with validation */
func (cm *CapabilitiesManager) SetModelVersion(modelName, version string) error {
	if cm == nil {
		return fmt.Errorf("capabilities manager is nil")
	}
	if modelName == "" {
		return fmt.Errorf("model name cannot be empty")
	}
	if version == "" {
		return fmt.Errorf("model version cannot be empty")
	}
	cm.mu.Lock()
	defer cm.mu.Unlock()
	if cm.modelVersions == nil {
		cm.modelVersions = make(map[string]string)
	}
	cm.modelVersions[modelName] = version
	return nil
}
