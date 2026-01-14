/*-------------------------------------------------------------------------
 *
 * plugin.go
 *    Plugin system for NeuronMCP
 *
 * Implements plugin framework as specified in Phase 4.1.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/plugin/plugin.go
 *
 *-------------------------------------------------------------------------
 */

package plugin

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/tools"
)

/* PluginType represents a plugin type */
type PluginType string

const (
	PluginTypeTool         PluginType = "tool"
	PluginTypeResource     PluginType = "resource"
	PluginTypeMiddleware   PluginType = "middleware"
	PluginTypeAuth         PluginType = "auth"
	PluginTypeExporter     PluginType = "exporter"
)

/* Plugin represents a plugin */
type Plugin interface {
	Name() string
	Version() string
	Type() PluginType
	Initialize(ctx context.Context, config map[string]interface{}) error
	Shutdown(ctx context.Context) error
}

/* ToolPlugin extends Plugin for tool plugins */
type ToolPlugin interface {
	Plugin
	GetTools() []tools.Tool
}

/* MiddlewarePlugin extends Plugin for middleware plugins */
type MiddlewarePlugin interface {
	Plugin
	GetMiddleware() interface{} /* Returns middleware.Middleware */
}

/* PluginManager manages plugins */
type PluginManager struct {
	plugins map[string]Plugin
}

/* NewPluginManager creates a new plugin manager */
func NewPluginManager() *PluginManager {
	return &PluginManager{
		plugins: make(map[string]Plugin),
	}
}

/* RegisterPlugin registers a plugin */
func (pm *PluginManager) RegisterPlugin(plugin Plugin) error {
	if _, exists := pm.plugins[plugin.Name()]; exists {
		return fmt.Errorf("plugin %s already registered", plugin.Name())
	}
	pm.plugins[plugin.Name()] = plugin
	return nil
}

/* LoadPlugin loads and initializes a plugin */
func (pm *PluginManager) LoadPlugin(ctx context.Context, plugin Plugin, config map[string]interface{}) error {
	if err := plugin.Initialize(ctx, config); err != nil {
		return fmt.Errorf("failed to initialize plugin %s: %w", plugin.Name(), err)
	}
	return pm.RegisterPlugin(plugin)
}

/* GetPlugin gets a plugin by name */
func (pm *PluginManager) GetPlugin(name string) (Plugin, bool) {
	plugin, exists := pm.plugins[name]
	return plugin, exists
}

/* GetAllPlugins returns all registered plugins */
func (pm *PluginManager) GetAllPlugins() []Plugin {
	plugins := make([]Plugin, 0, len(pm.plugins))
	for _, plugin := range pm.plugins {
		plugins = append(plugins, plugin)
	}
	return plugins
}

/* GetToolPlugins returns all tool plugins */
func (pm *PluginManager) GetToolPlugins() []ToolPlugin {
	toolPlugins := []ToolPlugin{}
	for _, plugin := range pm.plugins {
		if toolPlugin, ok := plugin.(ToolPlugin); ok {
			toolPlugins = append(toolPlugins, toolPlugin)
		}
	}
	return toolPlugins
}

/* ShutdownAll shuts down all plugins */
func (pm *PluginManager) ShutdownAll(ctx context.Context) error {
	var errors []error
	for _, plugin := range pm.plugins {
		if err := plugin.Shutdown(ctx); err != nil {
			errors = append(errors, fmt.Errorf("failed to shutdown plugin %s: %w", plugin.Name(), err))
		}
	}
	if len(errors) > 0 {
		return fmt.Errorf("errors shutting down plugins: %v", errors)
	}
	return nil
}

