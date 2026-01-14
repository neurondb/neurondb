/*-------------------------------------------------------------------------
 *
 * plugin_system.go
 *    Plugin system for NeuronAgent
 *
 * Provides a plugin architecture for community-contributed tools,
 * memory modules, reasoning patterns, and integrations.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/plugins/plugin_system.go
 *
 *-------------------------------------------------------------------------
 */

package plugins

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* PluginSystem manages plugins for NeuronAgent */
type PluginSystem struct {
	queries    *db.Queries
	plugins    map[string]*Plugin
	mu         sync.RWMutex
	loaders    map[string]PluginLoader
}

/* Plugin represents a loaded plugin */
type Plugin struct {
	ID          uuid.UUID
	Name        string
	Version     string
	Description string
	Author      string
	Type        PluginType
	Status      PluginStatus
	Metadata    map[string]interface{}
	RegisteredAt time.Time
	handler     PluginHandler
}

/* PluginType represents the type of plugin */
type PluginType string

const (
	PluginTypeTool      PluginType = "tool"
	PluginTypeMemory    PluginType = "memory"
	PluginTypeReasoning PluginType = "reasoning"
	PluginTypeIntegration PluginType = "integration"
	PluginTypeCustom    PluginType = "custom"
)

/* PluginStatus represents plugin status */
type PluginStatus string

const (
	PluginStatusLoaded   PluginStatus = "loaded"
	PluginStatusActive   PluginStatus = "active"
	PluginStatusInactive PluginStatus = "inactive"
	PluginStatusError    PluginStatus = "error"
)

/* PluginHandler interface for plugin implementations */
type PluginHandler interface {
	Initialize(ctx context.Context, config map[string]interface{}) error
	Execute(ctx context.Context, params map[string]interface{}) (interface{}, error)
	Cleanup(ctx context.Context) error
}

/* PluginLoader loads plugins */
type PluginLoader func(ctx context.Context, config map[string]interface{}) (PluginHandler, error)

/* PluginManifest describes a plugin */
type PluginManifest struct {
	Name        string                 `json:"name"`
	Version     string                 `json:"version"`
	Description string                 `json:"description"`
	Author      string                 `json:"author"`
	Type        PluginType             `json:"type"`
	EntryPoint  string                 `json:"entry_point"`
	Config      map[string]interface{} `json:"config"`
	Metadata    map[string]interface{} `json:"metadata"`
}

/* NewPluginSystem creates a new plugin system */
func NewPluginSystem(queries *db.Queries) *PluginSystem {
	return &PluginSystem{
		queries: queries,
		plugins: make(map[string]*Plugin),
		loaders: make(map[string]PluginLoader),
	}
}

/* RegisterLoader registers a plugin loader */
func (ps *PluginSystem) RegisterLoader(pluginType PluginType, loader PluginLoader) {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	ps.loaders[string(pluginType)] = loader
}

/* LoadPlugin loads a plugin from manifest */
func (ps *PluginSystem) LoadPlugin(ctx context.Context, manifest *PluginManifest) (*Plugin, error) {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	/* Check if plugin already loaded */
	if existing, exists := ps.plugins[manifest.Name]; exists {
		if existing.Version == manifest.Version {
			return existing, nil
		}
		/* Version mismatch - unload old version */
		ps.unloadPlugin(ctx, existing.ID)
	}

	/* Get loader for plugin type */
	loader, exists := ps.loaders[string(manifest.Type)]
	if !exists {
		return nil, fmt.Errorf("plugin loading failed: no_loader_for_type=true, plugin_name='%s', plugin_type='%s'", manifest.Name, manifest.Type)
	}

	/* Load plugin handler */
	handler, err := loader(ctx, manifest.Config)
	if err != nil {
		return nil, fmt.Errorf("plugin loading failed: handler_creation_error=true, plugin_name='%s', error=%w", manifest.Name, err)
	}

	/* Initialize plugin */
	if err := handler.Initialize(ctx, manifest.Config); err != nil {
		return nil, fmt.Errorf("plugin loading failed: initialization_error=true, plugin_name='%s', error=%w", manifest.Name, err)
	}

	/* Create plugin record */
	plugin := &Plugin{
		ID:          uuid.New(),
		Name:        manifest.Name,
		Version:     manifest.Version,
		Description: manifest.Description,
		Author:      manifest.Author,
		Type:        manifest.Type,
		Status:      PluginStatusLoaded,
		Metadata:    manifest.Metadata,
		RegisteredAt: time.Now(),
		handler:     handler,
	}

	/* Store plugin */
	ps.plugins[manifest.Name] = plugin

	/* Store in database */
	if err := ps.storePlugin(ctx, plugin); err != nil {
		return nil, fmt.Errorf("plugin loading failed: storage_error=true, plugin_name='%s', error=%w", manifest.Name, err)
	}

	return plugin, nil
}

/* UnloadPlugin unloads a plugin */
func (ps *PluginSystem) UnloadPlugin(ctx context.Context, pluginID uuid.UUID) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	return ps.unloadPlugin(ctx, pluginID)
}

/* unloadPlugin unloads a plugin (internal, assumes lock held) */
func (ps *PluginSystem) unloadPlugin(ctx context.Context, pluginID uuid.UUID) error {
	/* Find plugin */
	var plugin *Plugin
	for name, p := range ps.plugins {
		if p.ID == pluginID {
			plugin = p
			delete(ps.plugins, name)
			break
		}
	}

	if plugin == nil {
		return fmt.Errorf("plugin unloading failed: plugin_not_found=true, plugin_id='%s'", pluginID.String())
	}

	/* Cleanup plugin */
	if plugin.handler != nil {
		if err := plugin.handler.Cleanup(ctx); err != nil {
			return fmt.Errorf("plugin unloading failed: cleanup_error=true, plugin_name='%s', error=%w", plugin.Name, err)
		}
	}

	/* Update status in database */
	query := `UPDATE neurondb_agent.plugins SET status = 'inactive' WHERE id = $1`
	_, err := ps.queries.DB.ExecContext(ctx, query, pluginID)
	if err != nil {
		return fmt.Errorf("plugin unloading failed: database_update_error=true, error=%w", err)
	}

	return nil
}

/* ExecutePlugin executes a plugin */
func (ps *PluginSystem) ExecutePlugin(ctx context.Context, pluginID uuid.UUID, params map[string]interface{}) (interface{}, error) {
	ps.mu.RLock()
	defer ps.mu.RUnlock()

	/* Find plugin */
	var plugin *Plugin
	for _, p := range ps.plugins {
		if p.ID == pluginID {
			plugin = p
			break
		}
	}

	if plugin == nil {
		return nil, fmt.Errorf("plugin execution failed: plugin_not_found=true, plugin_id='%s'", pluginID.String())
	}

	if plugin.Status != PluginStatusActive && plugin.Status != PluginStatusLoaded {
		return nil, fmt.Errorf("plugin execution failed: plugin_not_active=true, plugin_name='%s', status='%s'", plugin.Name, plugin.Status)
	}

	/* Execute plugin */
	result, err := plugin.handler.Execute(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("plugin execution failed: execution_error=true, plugin_name='%s', error=%w", plugin.Name, err)
	}

	return result, nil
}

/* ListPlugins lists all loaded plugins */
func (ps *PluginSystem) ListPlugins(ctx context.Context, pluginType *PluginType) ([]*Plugin, error) {
	ps.mu.RLock()
	defer ps.mu.RUnlock()

	plugins := make([]*Plugin, 0)
	for _, plugin := range ps.plugins {
		if pluginType == nil || plugin.Type == *pluginType {
			plugins = append(plugins, plugin)
		}
	}

	return plugins, nil
}

/* GetPlugin gets a plugin by ID */
func (ps *PluginSystem) GetPlugin(ctx context.Context, pluginID uuid.UUID) (*Plugin, error) {
	ps.mu.RLock()
	defer ps.mu.RUnlock()

	for _, plugin := range ps.plugins {
		if plugin.ID == pluginID {
			return plugin, nil
		}
	}

	return nil, fmt.Errorf("plugin not found: plugin_id='%s'", pluginID.String())
}

/* storePlugin stores plugin in database */
func (ps *PluginSystem) storePlugin(ctx context.Context, plugin *Plugin) error {
	query := `INSERT INTO neurondb_agent.plugins
		(id, name, version, description, author, type, status, metadata, registered_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
		ON CONFLICT (name, version) DO UPDATE
		SET status = $7, metadata = $8::jsonb, registered_at = $9`

	_, err := ps.queries.DB.ExecContext(ctx, query,
		plugin.ID,
		plugin.Name,
		plugin.Version,
		plugin.Description,
		plugin.Author,
		string(plugin.Type),
		string(plugin.Status),
		plugin.Metadata,
		plugin.RegisteredAt,
	)

	return err
}

