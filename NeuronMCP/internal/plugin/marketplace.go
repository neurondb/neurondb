/*-------------------------------------------------------------------------
 *
 * marketplace.go
 *    Plugin Marketplace for NeuronMCP
 *
 * Provides plugin marketplace, discovery, and management.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/plugin/marketplace.go
 *
 *-------------------------------------------------------------------------
 */

package plugin

import (
	"context"
	"fmt"
	"strings"
	"time"
)

/* Marketplace provides plugin marketplace functionality */
type Marketplace struct {
	plugins map[string]*PluginMetadata
}

/* PluginMetadata represents plugin metadata */
type PluginMetadata struct {
	Name         string                 `json:"name"`
	Version      string                 `json:"version"`
	Description  string                 `json:"description"`
	Author       string                 `json:"author"`
	Category     string                 `json:"category"`
	Tags         []string               `json:"tags"`
	Downloads    int64                  `json:"downloads"`
	Rating       float64                `json:"rating"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
	Dependencies map[string]string      `json:"dependencies"`
	Config       map[string]interface{} `json:"config"`
}

/* NewMarketplace creates a new marketplace */
func NewMarketplace() *Marketplace {
	return &Marketplace{
		plugins: make(map[string]*PluginMetadata),
	}
}

/* RegisterPlugin registers a plugin in the marketplace */
func (m *Marketplace) RegisterPlugin(metadata *PluginMetadata) error {
	if metadata == nil {
		return fmt.Errorf("plugin metadata cannot be nil")
	}
	if metadata.Name == "" {
		return fmt.Errorf("plugin name is required")
	}

	m.plugins[metadata.Name] = metadata
	return nil
}

/* ListPlugins lists all plugins */
func (m *Marketplace) ListPlugins(ctx context.Context, category, tag string) []*PluginMetadata {
	plugins := []*PluginMetadata{}

	for _, plugin := range m.plugins {
		if category != "" && plugin.Category != category {
			continue
		}
		if tag != "" {
			found := false
			for _, t := range plugin.Tags {
				if t == tag {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}
		plugins = append(plugins, plugin)
	}

	return plugins
}

/* GetPlugin gets plugin metadata */
func (m *Marketplace) GetPlugin(name string) (*PluginMetadata, bool) {
	plugin, exists := m.plugins[name]
	return plugin, exists
}

/* SearchPlugins searches plugins */
func (m *Marketplace) SearchPlugins(ctx context.Context, query string) []*PluginMetadata {
	results := []*PluginMetadata{}

	queryLower := strings.ToLower(query)
	for _, plugin := range m.plugins {
		if strings.Contains(strings.ToLower(plugin.Name), queryLower) ||
			strings.Contains(strings.ToLower(plugin.Description), queryLower) {
			results = append(results, plugin)
		}
	}

	return results
}

