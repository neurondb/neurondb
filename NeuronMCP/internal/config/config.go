/*-------------------------------------------------------------------------
 *
 * config.go
 *    Configuration management for NeuronMCP
 *
 * Provides configuration loading, validation, and access for server settings,
 * database, logging, and features.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/config/config.go
 *
 *-------------------------------------------------------------------------
 */

package config

import (
	"fmt"
	"os"
)

/* ConfigManager manages configuration loading and access */
type ConfigManager struct {
	config *ServerConfig
}

/* NewConfigManager creates a new config manager */
func NewConfigManager() *ConfigManager {
	return &ConfigManager{}
}

/* Load loads configuration from file and environment */
func (m *ConfigManager) Load(configPath string) (*ServerConfig, error) {
	if m.config != nil {
		return m.config, nil
	}

	loader := NewConfigLoader()

	fileConfig, err := loader.LoadFromFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config file: %w", err)
	}

	var baseConfig *ServerConfig
	if fileConfig != nil {
		baseConfig = fileConfig
	} else {
		baseConfig = GetDefaultConfig()
	}

	m.config = loader.MergeWithEnv(baseConfig)

	validator := NewConfigValidator()
	valid, errors := validator.Validate(m.config)
	if !valid {
		fmt.Fprintf(os.Stderr, "Configuration validation failed with the following errors:\n")
		for _, err := range errors {
			fmt.Fprintf(os.Stderr, "  - %s\n", err)
		}
		return nil, fmt.Errorf("invalid configuration: %d validation error(s)", len(errors))
	}

	return m.config, nil
}

/* GetConfig returns the current configuration */
func (m *ConfigManager) GetConfig() *ServerConfig {
	if m.config == nil {
		if _, err := m.Load(""); err != nil {
			return GetDefaultConfig()
		}
	}
	return m.config
}

/* GetDatabaseConfig returns database configuration */
func (m *ConfigManager) GetDatabaseConfig() *DatabaseConfig {
	return &m.GetConfig().Database
}

/* GetServerSettings returns server settings */
func (m *ConfigManager) GetServerSettings() *ServerSettings {
	return &m.GetConfig().Server
}

/* GetLoggingConfig returns logging configuration */
func (m *ConfigManager) GetLoggingConfig() *LoggingConfig {
	return &m.GetConfig().Logging
}

/* GetFeaturesConfig returns features configuration */
func (m *ConfigManager) GetFeaturesConfig() *FeaturesConfig {
	return &m.GetConfig().Features
}

/* GetPlugins returns plugin configurations */
func (m *ConfigManager) GetPlugins() []PluginConfig {
	return m.GetConfig().Plugins
}

