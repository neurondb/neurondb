/*-------------------------------------------------------------------------
 *
 * loader.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/config/loader.go
 *
 *-------------------------------------------------------------------------
 */

package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

/* ConfigLoader handles loading configuration from multiple sources */
type ConfigLoader struct{}

/* NewConfigLoader creates a new config loader */
func NewConfigLoader() *ConfigLoader {
	return &ConfigLoader{}
}

/* GetDefaultConfig returns a default configuration */
func GetDefaultConfig() *ServerConfig {
	defaultLimit := 10
	defaultChunkSize := 500
	defaultOverlap := 50
	maxVectorDim := 16384
	maxProjects := 1000
	maxClusters := 1000
	maxIterations := 10000
	maxTrainingTime := 3600000
	timeout := 30000
	maxRequestSize := 10485760
	min := 2
	max := 10
	idleTimeout := 30000
	connTimeout := 5000
	output := "stderr"
	enableReqLog := true
	enableRespLog := false
	enableErrorStack := false
	enableMetrics := true
	enableHealthCheck := true
	gpuEnabled := false

	return &ServerConfig{
		Database: DatabaseConfig{
			Host:     stringPtr("localhost"),
			Port:     intPtr(5432),
			Database: stringPtr("postgres"),
			User:     stringPtr("postgres"),
			Pool: &PoolConfig{
				Min:                   &min,
				Max:                   &max,
				IdleTimeoutMillis:      &idleTimeout,
				ConnectionTimeoutMillis: &connTimeout,
			},
			SSL: false,
		},
		Server: ServerSettings{
			Name:            stringPtr("neurondb-mcp-server"),
			Version:         stringPtr("2.0.0"),
			Timeout:         &timeout,
			MaxRequestSize:  &maxRequestSize,
			EnableMetrics:   &enableMetrics,
			EnableHealthCheck: &enableHealthCheck,
		},
		Logging: LoggingConfig{
			Level:              "info",
			Format:             "text",
			Output:             &output,
			EnableRequestLogging:  &enableReqLog,
			EnableResponseLogging: &enableRespLog,
			EnableErrorStack:      &enableErrorStack,
		},
		Features: FeaturesConfig{
			Vector: &VectorFeatureConfig{
				Enabled:             true,
				DefaultDistanceMetric: stringPtr("l2"),
				MaxVectorDimension:    &maxVectorDim,
				DefaultLimit:          &defaultLimit,
			},
			ML: &MLFeatureConfig{
				Enabled: true,
				Algorithms: []string{
					"linear_regression",
					"ridge",
					"lasso",
					"logistic",
					"random_forest",
					"svm",
					"knn",
					"decision_tree",
					"naive_bayes",
				},
				MaxTrainingTime: &maxTrainingTime,
				GPUEnabled:      &gpuEnabled,
			},
			Analytics: &AnalyticsFeatureConfig{
				Enabled:      true,
				MaxClusters:  &maxClusters,
				MaxIterations: &maxIterations,
			},
			RAG: &RAGFeatureConfig{
				Enabled:        true,
				DefaultChunkSize: &defaultChunkSize,
				DefaultOverlap:   &defaultOverlap,
			},
			Projects: &ProjectsFeatureConfig{
				Enabled:    true,
				MaxProjects: &maxProjects,
			},
		},
	}
}

/* LoadFromFile loads configuration from a JSON file */
func (l *ConfigLoader) LoadFromFile(configPath string) (*ServerConfig, error) {
	possiblePaths := []string{}

	if configPath != "" {
		possiblePaths = append(possiblePaths, configPath)
	}

	if envPath := os.Getenv("NEURONDB_MCP_CONFIG"); envPath != "" {
		possiblePaths = append(possiblePaths, envPath)
	}

	cwd, _ := os.Getwd()
	possiblePaths = append(possiblePaths,
		filepath.Join(cwd, "mcp-config.json"),
		filepath.Join(cwd, "..", "..", "mcp-config.json"),
	)

	if home, err := os.UserHomeDir(); err == nil {
		possiblePaths = append(possiblePaths,
			filepath.Join(home, ".neurondb", "mcp-config.json"),
		)
	}

	for _, path := range possiblePaths {
		if data, err := os.ReadFile(path); err == nil {
			var config ServerConfig
			if err := json.Unmarshal(data, &config); err != nil {
				return nil, fmt.Errorf("failed to parse config from %s: %w", path, err)
			}
			return &config, nil
		}
	}

 	return nil, nil /* No config file found */
}

/* MergeWithEnv merges configuration with environment variables */
func (l *ConfigLoader) MergeWithEnv(config *ServerConfig) *ServerConfig {
	merged := *config

  /* Database config from env */
	if connStr := os.Getenv("NEURONDB_CONNECTION_STRING"); connStr != "" {
		merged.Database.ConnectionString = &connStr
	}
	if host := os.Getenv("NEURONDB_HOST"); host != "" {
		merged.Database.Host = &host
	}
	if portStr := os.Getenv("NEURONDB_PORT"); portStr != "" {
		if port, err := strconv.Atoi(portStr); err == nil {
			merged.Database.Port = &port
		}
	}
	if db := os.Getenv("NEURONDB_DATABASE"); db != "" {
		merged.Database.Database = &db
	}
	if user := os.Getenv("NEURONDB_USER"); user != "" {
		merged.Database.User = &user
	}
	if pass := os.Getenv("NEURONDB_PASSWORD"); pass != "" {
		merged.Database.Password = &pass
	}

  /* Logging config from env */
	if level := os.Getenv("NEURONDB_LOG_LEVEL"); level != "" {
		merged.Logging.Level = level
	}
	if format := os.Getenv("NEURONDB_LOG_FORMAT"); format != "" {
		merged.Logging.Format = format
	}
	if output := os.Getenv("NEURONDB_LOG_OUTPUT"); output != "" {
		merged.Logging.Output = &output
	}

  /* Feature flags from env */
	if gpu := os.Getenv("NEURONDB_ENABLE_GPU"); gpu != "" {
		gpuEnabled := gpu == "true"
		if merged.Features.ML != nil {
			merged.Features.ML.GPUEnabled = &gpuEnabled
		}
	}

	return &merged
}

/* Helper functions */
func stringPtr(s string) *string {
	return &s
}

func intPtr(i int) *int {
	return &i
}

