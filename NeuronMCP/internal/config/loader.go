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
	"strings"
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
		/* Validate user-provided config path to prevent path traversal */
		if err := l.validateConfigPath(configPath); err != nil {
			return nil, fmt.Errorf("invalid config path: %w", err)
		}
		possiblePaths = append(possiblePaths, configPath)
	}

	if envPath := os.Getenv("NEURONDB_MCP_CONFIG"); envPath != "" {
		/* Validate environment-provided config path */
		if err := l.validateConfigPath(envPath); err != nil {
			return nil, fmt.Errorf("invalid config path from environment: %w", err)
		}
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
		/* Additional validation: ensure resolved path is within allowed directories */
		if err := l.validateResolvedPath(path); err != nil {
			/* Skip invalid paths, continue to next */
			continue
		}
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

/* validateConfigPath validates a config path to prevent path traversal attacks */
func (l *ConfigLoader) validateConfigPath(path string) error {
	if path == "" {
		return fmt.Errorf("path cannot be empty")
	}
	
	/* Check for path traversal patterns */
	if strings.Contains(path, "..") {
		return fmt.Errorf("path contains traversal pattern (..)")
	}
	
	/* Get absolute path to check against whitelist */
	absPath, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("failed to resolve absolute path: %w", err)
	}
	
	/* Additional check: ensure no traversal in absolute path */
	if strings.Contains(absPath, "..") {
		return fmt.Errorf("resolved path contains traversal pattern")
	}
	
	return nil
}

/* validateResolvedPath validates that a resolved path is within allowed directories */
func (l *ConfigLoader) validateResolvedPath(path string) error {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("failed to resolve path: %w", err)
	}
	
	/* Get allowed base directories */
	cwd, _ := os.Getwd()
	home, _ := os.UserHomeDir()
	
	allowedDirs := []string{
		cwd,
		filepath.Join(cwd, "..", ".."),
	}
	if home != "" {
		allowedDirs = append(allowedDirs, filepath.Join(home, ".neurondb"))
	}
	
	/* Check if path is within any allowed directory */
	for _, allowedDir := range allowedDirs {
		allowedAbs, err := filepath.Abs(allowedDir)
		if err != nil {
			continue
		}
		/* Check if absPath is within allowedAbs */
		rel, err := filepath.Rel(allowedAbs, absPath)
		if err == nil && !strings.HasPrefix(rel, "..") {
			return nil /* Path is within allowed directory */
		}
	}
	
	/* Path not in allowed directories - reject */
	return fmt.Errorf("path not in allowed directories: %s", absPath)
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

  /* HTTP transport config from env */
	if httpEnabled := os.Getenv("NEURONMCP_HTTP_ENABLED"); httpEnabled != "" {
		enabled := httpEnabled == "true" || httpEnabled == "1"
		if merged.Server.HTTPTransport == nil {
			merged.Server.HTTPTransport = &HTTPTransportConfig{}
		}
		merged.Server.HTTPTransport.Enabled = &enabled
	}
	if httpAddr := os.Getenv("NEURONMCP_HTTP_ADDR"); httpAddr != "" {
		if merged.Server.HTTPTransport == nil {
			merged.Server.HTTPTransport = &HTTPTransportConfig{}
		}
		merged.Server.HTTPTransport.Address = &httpAddr
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

