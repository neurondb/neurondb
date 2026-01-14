/*-------------------------------------------------------------------------
 *
 * validator.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/config/validator.go
 *
 *-------------------------------------------------------------------------
 */

package config

import "fmt"

/* ConfigValidator validates configuration */
type ConfigValidator struct{}

/* NewConfigValidator creates a new config validator */
func NewConfigValidator() *ConfigValidator {
	return &ConfigValidator{}
}

/* Validate validates the complete server configuration */
func (v *ConfigValidator) Validate(config *ServerConfig) (bool, []string) {
	var errors []string

  /* Validate database config */
	errors = append(errors, v.validateDatabase(&config.Database)...)

  /* Validate server settings */
	errors = append(errors, v.validateServer(&config.Server)...)

  /* Validate logging */
	errors = append(errors, v.validateLogging(&config.Logging)...)

  /* Validate features */
	errors = append(errors, v.validateFeatures(&config.Features)...)

	return len(errors) == 0, errors
}

func (v *ConfigValidator) validateDatabase(config *DatabaseConfig) []string {
	var errors []string

	if config.ConnectionString == nil && config.Host == nil {
		errors = append(errors, "Database configuration must have either connectionString or host")
	}

	if config.Port != nil && (*config.Port < 1 || *config.Port > 65535) {
		errors = append(errors, "Database port must be between 1 and 65535")
	}

	if config.Pool != nil {
		if config.Pool.Min != nil && *config.Pool.Min < 0 {
			errors = append(errors, "Pool min connections must be >= 0")
		}
		if config.Pool.Max != nil && *config.Pool.Max < 1 {
			errors = append(errors, "Pool max connections must be >= 1")
		}
		if config.Pool.Min != nil && config.Pool.Max != nil && *config.Pool.Min > *config.Pool.Max {
			errors = append(errors, "Pool min connections must be <= max connections")
		}
		if config.Pool.IdleTimeoutMillis != nil && *config.Pool.IdleTimeoutMillis < 0 {
			errors = append(errors, "Pool idleTimeoutMillis must be >= 0")
		}
		if config.Pool.ConnectionTimeoutMillis != nil && *config.Pool.ConnectionTimeoutMillis < 0 {
			errors = append(errors, "Pool connectionTimeoutMillis must be >= 0")
		}
	}

	return errors
}

func (v *ConfigValidator) validateServer(config *ServerSettings) []string {
	var errors []string

	if config.Timeout != nil && *config.Timeout < 0 {
		errors = append(errors, "Server timeout must be >= 0")
	}

	if config.MaxRequestSize != nil && *config.MaxRequestSize < 0 {
		errors = append(errors, "Server maxRequestSize must be >= 0")
	}

	return errors
}

func (v *ConfigValidator) validateLogging(config *LoggingConfig) []string {
	var errors []string

	validLevels := []string{"debug", "info", "warn", "error"}
	if !contains(validLevels, config.Level) {
		errors = append(errors, fmt.Sprintf("Logging level must be one of: %v", validLevels))
	}

	validFormats := []string{"json", "text"}
	if !contains(validFormats, config.Format) {
		errors = append(errors, fmt.Sprintf("Logging format must be one of: %v", validFormats))
	}

	return errors
}

func (v *ConfigValidator) validateFeatures(config *FeaturesConfig) []string {
	var errors []string

	if config.Vector != nil && config.Vector.Enabled {
		if config.Vector.MaxVectorDimension != nil && *config.Vector.MaxVectorDimension < 1 {
			errors = append(errors, "Vector maxVectorDimension must be >= 1")
		}
		if config.Vector.DefaultLimit != nil && *config.Vector.DefaultLimit < 1 {
			errors = append(errors, "Vector defaultLimit must be >= 1")
		}
	}

	if config.ML != nil && config.ML.Enabled {
		if config.ML.MaxTrainingTime != nil && *config.ML.MaxTrainingTime < 0 {
			errors = append(errors, "ML maxTrainingTime must be >= 0")
		}
	}

	if config.Analytics != nil && config.Analytics.Enabled {
		if config.Analytics.MaxClusters != nil && *config.Analytics.MaxClusters < 1 {
			errors = append(errors, "Analytics maxClusters must be >= 1")
		}
		if config.Analytics.MaxIterations != nil && *config.Analytics.MaxIterations < 1 {
			errors = append(errors, "Analytics maxIterations must be >= 1")
		}
	}

	if config.RAG != nil && config.RAG.Enabled {
		if config.RAG.DefaultChunkSize != nil && *config.RAG.DefaultChunkSize < 1 {
			errors = append(errors, "RAG defaultChunkSize must be >= 1")
		}
		if config.RAG.DefaultOverlap != nil && *config.RAG.DefaultOverlap < 0 {
			errors = append(errors, "RAG defaultOverlap must be >= 0")
		}
	}

	if config.Projects != nil && config.Projects.Enabled {
		if config.Projects.MaxProjects != nil && *config.Projects.MaxProjects < 1 {
			errors = append(errors, "Projects maxProjects must be >= 1")
		}
	}

	return errors
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

