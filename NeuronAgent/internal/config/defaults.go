/*-------------------------------------------------------------------------
 *
 * defaults.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/config/defaults.go
 *
 *-------------------------------------------------------------------------
 */

package config

import (
	"time"
)

/* DefaultConfig returns a configuration with sensible defaults */
func DefaultConfig() *Config {
	return &Config{
		Server: ServerConfig{
			Host:         "0.0.0.0",
			Port:         8080,
			ReadTimeout:  30 * time.Second,
			WriteTimeout: 30 * time.Second,
		},
		Database: DatabaseConfig{
			Host:            "localhost",
			Port:            5432,
			Database:        "neurondb",
			User:            "postgres",
			Password:        "postgres",
			MaxOpenConns:    25,
			MaxIdleConns:    5,
			ConnMaxLifetime: 5 * time.Minute,
			ConnMaxIdleTime: 10 * time.Minute,
		},
		Auth: AuthConfig{
			APIKeyHeader: "Authorization",
		},
		Logging: LoggingConfig{
			Level:  "info",
			Format: "json",
		},
		Workflow: WorkflowConfig{
			BaseURL: "http://localhost:8080",
		},
	}
}

/* DevelopmentConfig returns a configuration optimized for development */
func DevelopmentConfig() *Config {
	cfg := DefaultConfig()
	cfg.Logging.Level = "debug"
	cfg.Logging.Format = "console"
	cfg.Database.MaxOpenConns = 10
	cfg.Database.MaxIdleConns = 2
	return cfg
}

/* ProductionConfig returns a configuration optimized for production */
func ProductionConfig() *Config {
	cfg := DefaultConfig()
	cfg.Logging.Level = "warn"
	cfg.Logging.Format = "json"
	cfg.Database.MaxOpenConns = 100
	cfg.Database.MaxIdleConns = 20
	cfg.Database.ConnMaxLifetime = 15 * time.Minute
	return cfg
}
